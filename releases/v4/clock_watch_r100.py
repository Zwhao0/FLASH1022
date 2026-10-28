"""R66 opt-in rank-progress host-side bounded clock injection; invoked only by the root-owned clock runner.

One persistent docker stream supplies training progress. A separate detached
watchdog restores the recorded original SM frequency even if this process dies.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import selectors
import signal
import shlex
import subprocess
import sys
import time


STREAM = "\nimport json,pathlib,time\nfrom flash_v4.rank_progress import validate_receipt,MAX_BYTES\nroot=pathlib.Path(ROOT)\nstarted=time.monotonic()\ndeadline=started+260\nlast=0\nlast_change=started\nlast_stamp=-1\nbound=False\nwhile time.monotonic()<deadline:\n try:\n  p=root/'actuation/physical_binding.json'\n  if p.exists() and not bound:\n   print(json.dumps(dict(binding=json.loads(p.read_text()))),flush=True);bound=True\n  p=root/'node_status.json'\n  if p.exists():\n   s=json.loads(p.read_text())\n   if s.get('status') in ('completed','timeout','failed'):\n    print(json.dumps(dict(terminal=s)),flush=True);break\n  p=root/'control/progress'/('rank'+str(TARGET_RANK)+'.json')\n  if p.exists():\n   with p.open() as f: raw=f.read(MAX_BYTES+1)\n   if len(raw.encode())>MAX_BYTES: raise ValueError('oversized progress receipt')\n   r=json.loads(raw)\n   if r.get('status')=='closed':\n    # Validate identity/freshness before accepting a terminal writer receipt.\n    candidate=dict(r,status='running')\n    validate_receipt(candidate,run_id=RUN_ID,rank=TARGET_RANK,now=time.monotonic(),previous=last)\n    print(json.dumps(dict(terminal=dict(status='progress_writer_closed'))),flush=True);break\n   i=validate_receipt(r,run_id=RUN_ID,rank=TARGET_RANK,now=time.monotonic(),previous=last)\n   if r['monotonic_seconds']<last_stamp: raise ValueError('nonmonotonic progress timestamp')\n   last_stamp=r['monotonic_seconds']\n   if i>last:\n    print(json.dumps(dict(iteration=i,progress_rank=TARGET_RANK,progress_run_id=RUN_ID,\n                         progress_node_rank=TARGET_RANK//8,progress_monotonic_seconds=last_stamp)),flush=True)\n    last=i;last_change=time.monotonic()\n   elif time.monotonic()-last_change>5: raise ValueError('stalled progress')\n  elif last or time.monotonic()-started>180:\n   raise ValueError('missing progress receipt')\n except Exception as e:\n  print(json.dumps(dict(progress_error=type(e).__name__+': '+str(e))),flush=True);break\n time.sleep(.1)\n"


def validate_config(cfg):
    if cfg.get('progress_contract') != 'r66_rank_schedule_v1':
        raise ValueError('explicit rank progress contract required')
    import re
    if not re.fullmatch(r'v4_[a-f0-9]{32}', cfg.get('progress_run_id', '')):
        raise ValueError('invalid progress run identity')
    gpu = cfg.get('gpu_uuid')
    original, lowered = cfg.get('original_sm_mhz'), cfg.get('lowered_sm_mhz')
    target_rank = cfg.get('target_rank')
    local_world = cfg.get('local_world_size', 8)
    node_count = cfg.get('node_count', 2)
    first, last = cfg.get('first'), cfg.get('last')
    max_lowered = cfg.get('maximum_lowered_seconds', 35)
    watchdog_delay = cfg.get('emergency_restore_seconds', 42)
    if not (isinstance(gpu, str) and gpu.startswith('GPU-')
            and type(original) is int and original == 1980
            and type(lowered) is int and 300 <= lowered < original
            and type(target_rank) is int
            and type(local_world) is int and local_world > 0
            and type(node_count) is int and node_count > 0
            and 0 <= target_rank < local_world * node_count
            and type(first) is int and first >= 34
            and type(last) is int and last == first + 31
            and type(max_lowered) in (int, float) and 0 < max_lowered
            and type(watchdog_delay) in (int, float) and watchdog_delay > max_lowered):
        raise ValueError('invalid bounded clock configuration')
    expected_node, local_rank = divmod(target_rank, local_world)
    configured_node = cfg.get('logical_node_rank', expected_node)
    if type(configured_node) is not int or configured_node != expected_node:
        raise ValueError('target global rank disagrees with configured logical node')
    prefix = cfg.get('privilege_prefix', [])
    if prefix not in ([], ['sudo', '-n']):
        raise ValueError('invalid privilege prefix')
    command = cfg.get('container_command')
    if not isinstance(command, list) or not command or any(not isinstance(part, str) for part in command):
        raise ValueError('invalid container command')
    if not isinstance(cfg.get('container_output'), str) or not cfg['container_output']:
        raise ValueError('invalid container output path')
    if not isinstance(cfg.get('host_output'), str) or not cfg['host_output']:
        raise ValueError('invalid host output path')
    return dict(gpu=gpu, original=original, lowered=lowered, target_rank=target_rank,
                local_world_size=local_world, node_count=node_count,
                expected_node_rank=expected_node, local_rank=local_rank,
                first=first, last=last, maximum_lowered_seconds=max_lowered,
                emergency_restore_seconds=watchdog_delay, prefix=prefix)


def verify_physical_binding(binding, target_rank, gpu, local_world_size=8, node_count=2):
    expected_node, local_rank = divmod(target_rank, local_world_size)
    mapping = binding.get('local_rank_to_gpu_uuid') if isinstance(binding, dict) else None
    node_rank = binding.get('node_rank') if isinstance(binding, dict) else None
    reasons = []
    if type(node_rank) is not int or node_rank != expected_node:
        reasons.append('logical_node_rank_mismatch')
    if not isinstance(mapping, list) or len(mapping) != local_world_size:
        reasons.append('local_rank_map_shape_mismatch')
        observed_gpu = None
    else:
        observed_gpu = mapping[local_rank]
        if observed_gpu != gpu:
            reasons.append('local_rank_gpu_uuid_mismatch')
    if not 0 <= target_rank < local_world_size * node_count:
        reasons.append('global_rank_out_of_range')
    return dict(verified=not reasons, target_rank=target_rank,
                expected_node_rank=expected_node, observed_node_rank=node_rank,
                local_rank=local_rank, expected_gpu_uuid=gpu,
                observed_gpu_uuid=observed_gpu, reasons=reasons)


def watchdog_source(restore, receipt_path, delay_seconds, attempts=5):
    """Build an independent bounded retrying restore process."""
    return '\n'.join((
        'import json,pathlib,subprocess,time',
        f'time.sleep({float(delay_seconds)!r})',
        f'command={restore!r}',
        f'receipt=pathlib.Path({str(receipt_path)!r})',
        'results=[]',
        f'for index in range({int(attempts)}):',
        ' try:',
        '  r=subprocess.run(command,capture_output=True,text=True,timeout=10)',
        '  item=dict(attempt=index+1,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)',
        ' except BaseException as e:',
        '  item=dict(attempt=index+1,returncode=None,stdout="",stderr=type(e).__name__+": "+str(e))',
        ' results.append(item)',
        ' if item["returncode"] == 0: break',
        f' if index+1 < {int(attempts)}: time.sleep(2)',
        'result=dict(attempts=results,restored=bool(results and results[-1]["returncode"]==0),wall_time=time.time())',
        'try:',
        ' tmp=receipt.with_name(receipt.name+".tmp")',
        ' tmp.write_text(json.dumps(result))',
        ' tmp.replace(receipt)',
        'except BaseException:',
        ' pass'))


def run(config, *, command_runner=None, popen_factory=None, selector_factory=None,
        monotonic=None, wall_clock=None):
    cfg = config if isinstance(config, dict) else json.loads(Path(config).read_text())
    values = validate_config(cfg)
    gpu, original, lowered = values['gpu'], values['original'], values['lowered']
    first, planned_last = values['first'], values['last']
    prefix = values['prefix']
    command_runner = command_runner or subprocess.run
    popen_factory = popen_factory or subprocess.Popen
    selector_factory = selector_factory or selectors.DefaultSelector
    monotonic = monotonic or time.monotonic
    wall_clock = wall_clock or time.time
    restore = prefix + ['nvidia-smi', '-i', gpu, '-lgc', f'{original},{original}']
    lower = prefix + ['nvidia-smi', '-i', gpu, '-lgc', f'{lowered},{lowered}']
    query = ['nvidia-smi', '-i', gpu,
             '--query-gpu=uuid,clocks.current.sm,clocks.applications.graphics,utilization.gpu',
             '--format=csv,noheader,nounits']
    output = Path(cfg['host_output'])
    output.mkdir(parents=True, exist_ok=False)
    journal = (output / 'clock_events.jsonl').open('x', buffering=1)
    journal_errors = []

    def event(kind, **fields):
        try:
            journal.write(json.dumps(dict(kind=kind, wall_time=wall_clock(), **fields)) + '\n')
        except Exception as exc:
            journal_errors.append(type(exc).__name__ + ': ' + str(exc))

    def command(args):
        try:
            result = command_runner(args, capture_output=True, text=True, timeout=10)
            value = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except BaseException as exc:
            value = dict(returncode=None, stdout='', stderr=type(exc).__name__ + ': ' + str(exc),
                         exception=type(exc).__name__)
        event('command', argv=args, result=value)
        return value

    progress = cfg['container_command'] + ['python3', '-u', '-c',
                    'ROOT=' + repr(cfg['container_output']) + '\nTARGET_RANK=' + repr(values['target_rank'])
                    + '\nRUN_ID=' + repr(cfg['progress_run_id']) + '\n' + STREAM]
    watchdog, stream, selector, stderr_file = None, None, None, None
    changed, restored, bound, start, last_iteration, error = False, False, False, None, -1, None
    binding_details = None
    previous_sigterm = None
    shutdown_requested = False

    def request_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    try:
        previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
        initial = command(query)
        event('initial_query', value=initial)
        query_uuids = []
        if initial['returncode'] == 0:
            try:
                query_uuids = [row[0].strip() for row in csv.reader(initial['stdout'].splitlines()) if row]
            except (csv.Error, IndexError):
                query_uuids = []
        if initial['returncode'] != 0 or gpu not in query_uuids:
            raise RuntimeError('initial GPU UUID query failed or selected UUID is not visible on this host')

        stderr_file = (output / 'stream_stderr.log').open('x')
        stream = popen_factory(progress, stdout=subprocess.PIPE, stderr=stderr_file,
                               text=True, bufsize=1)
        selector = selector_factory()
        selector.register(stream.stdout, selectors.EVENT_READ)
        deadline = monotonic() + 270
        while monotonic() < deadline and not shutdown_requested:
            if changed and not restored and monotonic() - start >= values['maximum_lowered_seconds']:
                result = command(restore)
                restored = result['returncode'] == 0
                event('restored_fault_budget', iteration=last_iteration, restored=restored,
                      maximum_lowered_seconds=values['maximum_lowered_seconds'])
                if not restored:
                    raise RuntimeError('clock restore failed at the bounded fault deadline')
            ready = selector.select(timeout=.2)
            if not ready:
                if stream.poll() is not None:
                    break
                continue
            line = stream.stdout.readline()
            if not line:
                break
            row = json.loads(line)
            if 'progress_error' in row:
                raise RuntimeError('rank progress rejected: ' + str(row['progress_error']))
            if 'binding' in row:
                binding_details = verify_physical_binding(
                    row['binding'], values['target_rank'], gpu,
                    values['local_world_size'], values['node_count'])
                if cfg.get('logical_node_rank', values['expected_node_rank']) != values['expected_node_rank']:
                    binding_details['verified'] = False
                    binding_details['reasons'].append('configured_logical_node_rank_mismatch')
                bound = binding_details['verified']
                event('physical_binding', **binding_details, binding=row['binding'],
                      transport_helper=cfg.get('transport_helper'))
                if not bound:
                    raise RuntimeError('physical rank/host mapping mismatch')
            if 'iteration' in row:
                stamp = row.get('progress_monotonic_seconds')
                if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                        or not 0 <= monotonic() - stamp <= 3):
                    raise RuntimeError('host rejected stale or future progress')
                if (type(row.get('iteration')) is not int or not 1 <= row['iteration'] <= 512
                        or row['iteration'] <= last_iteration
                        or row.get('progress_rank') != values['target_rank']
                        or row.get('progress_node_rank') != values['expected_node_rank']
                        or row.get('progress_run_id') != cfg['progress_run_id']):
                    raise RuntimeError('host progress identity or monotonicity mismatch')
                last_iteration = row['iteration']
                event('progress', **row)
                if bound and not changed and first - 1 <= last_iteration < planned_last:
                    # The process is detached from the driver and SSH parent.
                    receipt_path = output / 'watchdog_restore.json'
                    rescue_code = watchdog_source(restore, receipt_path,
                                                  values['emergency_restore_seconds'])
                    watchdog = popen_factory([sys.executable, '-c', rescue_code],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True)
                    # Mark changed before issuing the command so every failure
                    # path attempts restoration, including command timeouts.
                    changed, start = True, monotonic()
                    result = command(lower)
                    event('lowered', after_iteration=last_iteration, requested_mhz=lowered,
                          success=result['returncode'] == 0, query=command(query),
                          target_rank=values['target_rank'], expected_node_rank=values['expected_node_rank'],
                          local_rank=values['local_rank'])
                    if result['returncode'] != 0:
                        raise RuntimeError('clock lowering failed')
                if changed and not restored and last_iteration >= planned_last:
                    restored = command(restore)['returncode'] == 0
                    event('restored_iteration', after_iteration=last_iteration, restored=restored)
                    if not restored:
                        raise RuntimeError('clock restore failed at the planned interval end')
                    # The full fault interval is complete. Further training progress
                    # cannot change qualification and may legitimately become idle.
                    event('fault_interval_finished', after_iteration=last_iteration, restored=True)
                    break
            if 'terminal' in row:
                event('training_terminal', status=row['terminal'])
                break
    except BaseException as exc:
        if shutdown_requested and isinstance(exc, InterruptedError):
            event('shutdown_requested', interrupted=type(exc).__name__)
        else:
            error = type(exc).__name__ + ': ' + str(exc)
            event('error', error=error)
    finally:
        if changed:
            final_restore = command(restore)
            restored = final_restore['returncode'] == 0
            event('final_restore', restored=restored, after_iteration=last_iteration,
                  result=final_restore,
                  query=command(query))
            if not restored and error is None:
                error = 'RuntimeError: final clock restore failed'

        cleanup_errors = []
        if restored and watchdog is not None:
            try:
                if watchdog.poll() is None:
                    watchdog.terminate()
                watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    watchdog.kill()
                    watchdog.wait(timeout=5)
                except BaseException as exc:
                    cleanup_errors.append('watchdog_kill_' + type(exc).__name__ + ': ' + str(exc))
            except BaseException as exc:
                cleanup_errors.append('watchdog_stop_' + type(exc).__name__ + ': ' + str(exc))
        if stream is not None:
            try:
                if stream.poll() is None:
                    stream.terminate()
                    try:
                        stream.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        stream.kill()
                        stream.wait(timeout=5)
            except BaseException as exc:
                cleanup_errors.append('stream_stop_' + type(exc).__name__ + ': ' + str(exc))
        if selector is not None:
            try:
                if stream is not None and stream.stdout is not None:
                    selector.unregister(stream.stdout)
            except BaseException as exc:
                cleanup_errors.append('selector_unregister_' + type(exc).__name__ + ': ' + str(exc))
            try:
                selector.close()
            except BaseException as exc:
                cleanup_errors.append('selector_close_' + type(exc).__name__ + ': ' + str(exc))
        if stream is not None and stream.stdout is not None:
            try:
                stream.stdout.close()
            except BaseException as exc:
                cleanup_errors.append('stream_stdout_close_' + type(exc).__name__ + ': ' + str(exc))
        if stderr_file is not None:
            try:
                stderr_file.close()
            except BaseException as exc:
                cleanup_errors.append('stream_stderr_close_' + type(exc).__name__ + ': ' + str(exc))

        if journal_errors and error is None:
            error = 'OSError: event journal write failed'
        if cleanup_errors and error is None:
            error = 'RuntimeError: process cleanup failed'
        result = dict(progress_contract=cfg['progress_contract'], progress_run_id=cfg['progress_run_id'],
                      progress_source='target_rank_runtime_schedule_end_snapshot', changed=changed, restored=restored, physical_binding_verified=bound,
                      physical_binding=binding_details, target_rank=values['target_rank'],
                      logical_node_rank=values['expected_node_rank'], local_rank=values['local_rank'],
                      first=first, last=planned_last,
                      maximum_lowered_seconds=values['maximum_lowered_seconds'],
                      emergency_restore_seconds=values['emergency_restore_seconds'],
                      last_iteration=last_iteration, error=error,
                      shutdown_requested=shutdown_requested, cleanup_errors=cleanup_errors,
                      journal_errors=journal_errors, original_sm_mhz=original,
                      lowered_sm_mhz=lowered, gpu_uuid=gpu,
                      restoration_policy='restore fixed original 1980 MHz SM lock; pre-existing lock range is not queryable',
                      independent_watchdog_armed=watchdog is not None,
                      independent_watchdog_left_armed=watchdog is not None and not restored,
                      finished_wall_time=wall_clock())
        try:
            (output / 'clock_result.json').write_text(json.dumps(result, indent=2))
        except BaseException as exc:
            if error is None:
                error = 'clock_result_write_' + type(exc).__name__ + ': ' + str(exc)
        try:
            journal.close()
        except BaseException:
            pass
        if previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
            except BaseException:
                pass
    return 0 if changed and restored and error is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    raise SystemExit(run(parser.parse_args().config))
