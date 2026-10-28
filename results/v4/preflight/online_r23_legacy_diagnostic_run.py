"""Run the bounded R23 legacy sham and low-intensity diagnostic."""
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import shlex
import subprocess
import sys
import time

from flash_v4 import online_evaluate
from flash_v4 import online_harness as harness
from flash_v4.episode_qualification import qualify
from flash_v4.frozen_localizer import verify
from flash_v4.health_centered_fusion import fingerprint, write_new


PLAN = Path('results/v4/preflight/online_r23_legacy_diagnostic_plan.json')
MAX_SESSIONS = 2
MAX_SAMPLE_GROUPS = 40
MAX_TELEMETRY_BYTES = 4 * 1024 * 1024
SAMPLE_INTERVAL_SECONDS = 6
DRIVER_WAIT_SECONDS = 300

REMOTE_SENSOR_CODE = r'''import json, subprocess, time

def text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)

def run(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"returncode": None, "stdout": text(error.stdout),
                "stderr": text(error.stderr), "timed_out": True,
                "exception": "TimeoutExpired"}
    except Exception as error:
        return {"returncode": None, "stdout": "", "stderr": type(error).__name__,
                "timed_out": False, "exception": type(error).__name__}

try:
    loadavg = {"returncode": 0, "stdout": open("/proc/loadavg").read().strip(),
               "stderr": "", "timed_out": False}
except Exception as error:
    loadavg = {"returncode": None, "stdout": "", "stderr": type(error).__name__,
               "timed_out": False, "exception": type(error).__name__}

record = {
    "time": time.time(),
    "loadavg": loadavg,
    "gpu": run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu,clocks.current.sm",
                "--format=csv,noheader,nounits"]),
    "apps": run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"]),
    "pmon": run(["nvidia-smi", "pmon", "-c", "1", "-s", "um"]),
}
print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
'''


def precheck(plan, label):
    gate = plan['environment_contract']
    records = []
    for sample in range(gate['samples']):
        if sample:
            time.sleep(gate['sample_interval_seconds'])
        for node in (0, 1):
            code = ('import json,pathlib,subprocess; '
                    'q=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu",'
                    '"--format=csv,noheader,nounits"],capture_output=True,text=True); '
                    'p=subprocess.run(["ps","-eo","stat,comm"],capture_output=True,text=True); '
                    'print(json.dumps(dict(owner_exists=pathlib.Path("/tmp/flash1022_v4_gpu_owner").exists(),'
                    'query_returncode=q.returncode,gpu=q.stdout.splitlines(),'
                    'active_mps=[s for s in p.stdout.splitlines() if "nvidia-cuda-mps" in s '
                    'and not s.lstrip().startswith("Z")],process_returncode=p.returncode)))')
            data = json.loads(harness.transport(node, harness.container_command(
                node, 'python3 -c ' + shlex.quote(code), working_directory=harness.BASE)))
            records.append(dict(node=node, sample=sample, wall_time=time.time(), **data))
    passed = True
    for record in records:
        rows = [r.split(',') for r in record['gpu']]
        passed = passed and (record['query_returncode'] == record['process_returncode'] == 0
            and not record['owner_exists'] and not record['active_mps'] and len(rows) == 8
            and {int(r[0]) for r in rows} == set(range(8))
            and all(int(r[1]) >= gate['min_free_mib'] and int(r[2]) <= gate['max_utilization_percent']
                    for r in rows))
    stamp = time.time_ns()
    path = Path(f'results/v4/preflight/r23_environment_{label}_{stamp}.json')
    write_new(path, dict(passed=passed, records=records, gate=gate,
                         plan=fingerprint(PLAN), source=fingerprint(Path(__file__))))
    print(json.dumps(dict(environment_passed=passed, evidence=str(path))), flush=True)
    return passed


def _remote_sample(node, target):
    code = 'python3 -c ' + shlex.quote(REMOTE_SENSOR_CODE)
    command = (code if target == 'host' else
               harness.container_command(node, code, working_directory=harness.BASE))
    try:
        raw = harness.transport(node, command)
    except Exception as error:
        return dict(node=node, target=target, transport_ok=False,
                    transport_returncode=None, transport_stdout='',
                    transport_stderr=str(error), transport_exception=type(error).__name__)
    try:
        sample = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return dict(node=node, target=target, transport_ok=False,
                    transport_returncode=None, transport_stdout=raw,
                    transport_stderr='invalid_sensor_json')
    if not isinstance(sample, dict):
        return dict(node=node, target=target, transport_ok=False,
                    transport_returncode=None, transport_stdout=raw,
                    transport_stderr='sensor_result_not_object')
    return dict(node=node, target=target, transport_ok=True, **sample)


def _sample_failed(result):
    if not result.get('transport_ok'):
        return True
    sample_time = result.get('time')
    if type(sample_time) not in (int, float):
        return True
    for key in ('loadavg', 'gpu', 'apps', 'pmon'):
        command = result.get(key)
        if (not isinstance(command, dict) or command.get('returncode') != 0
                or command.get('timed_out')):
            return True
    return False


def _terminate_local_driver(process):
    """Stop only the local online_driver controller; remote cleanup stays unconfirmed."""
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def run_driver_with_telemetry(case, plan, root):
    telemetry_path = root / f'telemetry_{case}.jsonl'
    driver_log_path = root / f'r23_{case}_driver.log'
    command = [sys.executable, '-m', 'flash_v4.online_driver',
               '--plan', str(PLAN), '--case', case]
    telemetry_failed = False
    telemetry_reason = None
    telemetry_limit = False
    groups_written = 0
    bytes_written = 0
    timed_out = False
    current_results = {}
    pending = {}
    sample_started = None
    sampling_enabled = True
    next_sample = None

    with driver_log_path.open('x', buffering=1) as driver_log, \
            telemetry_path.open('x', buffering=1) as telemetry_log:
        process = subprocess.Popen(command, stdout=driver_log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        deadline = started + DRIVER_WAIT_SECONDS
        next_sample = started
        with ThreadPoolExecutor(max_workers=4) as pool:
            while True:
                now = time.monotonic()
                if process.poll() is None and now >= deadline and not timed_out:
                    timed_out = True
                    _terminate_local_driver(process)

                if pending:
                    done, _ = wait(tuple(pending), timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        key = pending.pop(future)
                        try:
                            current_results[key] = future.result()
                        except Exception as error:
                            current_results[key] = dict(node=key[1], target=key[0],
                                transport_ok=False, transport_returncode=None,
                                transport_stdout='', transport_stderr=type(error).__name__)
                    if not pending:
                        records = [current_results[key] for key in sorted(current_results)]
                        group = dict(sample_group=groups_written + 1,
                                     local_wall_time=time.time(), records=records)
                        line = json.dumps(group, ensure_ascii=False, separators=(',', ':')) + '\n'
                        encoded_size = len(line.encode('utf-8'))
                        if groups_written >= MAX_SAMPLE_GROUPS:
                            telemetry_limit = True
                            telemetry_reason = 'sample_group_limit'
                            sampling_enabled = False
                        elif bytes_written + encoded_size > MAX_TELEMETRY_BYTES:
                            telemetry_limit = True
                            telemetry_reason = 'telemetry_byte_limit'
                            sampling_enabled = False
                        else:
                            try:
                                telemetry_log.write(line)
                                telemetry_log.flush()
                            except OSError as error:
                                telemetry_failed = True
                                telemetry_reason = 'telemetry_write_' + type(error).__name__
                                sampling_enabled = False
                            else:
                                groups_written += 1
                                bytes_written += encoded_size
                                failed = [r for r in records if _sample_failed(r)]
                                if failed:
                                    telemetry_failed = True
                                    telemetry_reason = 'remote_sample_unavailable_or_failed'
                                    sampling_enabled = False
                        current_results = {}
                        pending = {}
                    continue

                returncode = process.poll()
                if returncode is not None:
                    break
                now = time.monotonic()
                if sampling_enabled and now >= next_sample:
                    if groups_written >= MAX_SAMPLE_GROUPS:
                        telemetry_limit = True
                        telemetry_reason = 'sample_group_limit'
                        sampling_enabled = False
                    else:
                        sample_started = now
                        for target in ('host', 'container'):
                            for node in (0, 1):
                                future = pool.submit(_remote_sample, node, target)
                                pending[future] = (target, node)
                        next_sample = sample_started + SAMPLE_INTERVAL_SECONDS
                        continue
                time.sleep(0.1)

        returncode = process.poll()

    if groups_written == 0 and not timed_out:
        telemetry_failed = True
        telemetry_reason = telemetry_reason or 'no_samples_during_driver'
    return dict(returncode=returncode, timed_out=timed_out,
                cleanup_unconfirmed=timed_out, telemetry_failed=telemetry_failed,
                telemetry_limit=telemetry_limit, telemetry_reason=telemetry_reason,
                telemetry_path=str(telemetry_path), driver_log=str(driver_log_path),
                sample_groups=groups_written, telemetry_bytes=bytes_written)


def _write_sham_gate(root, case, truth, plan, quality, actuation):
    timing = root / case / 'final/node0/training/train.jsonl'
    try:
        rows = [json.loads(line) for line in timing.read_text().splitlines() if line.strip()]
        result = qualify(rows, truth['window'], plan['sham_contract'], quality.get('window'))
    except Exception as error:
        result = dict(qualified=False, reasons=['timing_qualification_' + type(error).__name__])
    record = dict(case=case, **result, actuation_quality=actuation,
                  timing=str(timing), timing_fingerprint=fingerprint(timing) if timing.exists() else None,
                  profile_window=quality.get('window'))
    write_new(root / 'sham_gate.json', record)
    return result


def _verify_plan(plan_hash):
    if fingerprint(PLAN) != plan_hash:
        raise ValueError('public plan changed during R23 run')


def _verify_sources(plan):
    verify(plan['source_files'] + plan['training_sources_before'])


def main():
    plan = json.loads(PLAN.read_text())
    plan_hash = fingerprint(PLAN)
    _verify_sources(plan)
    private_path = Path(plan['private_execution'])
    if fingerprint(private_path) != plan['private_execution_receipt']:
        raise ValueError('private execution plan changed')
    private_truth = json.loads(private_path.read_text())['cases']
    sessions = plan['development_GPU_sessions']
    if len(sessions) != MAX_SESSIONS:
        raise ValueError('R23 requires exactly two planned sessions')
    if any(row['topology'] != 'legacy' for row in sessions):
        raise ValueError('R23 diagnostic is restricted to legacy topology')
    if (private_truth[sessions[0]['case']]['target_rank'] is not None
            or private_truth[sessions[0]['case']]['ratio'] != 0
            or private_truth[sessions[1]['case']]['target_rank'] != 6
            or private_truth[sessions[1]['case']]['ratio'] != 4):
        raise ValueError('R23 requires zero-delay sham before the rank-6 ratio-4 case')

    harness.configure_plan(PLAN)
    root = harness.LOCAL
    if root.exists():
        raise FileExistsError('release already exists; inspect evidence, no automatic retry')
    _verify_plan(plan_hash)
    if not precheck(plan, 'before_stage'):
        print('GPU diagnostic deferred: environment gate failed; no dispatch', flush=True)
        return 3
    _verify_plan(plan_hash)
    _verify_sources(plan)
    harness.stage()
    harness.build_actor()
    write_new(root / 'r23_batch_receipt.json', dict(
        plan=plan_hash, owner='root', runner=fingerprint(Path(__file__)),
        max_sessions=MAX_SESSIONS, started_wall_time=time.time()))

    completed = []
    case_runs = []
    for index, row in enumerate(sessions):
        case = row['case']
        if not precheck(plan, case):
            write_new(root / 'r23_batch_stopped.json', dict(
                completed=completed, next_case=case, reason='environment_gate_failed',
                no_more_dispatch=True))
            return 3
        _verify_sources(plan)
        _verify_plan(plan_hash)
        receipt = json.loads((root / 'r23_batch_receipt.json').read_text())
        if receipt['plan'] != plan_hash:
            raise ValueError('R23 batch receipt plan fingerprint changed')

        run = run_driver_with_telemetry(case, plan, root)
        case_runs.append(dict(case=case, **run))
        result_path = root / ('driver_' + case) / 'result.json'
        quality = json.loads(result_path.read_text()) if result_path.exists() else {}
        driver_passed = (not run['timed_out'] and run['returncode'] == 0
                         and quality.get('quality_passed') and quality.get('nodes_completed')
                         and not quality.get('error'))
        if not driver_passed:
            write_new(root / 'r23_batch_failed.json', dict(
                case=case, completed=completed, no_more_dispatch=True,
                driver_evidence=str(result_path), run=run,
                cleanup_unconfirmed=run['cleanup_unconfirmed']))
            return 1

        truth = private_truth[case]
        try:
            actuation = online_evaluate.actuation_quality(
                root / case / 'final', truth, row['steps'], budget=plan['actor_budget'])
        except Exception as error:
            actuation = dict(passed=False, errors=1,
                             error='actuation_quality_' + type(error).__name__)
        if not actuation['passed']:
            if truth['target_rank'] is None:
                write_new(root / 'sham_gate.json', dict(
                    case=case, qualified=False, reasons=['actuation_quality_failed'],
                    actuation_quality=actuation, driver_evidence=str(result_path)))
            write_new(root / 'r23_batch_failed.json', dict(
                case=case, completed=completed, no_more_dispatch=True,
                reason='actuation_quality_failed', actuation_quality=actuation,
                driver_evidence=str(result_path), run=run))
            return 1

        completed.append(case)
        if truth['target_rank'] is None:
            qualification = _write_sham_gate(root, case, truth, plan, quality, actuation)
            if not qualification.get('qualified'):
                write_new(root / 'r23_batch_stopped.json', dict(
                    completed=completed, next_case=sessions[index + 1]['case'],
                    reason='legacy_sham_qualification_failed',
                    sham_gate=str(root / 'sham_gate.json'), no_more_dispatch=True))
                return 1

        if run['telemetry_failed'] or run['telemetry_limit']:
            if index + 1 < len(sessions):
                write_new(root / 'r23_batch_stopped.json', dict(
                    completed=completed, next_case=sessions[index + 1]['case'],
                    reason=run['telemetry_reason'] or 'telemetry_failed',
                    telemetry_path=run['telemetry_path'], no_more_dispatch=True))
                return 1

    write_new(root / 'r23_batch_result.json', dict(
        completed=completed, total_sessions=len(completed), all_drivers_passed=True,
        case_runs=case_runs,
        telemetry_files=[str(root / f'telemetry_{row["case"]}.jsonl') for row in sessions],
        accuracy_claimed=False, finished_wall_time=time.time()))
    return subprocess.run([sys.executable, '-m', 'flash_v4.online_evaluate',
                           '--plan', str(PLAN)], check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
