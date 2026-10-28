"""Own one private MPS service and one bounded eight-worker node job."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid


def control(env, command):
    result = subprocess.run(['nvidia-cuda-mps-control'], input=command + '\n',
                            capture_output=True, text=True, env=env, timeout=15)
    if result.returncode:
        raise RuntimeError('private MPS control failed: ' + result.stderr[-500:])
    return result.stdout.strip()


def terminate_private_clients(env):
    """Never issue global/default-daemon commands or kill unrelated PIDs."""
    for server in control(env, 'get_server_list').splitlines():
        if not server.strip().isdigit():
            continue
        clients = control(env, 'get_client_list ' + server.strip())
        for client in clients.splitlines():
            if client.strip().isdigit():
                control(env, f'terminate_client {server.strip()} {client.strip()}')


def cleanup_private_service(env, proc, service_owned, service_attempted,
                            clients_terminated=False):
    """Reap this run's failed clients before quitting its private MPS service."""
    if not service_owned and not service_attempted:
        # A pre-service failure has no MPS service to stop. Never send quit to
        # the colliding directory, which can belong to a prior/external run.
        return (True if proc is None else None), None, None
    client_cleanup_error = None
    if (service_owned and not clients_terminated and proc is not None
            and proc.returncode not in (None, 0)):
        try:
            terminate_private_clients(env)
        except Exception as error:
            client_cleanup_error = type(error).__name__
    try:
        control(env, 'quit')
    except Exception as error:
        return False, client_cleanup_error, str(error)
    return True, client_cleanup_error, None


def check_name(name):
    if not re.fullmatch(r'v4_[a-z0-9_]{1,60}', name):
        raise ValueError('invalid private run identity')


def prepare_training_environment(env, observe_mode='off', observer_library=None,
                                 stream_delay_library=None):
    """Keep private actor controls out of MPS control and controller processes."""
    from flash_v4.train_entry import (
        PRIVATE_QUOTA_ENV,
        PRIVATE_STREAM_DELAY_ENV,
        PRIVATE_ACTOR_WINDOW_ENV,
        _parse_quota_mapping,
        _parse_stream_delay_mapping,
    )

    quota_raw = env.pop(PRIVATE_QUOTA_ENV, None)
    quotas = _parse_quota_mapping(quota_raw)
    if any(quota != 100 for quota in quotas.values()):
        raise ValueError('shared-node runs require CUDA MPS quota 100 on every rank')
    ratio_raw = env.pop(PRIVATE_STREAM_DELAY_ENV, None)
    window_raw = env.pop(PRIVATE_ACTOR_WINDOW_ENV, None)
    if window_raw is not None and ratio_raw is None:
        raise ValueError('private actor window requires stream-delay actor')
    if ratio_raw is not None:
        _parse_stream_delay_mapping(ratio_raw)
        if stream_delay_library is None or not Path(stream_delay_library).is_file():
            raise ValueError('stream-delay actor requires an existing native library')

    env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '100'
    training_env = env.copy()
    if ratio_raw is not None:
        training_env[PRIVATE_STREAM_DELAY_ENV] = ratio_raw
        if window_raw is not None:
            training_env[PRIVATE_ACTOR_WINDOW_ENV] = window_raw

    preload = []
    if observe_mode != 'off':
        if observer_library is None or not Path(observer_library).is_file():
            raise ValueError('observation requires an existing observer library')
        preload.append(str(Path(observer_library).resolve()))
    if ratio_raw is not None:
        preload.append(str(Path(stream_delay_library).resolve()))
    if preload:
        training_env['LD_PRELOAD'] = os.pathsep.join(preload)
    return env, training_env, ratio_raw is not None


def physical_uuid_order(text):
    devices = {}
    for line in text.splitlines():
        index, identifier = [v.strip() for v in line.split(',')]
        if not identifier.startswith('GPU-'):
            raise ValueError('physical GPU UUID required')
        uuid.UUID(identifier[4:])
        index = int(index)
        if index in devices or identifier in devices.values():
            raise ValueError('duplicate physical GPU identity')
        devices[index] = identifier
    if set(devices) != set(range(8)):
        raise ValueError('expected exactly eight physical GPUs per node')
    return [devices[index] for index in range(8)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--layout', choices=('legacy', 'node_local_pp'), required=True)
    p.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--master-addr', required=True)
    p.add_argument('--master-port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--observe-mode', choices=('off', 'monitor', 'profile', 'online'), default='off')
    p.add_argument('--observer-library', type=Path)
    p.add_argument('--stream-delay-library', type=Path)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--transport', choices=('tcp', 'ib'), default='tcp')
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--execution-budget')
    args = p.parse_args()
    check_name(args.run_id)
    if not 1024 <= args.master_port <= 65535:
        p.error('invalid bounded compatibility run')
    from flash_v4.execution_budget import validate_node_budget
    try:
        execution_budget = validate_node_budget(
            args.execution_budget, args.steps, args.timeout, args.observe_mode,
            actor=bool(os.environ.get('FLASH_V4_STREAM_DELAY_PRIVATE')
                       or os.environ.get('FLASH_V4_ACTOR_WINDOW_PRIVATE')),
            forced=os.environ.get('FLASH_V4_FORCE_PROFILE_AT', '0') != '0')
    except ValueError as error:
        p.error(str(error))
    if not 1 <= args.seq_len <= 2048:
        p.error('sequence length outside bounded calibration range')
    if os.environ.get('LD_PRELOAD'):
        p.error('unexpected preloaded instrumentation')
    owner = Path('/tmp/flash1022_v4_gpu_owner')
    from flash_v4.profile_sampling import profile_stride
    profile_stride(os.environ.get('FLASH_V4_PROFILE_SAMPLE_STRIDE'))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               CUDA_MPS_PIPE_DIRECTORY=f'/tmp/{args.run_id}_mps_pipe',
               CUDA_MPS_LOG_DIRECTORY=f'/tmp/{args.run_id}_mps_log',
               CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',
               CUDA_MPS_CLIENT_PRIORITY='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NCCL_NET='Socket', NCCL_IB_DISABLE='1',
               NCCL_SOCKET_IFNAME='bond0', GLOO_SOCKET_IFNAME='bond0',
               NCCL_CUMEM_ENABLE='0', NCCL_CUMEM_HOST_ENABLE='0',
               NCCL_NVLS_ENABLE='0', NCCL_MAX_NCHANNELS='4', NCCL_BUFFSIZE='1048576',
               NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    if args.transport == 'ib':
        env.update(NCCL_NET='IB', NCCL_IB_DISABLE='0',
                   NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1',
                   NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,NET')
    env['FLASH_V4_ACTUATION_DIR_PRIVATE'] = str(args.output / 'actuation')
    try:
        env, training_env, stream_delay_actor_enabled = prepare_training_environment(
            env, args.observe_mode, args.observer_library, args.stream_delay_library)
    except ValueError as error:
        p.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    if args.observe_mode != 'off':
        observed = args.output / 'observations'
        observed.mkdir(mode=0o700)
        training_env.update(FLASH_V4_OBSERVE_MODE=args.observe_mode,
                            FLASH_V4_PROFILE_START='4', FLASH_V4_SAMPLE_STRIDE=str(profile_stride(os.environ.get('FLASH_V4_PROFILE_SAMPLE_STRIDE'))),
                            FLASH_V4_OUTPUT_DIR=str(observed))
        if args.observe_mode == 'online':
            training_env['FLASH_V4_ONLINE_LOG_DIR'] = str(args.output / 'control')
            if execution_budget is not None:
                training_env['FLASH_V4_EXECUTION_BUDGET'] = execution_budget['name']
    status = {'run_id': args.run_id, 'layout': args.layout, 'node_rank': args.node_rank,
              'role': ('bounded_online_monitor_profile' if args.observe_mode == 'online'
                       else 'bounded_offline_observation_pilot' if args.observe_mode != 'off'
                       else 'shared_16gpu_compatibility_only'), 'clock_changed': False,
              'observation_mode': args.observe_mode,
              'execution_budget': execution_budget,
              'stream_delay_actor_enabled': stream_delay_actor_enabled,
              'transport_request': args.transport, 'sequence_length': args.seq_len,
              'status': 'initializing', 'start_wall_time': time.time()}
    status_path = args.output / 'node_status.json'
    proc, service_owned, service_attempted, owner_acquired = None, False, False, False
    private_clients_terminated = False

    def publish():
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_path)

    def telemetry(stream):
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu,clocks.current.sm',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        stream.write(json.dumps({'wall_time': time.time(), 'returncode': query.returncode,
                                 'gpu_rows': query.stdout.splitlines()}) + '\n')
        stream.flush()

    try:
        # Adopt and reap only this controller's daemon descendants. No global
        # init/service configuration is changed.
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError('cannot supervise private MPS descendants')
        owner.mkdir(mode=0o700)
        owner_acquired = True
        (owner / 'owner.json').write_text(json.dumps({'run_id': args.run_id, 'pid': os.getpid()}))
        physical = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=10, check=True)
        identifiers = physical_uuid_order(physical.stdout)
        # UUIDs avoid CUDA/NVML index-order ambiguity and MPS ordinal remapping.
        env['CUDA_VISIBLE_DEVICES'] = training_env['CUDA_VISIBLE_DEVICES'] = ','.join(identifiers)
        actuation = args.output / 'actuation'
        actuation.mkdir(mode=0o700)
        (actuation / 'physical_binding.json').write_text(json.dumps(
            {'node_rank': args.node_rank, 'local_rank_to_gpu_uuid': identifiers}))
        for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY'):
            Path(env[key]).mkdir(mode=0o700, exist_ok=False)
        publish()
        service_attempted = True
        start = subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env,
                               capture_output=True, text=True, timeout=15)
        if start.returncode:
            raise RuntimeError('private MPS startup failed: ' + start.stderr[-500:])
        service_owned = True
        status['status'] = 'training'
        status['controller_pid'] = os.getpid()
        publish()
        command = ['torchrun', '--nnodes=2', '--nproc-per-node=8',
                   f'--node-rank={args.node_rank}', f'--master-addr={args.master_addr}',
                   f'--master-port={args.master_port}', '--max-restarts=0',
                   '-m', 'flash_v4.train_entry', '--layout', args.layout,
                   '--steps', str(args.steps), '--warmup', '3', '--seq-len', str(args.seq_len),
                   '--seed', str(args.seed), '--model', str(args.model.resolve()),
                   '--output', str(args.output / 'training')]
        with (args.output / 'torchrun.log').open('x') as log, (args.output / 'background.jsonl').open('x') as bg:
            proc = subprocess.Popen(command, env=training_env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            deadline = time.monotonic() + args.timeout
            while proc.poll() is None and time.monotonic() < deadline:
                telemetry(bg)
                try:
                    servers = [s.strip() for s in control(env, 'get_server_list').splitlines()
                               if s.strip().isdigit()]
                    client_count = sum(sum(c.strip().isdigit() for c in
                                           control(env, 'get_client_list ' + s).splitlines())
                                       for s in servers)
                    status['maximum_private_mps_clients_seen'] = max(
                        status.get('maximum_private_mps_clients_seen', 0), client_count)
                except Exception:
                    # A server may exit between the list and client queries.
                    status['mps_observation_errors'] = status.get('mps_observation_errors', 0) + 1
                time.sleep(2)
            if proc.poll() is None:
                status['status'] = 'timeout'
                terminate_private_clients(env)
                private_clients_terminated = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
            else:
                status['status'] = 'completed' if proc.returncode == 0 else 'failed'
            telemetry(bg)
            status['returncode'] = proc.returncode
            if args.observe_mode == 'online' and proc.returncode == 0:
                final = []
                for path in sorted((args.output / 'control').glob('online_rank*.jsonl')):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    final.extend(r for r in rows if r.get('kind') == 'monitor_end')
                valid = len(final) == 8 and all(r.get('monitor_status') in (
                    'completed', 'peer_completed', 'manual_integration') for r in final)
                status['online_monitor_integrity'] = valid
                status['online_monitor_end_records'] = len(final)
                status['online_profile_ready_ranks'] = len(list(
                    (args.output / 'control').glob('ready_rank*.json')))
                if not valid:
                    status['status'] = 'failed'
                    status['error'] = 'online monitor integrity failed; retain worker/control evidence'
    except BaseException as error:
        status['status'] = 'failed'
        status['error'] = type(error).__name__ + ': ' + str(error)
        if proc is not None and proc.poll() is None:
            try:
                if service_owned:
                    terminate_private_clients(env)
                    private_clients_terminated = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
            except Exception as cleanup_error:
                status['training_cleanup_error'] = type(cleanup_error).__name__
    finally:
        stopped, client_error, mps_error = cleanup_private_service(
            env, proc, service_owned, service_attempted, private_clients_terminated)
        if stopped is not None:
            status['private_mps_stopped'] = stopped
        if client_error:
            status['private_client_cleanup_error'] = client_error
        if mps_error:
            status['mps_cleanup_error'] = mps_error
        status['end_wall_time'] = time.time()
        reap_deadline = time.monotonic() + 1
        while time.monotonic() < reap_deadline:
            try:
                child, _ = os.waitpid(-1, os.WNOHANG)
                if child == 0:
                    time.sleep(.02)
            except ChildProcessError:
                break
        publish()
        if owner_acquired:
            owner_record = json.loads((owner / 'owner.json').read_text())
            if owner_record['run_id'] == args.run_id and owner_record['pid'] == os.getpid():
                # Preserve the lock when cleanup is uncertain; require review.
                if (status.get('private_mps_stopped') is not False
                        and not status.get('training_cleanup_error')
                        and not status.get('private_client_cleanup_error')):
                    (owner / 'owner.json').unlink()
                    owner.rmdir()
        print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'completed' and status.get('private_mps_stopped') else 1


if __name__ == '__main__':
    raise SystemExit(main())
