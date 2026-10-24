"""One isolated node of a two-node acquisition session (launched over SSH)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess as sp
import sys
import time


def online_finish_barrier(client, session, node_rank, local_drained=False, timeout=30, check=None):
    """Keep Redis/drains alive until controller work and both final drains finish.

    Node one makes no Redis requests after publishing its final drain ACK;
    consequently node zero can safely close its private server after both ACKs.
    """
    prefix = f'online_finish:{session}:'
    deadline = time.monotonic() + timeout
    def wait_for(keys):
        while not all(client.exists(key) for key in keys):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'online finish barrier missing {keys}')
            if check is not None:
                check()
            time.sleep(.1)
    if not local_drained:
        if node_rank == 0:
            client.set(prefix + 'controller', '1')
        else:
            wait_for([prefix + 'controller'])
    else:
        client.set(prefix + f'drain_node{node_rank}', '1')
        if node_rank == 0:
            wait_for([prefix + f'drain_node{node}' for node in range(2)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--node-rank', type=int, required=True)
    p.add_argument('--master-addr', default='10.10.4.2')
    p.add_argument('--master-port', type=int, default=29622)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    output = Path('results/data16_sessions') / args.session
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.plan).read_text())
    parallel_layout = plan.get('parallel_layout', 'legacy')
    if parallel_layout not in ('legacy', 'node_local_pp'):
        raise ValueError(f'unknown parallel layout: {parallel_layout}')
    online = bool(plan.get('online', False))
    probe_enabled = bool(plan.get('probe_enabled', True))
    semantic_only = plan.get('profile_backend') == 'semantic_only'
    if semantic_only and (probe_enabled or ((online or plan.get('profile_enabled', True))
                                          and not plan.get('evidence_enabled', False))):
        raise ValueError('semantic_only requires evidence and disables the native probe')
    if online and semantic_only and not plan.get('evidence_model'):
        raise ValueError('online semantic_only requires an explicit v2 model')
    if online and not probe_enabled and not semantic_only:
        raise ValueError('online localization requires its on-demand probe')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    state = output / 'state.bin'
    state.write_bytes(int(probe_enabled and not online and plan.get('profile_enabled', True)).to_bytes(4, sys.byteorder))
    contexts = output / 'contexts'; contexts.mkdir()
    for rank in range(args.node_rank * 8, (args.node_rank + 1) * 8):
        (contexts / f'{rank}.bin').write_bytes(b'\0' * 16)
    if online and args.node_rank == 0:
        from flash_v1.monitor_channel import MonitorChannel
        MonitorChannel.create(output / 'monitor.bin').close()
    env = os.environ.copy()
    env['FLASH_PARALLEL_LAYOUT'] = parallel_layout
    env.update(OMP_NUM_THREADS='1', MALLOC_ARENA_MAX='2', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        PYTHONPATH=str(Path.cwd()), NCCL_MAX_NCHANNELS='4', NCCL_BUFFSIZE='1048576',
        NCCL_CUMEM_HOST_ENABLE='0', NCCL_CUMEM_ENABLE='0', NCCL_NVLS_ENABLE='0', NCCL_DEBUG='WARN', NCCL_NET='Socket', NCCL_IB_DISABLE='1', NCCL_SOCKET_IFNAME='bond0', GLOO_SOCKET_IFNAME='bond0',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    manifest = dict(session_id=args.session, node_rank=args.node_rank, plan=plan,
        configuration=dict(tp=2, pp=2, ep=2, expert_dp=2, nnodes=2, world_size=16,
            sequence_length=plan.get('seq_len', 16), microbatches=2, batch_size=1,
            fault_kind=plan.get('fault_kind', 'simulated_gpu_stream_delay'), collectives='AllReduce', sample_stride=8, profile_phase='attention-backward-plus-first-per-comm', model_id='bayliner1980/Qwen2.5-0.5B-MoE-A0.3B',
            model_revision='4b89028ef7376d0e92acbbc4a7f560c2374aeec0'),
        source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(Path('flash_v1').rglob('*')) if f.suffix in ('.py', '.cpp', '.cu')},
        environment={k: v for k, v in env.items() if k.startswith(('NCCL_', 'TORCH_NCCL_', 'PYTORCH_'))},
        probe_sha256=hashlib.sha256(Path('flash_v1/build/libflashprobe.so').read_bytes()).hexdigest() if not args.verify and probe_enabled else None,
        start_time=time.time())
    manifest['model_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(Path('models/qwen_moe').glob('*')) if f.suffix in ('.json', '.safetensors')}
    if plan.get('evidence_model'):
        manifest['evidence_model_sha256'] = hashlib.sha256(
            Path(plan['evidence_model']).read_bytes()).hexdigest()
    manifest['snapshot_sha256'] = hashlib.sha256(Path('flash_v1/build/libflashsnapshot.so').read_bytes()).hexdigest() if not args.verify else None
    if any(row.get('fault_mode') in ('attention_bwd_compute', 'whole_compute') for row in plan['intervals']):
        manifest['compute_workload_sha256'] = hashlib.sha256(
            Path('flash_v1/build/libflashcompute.so').read_bytes()).hexdigest()
    manifest['gpu_before'] = sp.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.free,clocks.sm',
        '--format=csv,noheader'], text=True).splitlines()
    manifest['online'] = online
    manifest['probe_enabled'] = probe_enabled
    if plan.get('scoped_compute_enabled', False):
        manifest['configuration']['scoped_compute'] = dict(
            enabled=True, catalog_only=bool(plan.get('scoped_catalog_only', False)),
            observer_independent=True, training_scope='schedule_only')
    if semantic_only:
        manifest['native_profile_coverage'] = 'not_applicable_semantic_only'
    manifest['configuration']['parallel_layout'] = parallel_layout
    if 'monitor_confirm_recent_three' in plan:
        if type(plan['monitor_confirm_recent_three']) is not bool:
            raise ValueError('monitor_confirm_recent_three must be boolean')
        manifest['configuration']['monitor_confirm_recent_three'] = plan['monitor_confirm_recent_three']
    if 'monitor_profile_transition' in plan:
        if type(plan['monitor_profile_transition']) is not bool:
            raise ValueError('monitor_profile_transition must be boolean')
        manifest['configuration']['monitor_profile_transition'] = plan['monitor_profile_transition']
        manifest['configuration']['training_steps'] = plan.get('steps',plan['intervals'][-1]['end'])
    if 'stage_forward_aux_enabled' in plan and type(plan['stage_forward_aux_enabled']) is not bool:
        raise ValueError('stage_forward_aux_enabled must be boolean')
    if plan.get('stage_forward_aux_enabled', False):
        manifest['configuration']['stage_forward_aux_enabled'] = True
        if plan.get('stage_forward_aux_model'):
            manifest['configuration']['stage_forward_aux_model'] = plan['stage_forward_aux_model']
            manifest['stage_forward_aux_model_sha256'] = hashlib.sha256(
                Path(plan['stage_forward_aux_model']).read_bytes()).hexdigest()
    if plan.get('stage_forward_aux_enabled', False) and plan.get('stage_forward_aux_context_binding'):
        binding_path=Path(plan['stage_forward_aux_context_binding'])
        manifest['configuration']['stage_forward_aux_context_binding']=str(binding_path)
        manifest['stage_forward_aux_context_binding_file_sha256']=hashlib.sha256(binding_path.read_bytes()).hexdigest()
    if plan.get('evidence_enabled', False):
        manifest['configuration']['evidence'] = dict(
            enabled=True, schema_version=1,
            selection_index=plan.get('evidence_selection_index', 0),
            output='evidence_rankN.jsonl',
            activation='PROFILE only')
    if plan.get('stage_forward_aux_sampling') is not None:
        from flash_v1.stage_forward_sampling import CONFIG
        if (plan['stage_forward_aux_sampling'] != CONFIG or not online
                or not plan.get('stage_forward_aux_enabled', False)):
            raise ValueError('sparse3 requires explicit online auxiliary sampling contract')
        manifest['configuration']['stage_forward_aux_sampling'] = plan['stage_forward_aux_sampling']
    if 'startup_gc_fixed_model' in plan or plan.get('startup_gc_policy') is not None:
        from flash_v1.startup_gc import validate_plan
        manifest['configuration']['startup_gc_fixed_model'] = validate_plan(plan)
        if manifest['configuration']['startup_gc_fixed_model']:
            manifest['configuration']['startup_gc_policy'] = plan['startup_gc_policy']
    manifest['environment']['FLASH_PARALLEL_LAYOUT'] = parallel_layout
    if plan.get('monitor_benchmark'):
        manifest['configuration']['monitor_benchmark_blocks'] = plan['benchmark_blocks']
    manifest['status'] = 'starting'
    (output / 'configuration.json').write_text(json.dumps(manifest['configuration'], indent=2))
    (output / f'node{args.node_rank}_manifest.json').write_text(json.dumps(manifest, indent=2))
    processes, logs = [], []
    def spawn(command, name, environment=env):
        log = (output / name).open('w'); logs.append(log)
        process = sp.Popen(command, env=environment, stdout=log, stderr=sp.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        port = 17422
        if not args.verify:
            if online and args.node_rank == 0:
                import redis
                redis_command = ['redis-server', '--port', str(port), '--bind', '127.0.0.1']
                if online:
                    redis_command += [args.master_addr, '--protected-mode', 'no']
                server = spawn(redis_command + ['--save', '', '--appendonly', 'no'], 'redis.log')
                client = redis.Redis(port=port)
                for _ in range(50):
                    try:
                        if client.info('server')['process_id'] == server.pid:
                            break
                        raise RuntimeError('private Redis port occupied')
                    except redis.ConnectionError:
                        time.sleep(.1)
                else:
                    raise RuntimeError('private Redis unavailable')
                if online:
                    controller_module = ('flash_v1.online_v2' if plan.get('evidence_model') else 'flash_v1.online')
                    spawn([sys.executable, '-m', controller_module, '--output', str(output),
                           '--redis-port', str(port), '--model', plan.get('evidence_model', 'results/data16_models/wait_frozen.json'),
                           '--warmup', '12', '--max-profiles', str(plan.get('max_profiles', 3))], 'online.log')
            drain_process = None
            drain_command = [sys.executable, '-m', 'flash_v1.scripts.drain_collection',
                             '--run-id', args.session, '--output', str(output)]
            if online:
                drain_command += ['--redis-host', args.master_addr, '--redis-port', str(port),
                                  '--node-rank', str(args.node_rank)]
            if probe_enabled:
                drain_process = spawn(drain_command, 'drain.log')
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=2', '--nproc-per-node=8',
                   f'--node-rank={args.node_rank}', f'--master-addr={args.master_addr}',
                   f'--master-port={args.master_port}', '--max-restarts=0']
        training_env = env.copy()
        if args.verify:
            command += ['-m', 'flash_v1.tests.tensor_parallel_verify', '--output', str(output)]
        else:
            training_env.update(FLASH_RUN_ID=args.session, FLASH_STATE_FILE=str(state.resolve()),
                FLASH_CONTEXT_DIR=str(contexts.resolve()), FLASH_SAMPLE_STRIDE='8', FLASH_PROFILE_PHASE='attention-backward')
            if probe_enabled:
                training_env['LD_PRELOAD'] = str(Path('flash_v1/build/libflashprobe.so').resolve())
            else:
                training_env.pop('LD_PRELOAD', None)
            command += ['-m', 'flash_v1.train_pipeline', '--output', str(output),
                '--collection-plan', args.plan, '--tensor-parallel-size', '2',
                '--seq-len', str(plan.get('seq_len', 16)), '--batch-size', '1', '--microbatches', '2',
                '--delay-placement', 'attention-experts', '--redis-port', str(port),
                '--seed', str(plan['seed'])]
            if online:
                command.append('--online')
            if plan.get('monitor_benchmark'):
                command.append('--monitor-benchmark')
        train = spawn(command, 'training.log', training_env)
        evidence_publisher = auxiliary_publisher = None
        if online and plan.get('evidence_enabled', False):
            import redis
            from flash_v1.scripts.v2_transport import EvidencePublisher
            evidence_client = redis.Redis(host=args.master_addr, port=port,
                                          socket_timeout=5, socket_connect_timeout=5)
            evidence_publisher = EvidencePublisher(output, args.node_rank, evidence_client,
                                                   publish_completion=semantic_only)
            if plan.get('stage_forward_aux_enabled', False):
                auxiliary_publisher = EvidencePublisher(output, args.node_rank, evidence_client,
                    filename='stage_forward_aux_rank{rank}.jsonl', key_prefix='online_stage_forward_aux_node',
                    completed_record='stage_forward_aux', completion_prefix='online_stage_forward_drained_node')
        deadline = time.monotonic() + args.timeout
        while train.poll() is None:
            if auxiliary_publisher is not None:
                auxiliary_publisher.publish()
            if evidence_publisher is not None:
                evidence_publisher.publish()
            if (output / 'STOP').exists():
                raise InterruptedError('session STOP requested')
            if time.monotonic() > deadline:
                raise TimeoutError('node session timeout')
            if any(proc.poll() is not None for proc in processes[:-1]):
                raise RuntimeError('collection auxiliary process exited')
            time.sleep(.5)
        if train.returncode:
            raise RuntimeError(f'torchrun exited {train.returncode}')
        if auxiliary_publisher is not None:
            auxiliary_publisher.publish()
            if any(auxiliary_publisher.partial.values()):
                raise RuntimeError('incomplete final stage forward auxiliary JSONL record')
            (output / 'stage_forward_transport.json').write_text(json.dumps(dict(
                published=auxiliary_publisher.published,
                partial_bytes=sum(len(x) for x in auxiliary_publisher.partial.values()))))
        if evidence_publisher is not None:
            evidence_publisher.publish()
            if any(evidence_publisher.partial.values()):
                raise RuntimeError('incomplete final evidence JSONL record')
            (output / 'evidence_transport.json').write_text(json.dumps(dict(
                published=evidence_publisher.published,
                partial_bytes=sum(len(x) for x in evidence_publisher.partial.values()))))
            evidence_client.close()
        if not args.verify:
            if online and args.node_rank == 0:
                finish_deadline = time.monotonic() + 30
                last_iteration = 204 if plan.get('monitor_sameprocess_cost') else max(row['end'] for row in plan['intervals'])
                client.set('online_training_done', json.dumps(dict(last_iteration=last_iteration)))
                while (int(client.get('online_processed_iteration') or 0) < last_iteration
                       or int(client.get('online_pending') or 0)):
                    if time.monotonic() > finish_deadline:
                        raise TimeoutError('online controller did not finish pending localization')
                    if any(proc.poll() is not None for proc in processes[:-1]):
                        raise RuntimeError('online auxiliary exited before result completion')
                    time.sleep(.1)
            if online:
                import redis
                finish_client = (client if args.node_rank == 0 else redis.Redis(
                    host=args.master_addr, port=port, socket_timeout=5, socket_connect_timeout=5))
                def check_finish_auxiliaries():
                    if (output / 'STOP').exists():
                        raise InterruptedError('session STOP requested')
                    if any(proc.poll() is not None for proc in processes[:-1]):
                        raise RuntimeError('online auxiliary exited during finish barrier')
                online_finish_barrier(finish_client, args.session, args.node_rank,
                                      timeout=60, check=check_finish_auxiliaries)
            if drain_process is not None:
                time.sleep(.5)
                (output / 'drain.stop').touch()
                drain_process.wait(timeout=10)
                drain = json.loads((output / 'drain_status.json').read_text())
                if drain['dropped'] or (not online and plan.get('profile_enabled', True) and not drain['samples']):
                    raise RuntimeError(f'incomplete ring acquisition: {drain}')
            if online:
                # The local drain has exited and written status. Its final
                # publication is complete before announcing readiness to close Redis.
                online_finish_barrier(finish_client, args.session, args.node_rank, local_drained=True)
                finish_client.close()
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except sp.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for log in logs:
            log.close()
        manifest['end_time'] = time.time()
        (output / f'node{args.node_rank}_manifest.json').write_text(json.dumps(manifest, indent=2))
        for rank in range(args.node_rank * 8, (args.node_rank + 1) * 8):
            for prefix in ('flash_record_', 'flash_lock_'):
                Path('/dev/shm', f'{prefix}{args.session}_{rank}').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
