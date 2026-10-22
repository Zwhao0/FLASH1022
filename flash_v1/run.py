"""Isolated FLASH experiments: real clocks or calibrated GPU stream delay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess as sp
import sys
import time
import uuid

import redis


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['baseline','monitor','fault'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', default='4,5,6,7')
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--seq-len', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--inject-at', type=int, default=45)
    p.add_argument('--fault-seconds', type=float, default=50)
    p.add_argument('--frequency', type=int, default=345)
    p.add_argument('--fault-kind', choices=['clock','gpu-delay'], default='clock')
    p.add_argument('--delay-ms', type=float, default=250.0)
    p.add_argument('--fault-steps', type=int, default=80)
    p.add_argument('--fault-rank', type=int, default=3)
    p.add_argument('--redis-port', type=int, default=16422)
    p.add_argument('--gradient-backend', choices=['serial', 'megatron', 'pipeline'], default='serial')
    p.add_argument('--bucket-size', type=int, default=2_097_152)
    p.add_argument('--expert-parallel-size', type=int, default=2)
    p.add_argument('--delay-placement', choices=['attention', 'attention-experts'], default='attention')
    p.add_argument('--microbatches', type=int, default=2)
    p.add_argument('--profile-at', type=int, help='Force one diagnostic PROFILE at this iteration; not a detection')
    p.add_argument('--reference-profile', help='Matched healthy PROFILE JSON for calibrated rank localization')
    args = p.parse_args()
    if args.reference_profile:
        p.error('healthy PROFILE differential localization is temporarily disabled; omit --reference-profile')
    if args.reference_profile:
        reference = Path(args.reference_profile).resolve()
        reference_manifest = json.loads((reference.parent / 'manifest.json').read_text())
        if reference_manifest.get('mode') != 'monitor' or reference_manifest.get('status') != 'completed':
            p.error('reference-profile must come from a completed healthy monitor run')
        for key in ('gpus', 'gradient_backend', 'seq_len', 'batch_size', 'microbatches',
                    'bucket_size', 'expert_parallel_size', 'delay_placement'):
            if reference_manifest.get(key) != getattr(args, key):
                p.error(f'reference-profile configuration mismatch: {key}')
        for key in ('NCCL_MAX_NCHANNELS', 'NCCL_BUFFSIZE'):
            if reference_manifest.get('environment', {}).get(key) != os.environ.get(key):
                p.error(f'reference-profile environment mismatch: {key}')
        args.reference_profile = str(reference)
    gpus = args.gpus.split(',')
    if not 0 <= args.fault_rank < len(gpus) or args.fault_steps <= 0:
        p.error('fault-rank must be in selected GPUs and fault-steps positive')
    if not math.isfinite(args.delay_ms) or args.delay_ms <= 0 or not math.isfinite(args.fault_seconds) or args.fault_seconds <= 0:
        p.error('delay-ms and fault-seconds must be finite and positive')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex[:12]
    state = output / 'state.bin'; state.write_bytes(b'\0'*4)
    if args.gradient_backend == 'serial' and (len(gpus) != 4 or args.expert_parallel_size != 2):
        raise ValueError('serial backend requires EP2 and four physical GPUs')
    if args.expert_parallel_size < 2 or len(gpus) % args.expert_parallel_size:
        raise ValueError('GPU count must be divisible by EP size >=2')
    if args.gradient_backend == 'pipeline' and (len(gpus) != 8 or args.expert_parallel_size != 2):
        raise ValueError('pipeline backend requires eight GPUs and EP2')
    target = gpus[args.fault_rank]
    processes=[]; files=[]; locked=False; watchdog=None
    client = None
    manifest=dict(vars(args),run_id=run_id,wall_start=time.time(),clock_injected=False,fault_injected=False,
                  environment={key: value for key, value in os.environ.items()
                               if key.startswith('NCCL_') or key == 'PYTORCH_CUDA_ALLOC_CONF'})
    sources = list(Path('flash_v1').rglob('*.py')) + list(Path('flash_v1/native').glob('*'))
    sources += [Path('Greyhound-master/Megatron-LM/megatron/core') / name for name in (
        'parallel_state.py', 'distributed/distributed_data_parallel.py',
        'distributed/grad_buffer.py', 'pipeline_parallel/schedules.py')]
    manifest['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(sources) if path.is_file()}
    if args.reference_profile:
        manifest['reference_profile_sha256'] = hashlib.sha256(Path(args.reference_profile).read_bytes()).hexdigest()
    def record(event, **fields):
        with (output/'injection.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event,walltime=time.time(),**fields))+'\n')
    def stop(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        # Bind loopback and use a private port; no existing Redis database is touched.
        log=(output/'redis.log').open('w');files.append(log)
        redis_proc=sp.Popen(['redis-server','--port',str(args.redis_port),'--bind','127.0.0.1','--save','','--appendonly','no'],stdout=log,stderr=log)
        processes.append(redis_proc)
        client=redis.Redis(port=args.redis_port,decode_responses=True)
        for _ in range(50):
            try:
                client.ping();break
            except redis.ConnectionError:time.sleep(.1)
        else:raise RuntimeError('private Redis failed')
        if redis_proc.poll() is not None or client.info('server')['process_id'] != redis_proc.pid:
            raise RuntimeError('private Redis port unavailable; existing Redis was not modified')
        client.set('state','0')
        if args.profile_at is not None:
            if args.mode == 'baseline' or args.profile_at < 4:
                raise ValueError('profile-at requires monitoring and iteration >=4')
            client.set('manual_profile_iteration', args.profile_at)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,OMP_NUM_THREADS='1',MALLOC_ARENA_MAX='2',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONPATH=str(Path.cwd()))
        # The parent and controllers never preload the NCCL hook.
        if args.mode!='baseline':
            for role in ('local','global'):
                log=(output/f'{role}.log').open('w');files.append(log)
                command=[sys.executable,'-m','flash_v1.control',role,'--redis-port',str(args.redis_port),'--output',str(output),'--run-id',run_id,'--state-file',str(state)]
                if role == 'global' and args.reference_profile:
                    command += ['--reference-profile', args.reference_profile]
                processes.append(sp.Popen(command,stdout=log,stderr=log,env=env))
            for _ in range(100):
                if client.exists('local_ready'):break
                time.sleep(.1)
            else:raise RuntimeError('local controller startup failed')
            env.update(LD_PRELOAD=str(Path('flash_v1/build/libflashprobe.so').resolve()),FLASH_RUN_ID=run_id,FLASH_STATE_FILE=str(state))
        log=(output/'training.log').open('w');files.append(log)
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4','flash_v1/train.py','--steps',str(args.steps),'--seq-len',str(args.seq_len),'--batch-size',str(args.batch_size),'--output',str(output),'--redis-port',str(args.redis_port)]
        command += ['--gradient-backend', args.gradient_backend, '--bucket-size', str(args.bucket_size)]
        command += ['--expert-parallel-size', str(args.expert_parallel_size), '--delay-placement', args.delay_placement]
        command[command.index('--nproc-per-node=4')] = f'--nproc-per-node={len(gpus)}'
        if args.gradient_backend == 'pipeline':
            command[command.index('flash_v1/train.py')] = 'flash_v1/train_pipeline.py'
            command += ['--microbatches', str(args.microbatches)]
        if args.mode=='fault' and args.fault_kind=='gpu-delay':
            command += ['--compute-delay-ms',str(args.delay_ms),'--delay-start',str(args.inject_at),'--delay-end',str(args.inject_at+args.fault_steps),'--fault-rank',str(args.fault_rank)]
        # The library initializes only on a worker's first NCCL call.
        train=sp.Popen(command,stdout=log,stderr=log,env=env,start_new_session=True)
        processes.append(train)
        reset_at=None;start=time.monotonic()
        while train.poll() is None:
            if time.monotonic()-start>1200:raise TimeoutError('experiment exceeded20 minutes')
            for controller in processes[1:-1]:
                if controller.poll() is not None:raise RuntimeError('FLASH controller exited; see logs')
            iteration=int(client.get('latest_iteration') or 0)
            if args.mode=='fault' and not manifest['fault_injected'] and iteration>=args.inject_at and args.fault_kind=='clock':
                before=sp.check_output(['nvidia-smi','-i',target,'--query-gpu=clocks.sm,clocks.max.sm','--format=csv,noheader'],text=True).strip()
                sp.run(['nvidia-smi','-i',target,'-lgc',str(args.frequency)],check=True,capture_output=True)
                locked=True;manifest['clock_injected']=True;manifest['fault_injected']=True;reset_at=time.monotonic()+args.fault_seconds
                # Independent emergency reset survives a killed Python runner.
                watchdog=sp.Popen(['bash','-c',f'sleep {args.fault_seconds+10}; nvidia-smi -i {int(target)} -rgc >/dev/null'],start_new_session=True)
                record('clock_lowered',iteration=iteration,physical_gpu=target,rank=args.fault_rank,requested_mhz=args.frequency,before=before)
            if args.mode=='fault' and args.fault_kind=='gpu-delay' and not manifest['fault_injected'] and iteration>=args.inject_at:
                manifest['fault_injected']=True
                record('compute_delay_requested',iteration=iteration,physical_gpu=target,rank=args.fault_rank,delay_ms=args.delay_ms,first_fault_iteration=args.inject_at+1,last_fault_iteration=args.inject_at+args.fault_steps,kind='simulated_gpu_stream_delay')
            if locked and time.monotonic()>=reset_at:
                sp.run(['nvidia-smi','-i',target,'-rgc'],check=True,capture_output=True);locked=False
                record('clock_restored',iteration=iteration)
            if locked and iteration%10==0:
                actual=sp.check_output(['nvidia-smi','-i',target,'--query-gpu=clocks.sm','--format=csv,noheader'],text=True).strip()
                record('clock_sample',iteration=iteration,actual=actual)
            time.sleep(.25)
        if train.returncode:raise RuntimeError(f'training exit {train.returncode}; see training.log')
        if args.mode=='fault' and not manifest['fault_injected']:
            raise RuntimeError('training ended before fault injection')
        # PROFILE->MONITOR happens before collection/localization finishes.
        # Require the epoch's result before reporting a completed experiment.
        if args.mode!='baseline':
            until=time.monotonic()+35
            while True:
                epoch = client.get('profile_epoch')
                latest = int(client.get('latest_iteration') or 0)
                processed = int(client.get('processed_iteration') or 0)
                if processed >= latest and (epoch is None or client.exists(f'profile_completed_{epoch}')):
                    break
                if time.monotonic() >= until:
                    raise TimeoutError(f'profile epoch {epoch} did not finish localization before shutdown')
                for controller in processes[1:-1]:
                    if controller.poll() is not None:
                        raise RuntimeError('FLASH controller exited before localization completed')
                time.sleep(.2)
            manifest['completed_profile_epoch'] = epoch
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed',error=repr(error))
        raise
    finally:
        if locked:
            sp.run(['nvidia-smi','-i',target,'-rgc'],check=True,capture_output=True)
            record('clock_restored_finally')
        if watchdog and watchdog.poll() is None:
            os.killpg(watchdog.pid,signal.SIGTERM)
        if client:
            try:client.set('shutdown','1')
            except redis.RedisError:pass
        for process in reversed(processes):
            if process.poll() is None:
                if process is processes[-1]:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                else:process.terminate()
                try:process.wait(timeout=8)
                except sp.TimeoutExpired:process.kill()
        for f in files:f.close()
        for rank in range(len(gpus)):
            for prefix in ('flash_record_','flash_lock_'):
                Path('/dev/shm',f'{prefix}{run_id}_{rank}').unlink(missing_ok=True)
        manifest['wall_end']=time.time()
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
