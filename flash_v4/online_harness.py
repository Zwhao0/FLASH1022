"""Bounded R12 runtime development harness; the root agent owns all GPU jobs.

Private SSH credentials remain inside the existing private transport helper.
No clock changes, outside workload changes, or automatic case retries exist.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import zipfile

from flash_v4.health_centered_fusion import fingerprint, write_new

RELEASE = 'v4_online_20261027_r14'
BASE = '/workspace/FLASH1022'
CODE = BASE + '/results/v4/code/' + RELEASE
LOCAL = Path('results/v4/online') / RELEASE
CONTAINERS = {0: 'flash1022-v4-rdma-r2', 1: 'flash1022-v4-rdma-r3'}
CLI = 'env DOCKER_HOST=unix:///run/greyhound-docker.sock /opt/greyhound-docker/docker'
PLAN = Path('results/v4/preflight/online_r14_development_plan.json')


def configure_plan(path):
    global PLAN, RELEASE, CODE, LOCAL
    plan = json.loads(Path(path).read_text())
    import re
    release = plan.get('runtime_release', '')
    if not re.fullmatch(r'v4_[a-z0-9_]{1,60}', release):
        raise ValueError('invalid isolated runtime release')
    RELEASE, CODE = release, BASE + '/results/v4/code/' + release
    LOCAL = Path('results/v4/online') / release
    PLAN = Path(path)


def transport(node, command, log=None):
    args = [sys.executable, '/tmp/flash_v4_remote.py', str(node), command]
    if log is not None:
        args.append(str(log))
    result = subprocess.run(args, capture_output=True, text=True, timeout=100)
    if result.returncode:
        raise RuntimeError(f'remote operation failed on node {node}: ' + result.stdout[-500:] + result.stderr[-200:])
    return result.stdout


def container_command(node, command, detached=False, environment=None, working_directory=None):
    prefix = ('sudo -n ' if node == 0 else '') + CLI
    flags = ' exec ' + ('-d ' if detached else '') + '-w ' + shlex.quote(working_directory or CODE)
    for key, value in (environment or {}).items():
        flags += ' -e ' + shlex.quote(key + '=' + value)
    return prefix + flags + ' ' + CONTAINERS[node] + ' ' + command


def case_row(case):
    plan = json.loads(PLAN.read_text())
    return next(r for r in plan['development_GPU_sessions'] if r['case'] == case)


def private_case(case):
    """Validate the actor on the controller before either node can be dispatched."""
    from flash_v4.train_entry import _parse_actor_window, _parse_stream_delay_mapping
    plan = json.loads(PLAN.read_text())
    path = plan.get('private_execution')
    if not path:
        return None
    receipt = plan.get('private_execution_receipt')
    if receipt and fingerprint(Path(path))['sha256'] != receipt['sha256']:
        raise ValueError('private execution plan changed after freeze')
    private = json.loads(Path(path).read_text())['cases'][case]
    target, ratio = private['target_rank'], private['ratio']
    ratio_limit = plan.get('actor_budget', {}).get('max_requested_ratio', 1.0)
    if (type(ratio_limit) not in (int, float) or not 0 <= ratio_limit <= 8
            or type(ratio) not in (int, float) or not 0 <= ratio <= ratio_limit):
        raise ValueError('actor ratio exceeds declared plan budget')
    if target is None and (type(ratio) not in (int, float) or ratio != 0):
        raise ValueError('healthy actor requires zero ratio')
    mapping = {} if target is None else {str(target): ratio}
    _parse_stream_delay_mapping(json.dumps(mapping))
    first, last = _parse_actor_window(json.dumps(private['window']))
    if last - first + 1 > plan.get('actor_budget', {}).get('max_fault_iterations', 32):
        raise ValueError('actor window exceeds native iteration budget')
    if last > case_row(case)['steps'] + 3:
        raise ValueError('actor window exceeds training schedules')
    return private


def bootstrap_command(args, output):
    """Retain errors that occur before shared_node can publish node_status.json."""
    output = Path(output)
    log = output.parent / (output.name + '_bootstrap.log')
    exit_file = output.parent / (output.name + '_bootstrap_exit')
    script = ('mkdir -p ' + shlex.quote(str(output.parent)) + '; ' + shlex.join(args)
              + ' > ' + shlex.quote(str(log)) + ' 2>&1; rc=$?; '
              + 'printf "%s\\n" "$rc" > ' + shlex.quote(str(exit_file)) + '; exit "$rc"')
    return 'sh -c ' + shlex.quote(script)


def remote_output(case, node):
    return BASE + '/results/v4/online/' + RELEASE + '/' + case + '/node' + str(1 - node)


def runtime_run_id(case):
    # Full release identity is required: descriptive suffixes recur across rounds.
    return 'v4_' + hashlib.sha256((RELEASE + ':' + case).encode()).hexdigest()[:32]


def case_timeout(row):
    from flash_v4.execution_budget import long_health_budget
    budget = long_health_budget(row)
    if budget is not None:
        return budget['node_timeout_seconds']
    value = row.get('timeout', 180)
    if type(value) is not int or not 30 <= value <= 180:
        raise ValueError('case timeout must be30-180 seconds')
    return value


def stage():
    # Explicit plan opt-in; historical plans keep their frozen r11 observer.
    library = Path(json.loads(PLAN.read_text()).get(
        'observer_library_artifact', 'results/v4/build/libflash_v4_observer_r11.so'))
    if not library.is_file():
        raise FileNotFoundError(library)
    LOCAL.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path('flash_v4').glob('*.py')):
            archive.write(path, str(path))
        archive.write(library,
                      'flash_v4/native/libflash_v4_observer.so')
        for name in ('stream_delay.cpp','stream_delay.cu','build_stream_delay.sh'):
            archive.write(Path('flash_v4/native') / name, 'flash_v4/native/' + name)
    payload = base64.b64encode(buffer.getvalue()).decode()
    with (LOCAL / 'release.zip').open('xb') as stream:
        stream.write(buffer.getvalue())
    source_files = [fingerprint(p) for p in sorted(Path('flash_v4').glob('*.py'))]
    write_new(LOCAL / 'release_receipt.json', {'release': RELEASE, 'sources': source_files,
               'observer_library': fingerprint(library),
               'archive': fingerprint(LOCAL / 'release.zip'), 'plan': fingerprint(PLAN)})
    expected = {str(Path(r['path']).relative_to(Path.cwd())): r['sha256']
                for r in json.loads(PLAN.read_text())['training_sources_before']}
    python = (
        'import pathlib,hashlib,json,base64,io,zipfile; '
        f'root=pathlib.Path({BASE!r}); expected={expected!r}; '
        'assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in expected.items()), "v1_source_mismatch"; '
        f'p=pathlib.Path({CODE!r}); p.mkdir(parents=True,exist_ok=False); '
        'print("v1_sources_match")'
    )
    def one(node):
        with (LOCAL / f'stage_node{node}.jsonl').open('x') as journal:
            def send(phase, code):
                value = transport(node, container_command(node, 'python3 -c ' + shlex.quote(code), working_directory=BASE))
                journal.write(json.dumps({'phase':phase,'result':value.strip()})+'\n');journal.flush()
            send('initialize',python)
            for i in range(0,len(payload),32768):
                code=(f'import pathlib,base64; p=pathlib.Path({CODE!r})/"release.zip"; '
                      f'f=p.open("ab"); f.write(base64.b64decode({payload[i:i+32768]!r})); f.close(); print("chunk_ok")')
                send('chunk'+str(i//32768),code)
            sha=hashlib.sha256(buffer.getvalue()).hexdigest()
            send('verify_extract',f'import pathlib,hashlib,zipfile; p=pathlib.Path({CODE!r}); z=p/"release.zip"; '
                 f'assert hashlib.sha256(z.read_bytes()).hexdigest()=={sha!r}; zipfile.ZipFile(z).extractall(p); print("staged")')
        return 'node '+str(node)+' staged'
    with ThreadPoolExecutor(2) as pool:
        for result in pool.map(one, (0, 1)):
            print(result.strip())


def cpu_test(node):
    python = r'''
import json,pathlib,time
from flash_v1.analysis import WindowDetector
root=pathlib.Path('cpu_monitor_check');root.mkdir(exist_ok=False)
summary={}
for name,values in [('healthy',[100+((i*17)%9-4)*0.1 for i in range(42)]),
                    ('persistent',[100+((i*17)%9-4)*0.1 if i<26 else 150+((i*17)%9-4)*0.1 for i in range(42)])]:
    detector=WindowDetector();event=None;start=time.process_time()
    for i,value in enumerate(values,1):
        event=detector.add(i,value)
        if event:break
    summary[name]={'event':event,'consumed':i,'cpu_ms':1000*(time.process_time()-start)}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
assert summary['healthy']['event'] is None,'synthetic_health_trigger'
assert summary['persistent']['event'] is not None,'synthetic_persistent_not_triggered'
print(json.dumps(summary))
'''
    result = transport(node, container_command(node, 'python3 -c ' + shlex.quote(python),
                       environment={'PYTHONPATH': CODE + ':' + BASE, 'CUDA_VISIBLE_DEVICES': '',
                                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}),
                       LOCAL / f'cpu_test_node{node}.txt')
    print(result.strip())


def build_actor():
    budget = json.loads(PLAN.read_text()).get('actor_budget', {})
    step_us = budget.get('max_step_us', 200000)
    fault_iterations = budget.get('max_fault_iterations', 32)
    if (type(step_us) is not int or not 1 <= step_us <= 1000000
            or type(fault_iterations) is not int or not 1 <= fault_iterations <= 32):
        raise ValueError('native actor build exceeds bounded plan budget')
    args=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-shared','-cudart','shared',
          f'-DFLASH_V4_ACTOR_MAX_FAULT_ITERATIONS={fault_iterations}',
          f'-DFLASH_V4_ACTOR_MAX_STEP_US={step_us}',
          '-gencode','arch=compute_90,code=sm_90','-gencode','arch=compute_90,code=compute_90',
          '-Xcompiler=-fPIC,-pthread,-Wall,-Wextra',
          'flash_v4/native/stream_delay.cpp','flash_v4/native/stream_delay.cu',
          '-L/usr/local/cuda/lib64','-Xlinker=-rpath','-Xlinker=/usr/local/cuda/lib64',
          '-ldl','-lpthread','-o','flash_v4/native/libflash_v4_stream_delay.so']
    def one(node):
        return transport(node,container_command(node,shlex.join(args)),LOCAL/f'actor_build_node{node}.txt')
    with ThreadPoolExecutor(2) as pool:
        for value in pool.map(one,(0,1)):print(value.strip() or 'actor compiled')


def launch(case):
    row = case_row(case)
    timeout = case_timeout(row)
    private = private_case(case)
    finalize_contract = row.get('profile_finalize')
    if finalize_contract is not None:
        from flash_v4.profile_finalize import validate_contract
        validate_contract(finalize_contract, row['mode'], finalize_contract)
    progress_contract = row.get('rank_progress_observation')
    if progress_contract is not None:
        from flash_v4.rank_progress import CONTRACT, MAX_ITERATIONS
        if (progress_contract != CONTRACT or row['mode'] != 'online'
                or not 1 <= row['steps'] + 3 <= MAX_ITERATIONS):
            raise ValueError('invalid explicit rank progress observation')
    if row.get('execution_budget') is not None and private is not None:
        raise ValueError('long healthy execution excludes private actors')
    marker = LOCAL / case
    marker.mkdir(parents=True, exist_ok=False)
    write_new(marker / 'launch_receipt.json', {'case': row, 'dispatch_wall_time': time.time(),
                                              'owner': 'root', 'clock_changed': False})
    def one(node):
        args = ['python3', '-m', 'flash_v4.shared_node', '--run-id', runtime_run_id(case),
                '--layout', row['topology'], '--node-rank', str(1 - node),
                '--master-addr', '__V4_MASTER_ADDR__', '--master-port', str(json.loads(PLAN.read_text()).get('master_port',35544)),
                '--model', BASE + '/models/qwen_moe', '--output', remote_output(case, node),
                '--steps', str(row['steps']), '--seed', str(row['seed']), '--timeout', str(timeout),
                '--transport', 'ib', '--seq-len', str(row.get('sequence_length', 256)), '--observe-mode', row['mode'],
                '--observer-library', CODE + '/flash_v4/native/libflash_v4_observer.so']
        if row.get('execution_budget') is not None:
            args.extend(['--execution-budget', row['execution_budget']])
        env = {'PYTHONPATH': CODE + ':' + BASE, 'FLASH_V4_KERNEL_PROFILE': '1',
               'FLASH_V4_FORCE_PROFILE_AT': str(row.get('force_profile_at', 0))}
        if 'monitor_control' in row:
            from flash_v4.monitor_control import validate_contract
            if row['mode'] != 'online':
                raise ValueError('MONITOR control requires online mode')
            env['FLASH_V4_MONITOR_CONTROL'] = validate_contract(row['monitor_control'])
            env['FLASH_V4_MONITOR_RUN_ID'] = runtime_run_id(case)
        if 'sample_stride' in row:
            from flash_v4.profile_sampling import profile_stride
            env['FLASH_V4_PROFILE_SAMPLE_STRIDE'] = str(profile_stride(str(row['sample_stride'])))
        if finalize_contract is not None:
            env.update(FLASH_V4_PROFILE_FINALIZE=finalize_contract,
                       FLASH_V4_PROFILE_FINALIZE_SUPERVISED=finalize_contract)
        if progress_contract is not None:
            env.update(FLASH_V4_RANK_PROGRESS=progress_contract,
                       FLASH_V4_PROGRESS_RUN_ID=runtime_run_id(case))
        if 'monitor_degradation' in row:
            degradation = float(row['monitor_degradation'])
            if not 0 < degradation < 1:
                raise ValueError('invalid per-case monitor degradation')
            env['FLASH_V4_MONITOR_DEGRADATION'] = str(degradation)
        if 'allocator_budget' in row:
            if row['allocator_budget'] not in ('fixed', 'available'):
                raise ValueError('invalid per-case allocator budget')
            env['FLASH_V4_ALLOCATOR_BUDGET'] = row['allocator_budget']
        if private is not None:
            mapping={} if private['target_rank'] is None else {str(private['target_rank']):private['ratio']}
            env['FLASH_V4_STREAM_DELAY_PRIVATE']=json.dumps(mapping)
            env['FLASH_V4_ACTOR_WINDOW_PRIVATE']=json.dumps(private['window'])
            args.extend(['--stream-delay-library',CODE+'/flash_v4/native/libflash_v4_stream_delay.so'])
        # Abort on other MPS owners; never terminate them or change their daemon.
        guard = "ps -eo stat,comm | awk '$2 ~ /^nvidia-cuda-mps/ && $1 !~ /^Z/ {found=1} END {exit found}'; "
        guard += 'check=$?; if [ "$check" -ne 0 ]; then echo external_mps_present_abort; exit 20; fi; '
        command = bootstrap_command(args, remote_output(case, node))
        return transport(node, guard + container_command(node, command, detached=True, environment=env),
                         marker / f'launch_node{node}.txt')
    with ThreadPoolExecutor(2) as pool:
        for result in pool.map(one, (0, 1)):
            print(result.strip() or 'node dispatched')


def status(case):
    def one(node):
        python = (f'import json,pathlib; p=pathlib.Path({remote_output(case,node)!r}); '
                  's=p/"node_status.json"; d=json.loads(s.read_text()) if s.exists() else {}; '
                  'b=p.parent/(p.name+"_bootstrap_exit"); '
                  'early=(not s.exists() and b.exists() and int(b.read_text())!=0); '
                  'd.update(status="failed",returncode=int(b.read_text()),error="bootstrap_failed_before_status",'
                  'private_mps_stopped=not pathlib.Path("/tmp/flash1022_v4_gpu_owner").exists()) if early else None; '
                  'd={k:d.get(k) for k in ["status","returncode","error","private_mps_stopped","online_monitor_integrity"]}; '
                  'd["profile_ready_ranks"]=len(list((p/"control").glob("ready_rank*.json"))); print(json.dumps(d))')
        value = transport(node, container_command(node, 'python3 -c ' + shlex.quote(python)))
        return {'node': 1 - node, **json.loads(value.strip())}
    with ThreadPoolExecutor(2) as pool:
        print(json.dumps(list(pool.map(one, (0, 1)))))


def fetch(case, snapshot):
    target = LOCAL / case / snapshot
    target.mkdir(parents=True, exist_ok=False)
    def one(node):
        python = (f'import pathlib,tarfile,io,base64; p=pathlib.Path({remote_output(case,node)!r}); '
                  'b=io.BytesIO(); t=tarfile.open(fileobj=b,mode="w:gz"); '
                  f't.add(p,arcname="node{1-node}") if p.exists() else None; '
                  '[t.add(q,arcname=q.name) for suffix in ("_bootstrap.log","_bootstrap_exit") '
                  'if (q:=p.parent/(p.name+suffix)).exists()]; '
                  't.close(); assert len(b.getvalue())<12*1024*1024; print("ARCHIVE:"+base64.b64encode(b.getvalue()).decode())')
        result = transport(node, container_command(node, 'python3 -c ' + shlex.quote(python)))
        archive = base64.b64decode(result.split('ARCHIVE:', 1)[1].strip())
        with (target / f'node{1-node}.raw.tar.gz').open('xb') as stream:
            stream.write(archive)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
            for member in tar.getmembers():
                path = (target / member.name).resolve()
                if not path.is_relative_to(target.resolve()) or not (member.isfile() or member.isdir()):
                    raise RuntimeError('unsafe archive entry')
            tar.extractall(target)
        return {'node': 1-node, 'archive_bytes': len(archive)}
    with ThreadPoolExecutor(2) as pool:
        print(json.dumps(list(pool.map(one, (0, 1)))))


def localize(case, snapshot):
    row = case_row(case)
    data_root = LOCAL / case / snapshot
    command=[sys.executable,'-m','flash_v4.online_infer','--input-root',str(data_root),
             '--topology',row['topology'],'--case',case]
    if row.get('force_profile_at'):command.append('--manual')
    from flash_v4.online_runtime import child_environment
    result=subprocess.run(command,env=child_environment(__import__('os').environ),capture_output=True,text=True)
    if result.returncode:raise RuntimeError('isolated inference failed: '+result.stderr[-500:])
    print(result.stdout.strip())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('stage','cpu-test','build-actor','launch','status','fetch','localize'))
    p.add_argument('--node', type=int, choices=(0,1), default=1)
    p.add_argument('--case')
    p.add_argument('--snapshot', choices=('live','final'), default='final')
    p.add_argument('--plan',type=Path)
    a = p.parse_args()
    if a.plan:configure_plan(a.plan)
    if a.action == 'stage': stage()
    elif a.action == 'cpu-test': cpu_test(a.node)
    elif a.action == 'build-actor':build_actor()
    else:
        case_row(a.case)
        if a.action == 'launch': launch(a.case)
        elif a.action == 'status': status(a.case)
        elif a.action == 'fetch': fetch(a.case,a.snapshot)
        else: localize(a.case,a.snapshot)
