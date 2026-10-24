"""Reproducible two-node acquisition and run-level dataset packaging.

Run on the workstation with existing host1/host2 SSH aliases. Each SSH command
starts only this experiment's node supervisor; STOP files stop only that session.
"""
import argparse
from collections import Counter
import hashlib
import gzip
import json
from pathlib import Path
import random
import shlex
import subprocess as sp
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/home/campus/zhangweihao/FLASH1022'
CONTAINER_ROOT = '/workspace/FLASH1022'
NODES = [('host2', 'flash1022-run', False), ('host1', 'flash1022-data16', True)]


def make_plan(session, role, seed, delay, ranks, measured=8):
    rng = random.Random(seed)
    # Healthy episodes are interleaved with a randomized rank ordering.
    order = list(ranks) + [None] * 4
    rng.shuffle(order)
    rows = [dict(id='warmup', start=1, end=12, record_start=13,
                 target_ranks=[], delay_ms=0, warmup=True)]
    start = 13
    for index, rank in enumerate(order):
        strength = delay if role != 'blind_test' else rng.uniform(delay * .65, delay * 1.35)
        end = start + measured + 2 - 1
        rows.append(dict(id=f'episode{index:02d}', start=start, end=end,
            record_start=start + 2, target_ranks=[] if rank is None else [rank],
            delay_ms=0 if rank is None else round(strength, 3)))
        start = end + 1
    return dict(session_id=session, role=role, seed=seed, seq_len=16,
                fault_kind='simulated_gpu_stream_delay', intervals=rows)


def command_for(host, container, sudo, session, verify, timeout):
    parts = (['sudo', '-n', 'env'] if sudo else ['env']) + [
        'DOCKER_HOST=unix:///run/greyhound-docker.sock', '/opt/greyhound-docker/docker',
        'exec', container, 'python', '-m', 'flash_v1.scripts.node_collection',
        '--session', session, '--plan', f'data16_plans/{session}.json',
        '--node-rank', str(0 if host == 'host2' else 1), '--timeout', str(timeout)]
    if verify:
        parts.append('--verify')
    # Long experiment supervisors need a dedicated channel: an expiring shared
    # ControlMaster must not turn completed node work into a harness failure.
    return ['ssh', '-o', 'ControlPath=none', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=4', host, shlex.join(parts)]


def run(plan, verify=False, timeout=900, *, startup_gc_fixed_model=False):
    if type(startup_gc_fixed_model) is not bool:
        raise ValueError('startup_gc_fixed_model must be boolean')
    if startup_gc_fixed_model:
        from flash_v1.startup_gc import configure_plan
        plan = configure_plan(plan)
    session = plan['session_id']
    plans = ROOT / 'tmp/data16_plans'; plans.mkdir(parents=True, exist_ok=True)
    path = plans / f'{session}.json'
    path.write_text(json.dumps(plan, indent=2))
    local = ROOT / 'results/data16_sessions' / session
    local.mkdir(parents=True, exist_ok=False)
    (local / 'plan.json').write_text(json.dumps(plan, indent=2))
    snapshot_command = ['env', 'DOCKER_HOST=unix:///run/greyhound-docker.sock',
        '/opt/greyhound-docker/docker', 'exec', 'flash1022-run',
        'tar', '-czf', '-', '--exclude=.git', '--exclude=__pycache__',
        'flash_v1', 'Greyhound-master']
    with (local / 'source_snapshot.tar.gz').open('wb') as archive:
        sp.run(['ssh', '-o', 'ControlPath=none', 'host2', shlex.join(snapshot_command)], stdout=archive, check=True, timeout=60)
    for host, _, _ in NODES:
        sp.run(['scp', '-o', 'ControlPath=none', '-q', str(path), f'{host}:{REMOTE}/data16_plans/'], check=True)
    processes, logs = [], []
    try:
        for host, container, sudo in NODES:
            log = (local / f'{host}_supervisor.log').open('w'); logs.append(log)
            processes.append(sp.Popen(command_for(host, container, sudo, session, verify, timeout),
                                      stdout=log, stderr=sp.STDOUT))
        deadline = time.monotonic() + timeout + 60
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('node supervisor failed; see session logs')
            if time.monotonic() > deadline:
                raise TimeoutError('two-node session timeout')
            time.sleep(1)
        if any(process.returncode for process in processes):
            raise RuntimeError('node supervisor failed')
    finally:
        for process, (host, _, _) in zip(processes, NODES):
            if process.poll() is None:
                container = NODES[0][1] if host == 'host2' else NODES[1][1]
                stop = ([] if host == 'host2' else ['sudo', '-n', 'env'])
                if not stop:
                    stop = ['env']
                stop += ['DOCKER_HOST=unix:///run/greyhound-docker.sock', '/opt/greyhound-docker/docker',
                         'exec', container, 'touch', f'{CONTAINER_ROOT}/results/data16_sessions/{session}/STOP']
                sp.run(['ssh', '-o', 'ControlPath=none', host, shlex.join(stop)], check=False)
        for process in processes:
            try:
                process.wait(timeout=20)
            except sp.TimeoutExpired:
                process.terminate()
        for log in logs:
            log.close()
        for host, container, sudo in NODES:
            destination = local / host
            destination.mkdir(exist_ok=True)
            if plan.get('online'):
                # The live mailbox is private to the container user. Make the
                # completed experiment artifact readable for SSH retrieval.
                mailbox = f'{CONTAINER_ROOT}/results/data16_sessions/{session}/monitor.bin'
                code = f'from pathlib import Path; p=Path({mailbox!r}); p.exists() and p.chmod(0o644)'
                command = (['sudo', '-n', 'env'] if sudo else ['env']) + [
                    'DOCKER_HOST=unix:///run/greyhound-docker.sock', '/opt/greyhound-docker/docker',
                    'exec', container, 'python', '-c', code]
                sp.run(['ssh', '-o', 'ControlPath=none', host, shlex.join(command)], check=True, timeout=30)
            sp.run(['rsync', '-az', '-e', 'ssh -o ControlPath=none', f'{host}:{REMOTE}/results/data16_sessions/{session}/', str(destination) + '/'], check=True)
    return local


def package(local):
    from flash_v1.topology import label_cliques
    plan = json.loads((local / 'plan.json').read_text())
    topology = json.loads((local / 'host2/topology.json').read_text())
    if topology['world_size'] != 16:
        raise ValueError('not a 16-rank dataset')
    for report in topology['rank_reports']:
        sizes = {group['name']: len(group['members']) for group in report['groups']}
        if sizes != dict(dense_world=4, expert_parallel=2, expert_dp=2, tensor_parallel=2, pipeline=2):
            raise ValueError(f'incorrect parallel layout: {sizes}')
    sources = {}
    config = None
    fingerprints = None
    raw_paths = []
    injections = []
    clique_examples = {}
    total_samples = 0
    for index, (host, _, _) in enumerate(NODES):
        manifest = json.loads((local / host / f'node{index}_manifest.json').read_text())
        drain = json.loads((local / host / 'drain_status.json').read_text())
        if manifest['status'] != 'completed' or drain['status'] != 'complete':
            raise ValueError(f'incomplete node {host}')
        if config is not None and config != manifest['configuration']:
            raise ValueError('configuration mismatch across nodes')
        config = manifest['configuration']
        identity = {key: manifest.get(key) for key in
                    ('source_sha256', 'probe_sha256', 'snapshot_sha256', 'model_sha256')}
        if fingerprints is not None and fingerprints != identity:
            raise ValueError('code, probe/snapshot binary or model differs across nodes')
        fingerprints = identity
        source = local / host / 'raw_profile.jsonl.gz'
        sources[host] = hashlib.sha256(source.read_bytes()).hexdigest()
        raw_paths.append(source)
        for truth_path in (local / host).glob('injection_rank*.jsonl'):
            injections.extend(json.loads(line) for line in truth_path.read_text().splitlines()
                              if 'injected_cycles' in json.loads(line))
        with gzip.open(source, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                clique_examples[(row['clique'], row['rank'])] = row
                total_samples += 1
    config = dict(config, attention_backward_block_size=3, sampling_stride=config['sample_stride'])
    examples, diagnostics = label_cliques(list(clique_examples.values()), topology)
    labels = {row['clique']: {key: row[key] for key in ('tag', 'group_name', 'group_members')}
              for row in examples}
    (local / 'label_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    output = ROOT / ('results/data16_pilot' if plan['role'] == 'pilot' else 'results/data16')
    output.mkdir(exist_ok=True)
    episodes = []
    stable_samples = 0
    try:
        for interval in plan['intervals']:
            if interval.get('warmup'):
                continue
            destination = output / f'{plan["session_id"]}_{interval["id"]}'
            destination.mkdir(exist_ok=False)
            episodes.append(dict(interval=interval, destination=destination, coverage=set(), samples=0,
                                 stream=(destination / 'profile.json').open('w')))
            episodes[-1]['stream'].write('[')
        for source in raw_paths:
            with gzip.open(source, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    label = labels.get(row['clique'])
                    if label is None:
                        continue
                    row.update(label)
                    stable_samples += 1
                    episode = next((e for e in episodes if e['interval']['record_start'] <= row['iteration'] <= e['interval']['end']), None)
                    if episode is None:
                        continue
                    if episode['samples']:
                        episode['stream'].write(',')
                    episode['stream'].write(json.dumps(row))
                    episode['samples'] += 1
                    episode['coverage'].add((row['iteration'], row['rank']))
        for episode in episodes:
            episode['stream'].write(']')
            episode['stream'].close()
            interval, destination = episode['interval'], episode['destination']
            missing = [(step, rank) for step in range(interval['record_start'], interval['end'] + 1)
                       for rank in range(16) if (step, rank) not in episode['coverage']]
            if missing:
                raise ValueError(f'incomplete episode coverage {interval["id"]}: {missing[:10]}')
            (destination / 'topology.json').write_text(json.dumps(topology, indent=2))
            target = interval['target_ranks']
            actual = [event for event in injections
                      if interval['record_start'] <= event['iteration'] <= interval['end']]
            expected = {(step, rank) for step in range(interval['record_start'], interval['end'] + 1)
                        for rank in target}
            if {(event['iteration'], event['rank']) for event in actual} != expected:
                raise ValueError(f'injection target/iteration mismatch: {interval["id"]}')
            requested = interval['delay_ms'] * config['microbatches']
            if any(abs(event['calibrated_injected_ms'] - requested) > max(1.0, requested * .05)
                   or event['hook_calls'] <= 0 for event in actual):
                raise ValueError('actual hook execution differs from requested injection')
            manifest = dict(session_id=plan['session_id'], run_id=destination.name,
                role=plan['role'], status='completed', fault_rank=target[0] if target else None,
                configuration=config, configuration_metadata_derived=['attention_backward_block_size=3 from the verified unfused Q/K/V TP implementation', 'sampling_stride aliases sample_stride'], seed=plan['seed'], interval=interval,
                raw_sha256=sources, source_archive_sha256=hashlib.sha256((local / 'source_snapshot.tar.gz').read_bytes()).hexdigest(), topology_sha256=hashlib.sha256((destination / 'topology.json').read_bytes()).hexdigest(),
                profile_sha256=hashlib.sha256((destination / 'profile.json').read_bytes()).hexdigest(),
                fault_kind=plan['fault_kind'], samples=episode['samples'],
                injection_verification=dict(observed_step_records=len(actual),
                    expected_step_records=len(expected), requested_total_ms_per_step=requested),
                rank_to_node={str(rank): ('host2' if rank < 8 else 'host1') for rank in range(16)})
            (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    finally:
        for episode in episodes:
            episode['stream'].close()
    print(json.dumps(dict(session=plan['session_id'], packaged_episodes=len(episodes),
                         stable_samples=stable_samples, total_samples=total_samples)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True)
    p.add_argument('--role', choices=['pilot', 'train', 'validation', 'blind_test'], default='pilot')
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--delay-ms', type=float, default=80)
    p.add_argument('--ranks', default=','.join(map(str, range(16))))
    p.add_argument('--measured-steps', type=int, default=8)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--package-only', action='store_true')
    args = p.parse_args()
    if args.package_only:
        local = ROOT / 'results/data16_sessions' / args.session
    else:
        plan = make_plan(args.session, args.role, args.seed, args.delay_ms,
                         [int(x) for x in args.ranks.split(',')], args.measured_steps)
        local = run(plan, timeout=args.timeout)
    package(local)


if __name__ == '__main__':
    main()
