"""Bounded development protocol; the final blind schedule is never generated."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

TOPOLOGIES = ('legacy', 'node_local_pp')
FAULT_STRATA = ('light', 'medium', 'strong')
FAULT_MECHANISM = 'mps_quota'
MPS_QUOTA_REQUESTS = {
    'development': {'light': 75.0, 'medium': 50.0, 'strong': 25.0},
    # Unseen interpolation points relative to development requests.
    'validation': {'light': 88.0, 'medium': 63.0, 'strong': 38.0},
}
HEALTHY_MPS_QUOTA_PERCENT = 100.0
MATRIX = {
    'development': {'healthy_sessions': 4, 'faults_per_stratum': (4, 4, 4)},
    'validation': {'healthy_sessions': 2, 'faults_per_stratum': (2, 2, 2)},
}
BLIND_TARGET = {'fault_localizations': 32, 'per_topology': 16,
                'schedule_generated': False}
BUDGET = {
    'world_size': 16,
    'max_training_sessions': 48,
    'max_healthy_sessions': 12,
    'max_fault_sessions': 36,
    'profile_iterations': 8,
    'profile_seconds': 30,
    'profile_bytes_per_rank': 1_048_576,
    'event_bytes': 1024,
    'fault_session_seconds': 180,
    'healthy_windows_per_session': 5,
    'healthy_window_seconds': 60,
    'max_active_collection_seconds': 10080,
}
BASELINE_FILES = (
    'flash_v1/train.py', 'flash_v1/train_pipeline.py',
    'flash_v1/megatron_adapter.py', 'flash_v1/pipeline_adapter.py',
    'flash_v1/tensor_parallel.py', 'flash_v1/rank_layout.py',
)


def roles(rank):
    return (rank // 8, (rank // 4) % 2, (rank // 2) % 2, rank % 2)


def representative(ranks):
    return all({roles(r)[i] for r in ranks} == {0, 1} for i in range(4))


def partition(seed):
    """Disjoint physical GPU slots: development 4, validation 6, blind 6."""
    rng = random.Random(seed)
    for _ in range(1000):
        candidates = list(range(16))
        rng.shuffle(candidates)
        dev, val, blind = candidates[:4], candidates[4:10], candidates[10:]
        if all(representative(group) for group in (dev, val, blind)):
            return {'development': sorted(dev), 'validation': sorted(val),
                    'blind_only_fault_devices': sorted(blind)}
    raise ValueError('cannot form representative disjoint device partitions')


def source_manifest(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in BASELINE_FILES}


def _new_session(split, topology, workload_seed, session_id, *, healthy,
                 fault_case=None):
    return {
        'session_id': session_id,
        'split': split,
        'topology': topology,
        'workload_seed': workload_seed,
        'condition': 'healthy' if healthy else 'mps_quota_fault',
        'mps_mode': 'shared',
        'healthy_windows': BUDGET['healthy_windows_per_session'] if healthy else 0,
        # A healthy control starts every rank process at the unrestricted request.
        'healthy_mps_quota_percent': HEALTHY_MPS_QUOTA_PERCENT if healthy else None,
        # A fault request is static for one target rank process from client start.
        'fault_cases': [] if fault_case is None else [fault_case],
    }


def make_protocol(seed, root):
    groups = partition(seed)
    rng = random.Random(seed ^ 0xF104)
    sessions = []
    used_seeds = set()

    def next_seed():
        workload_seed = rng.randrange(1, 2**63)
        while workload_seed in used_seeds:
            workload_seed = rng.randrange(1, 2**63)
        used_seeds.add(workload_seed)
        return workload_seed

    for split in ('development', 'validation'):
        for topology in TOPOLOGIES:
            count = MATRIX[split]['healthy_sessions']
            for _ in range(count):
                sessions.append(_new_session(
                    split, topology, next_seed(), f'v4-{len(sessions):03d}',
                    healthy=True))

            if split == 'development':
                assignments = [(slot, stratum)
                               for slot in groups[split]
                               for stratum in FAULT_STRATA]
            else:
                # Six held-out GPUs, two assigned to each stratum per topology.
                assignments = list(zip(groups[split], FAULT_STRATA * 2))
                rng.shuffle(assignments)

            for slot, stratum in assignments:
                fault_case = {
                    'mechanism': FAULT_MECHANISM,
                    'quota_scope': 'target_rank_process',
                    'apply': 'static_at_client_start',
                    'physical_slot': slot,
                    'fault_stratum': stratum,
                    'requested_mps_quota_percent': MPS_QUOTA_REQUESTS[split][stratum],
                }
                sessions.append(_new_session(
                    split, topology, next_seed(), f'v4-{len(sessions):03d}',
                    healthy=False, fault_case=fault_case))

    plan = {
        'schema': 'flash-v4-collection-2',
        'status': 'development_protocol',
        'seed': seed,
        'injection_mechanism': FAULT_MECHANISM,
        'mps_execution_mode': 'shared',
        'matrix_per_topology': {split: dict(values)
                                for split, values in MATRIX.items()},
        'blind_target': dict(BLIND_TARGET),
        'budget': dict(BUDGET),
        'partitions_by_physical_slot': groups,
        'sessions': sessions,
        'source_sha256': source_manifest(root),
        'baseline_status': 'candidate_v1_numeric_core_not_yet_runtime_audited',
        'blind_schedule_generated': False,
        'severity_semantics': (
            'fault_stratum is a sampling bucket; requested MPS quota is not a '
            'measured slowdown or error severity. Measure actual compute impact separately.'),
        'requirements': [
            'clock actuation and mps_quota are distinct mechanisms; this plan selects mps_quota only',
            'bind physical slots to GPU UUIDs before collection and map each target slot to its rank process',
            'use the same disjoint physical-device partition across both topologies',
            'each MPS fault case is a separate training session with a unique workload seed',
            'set the target rank process quota once at client start; never switch it mid-session',
            'validation quotas are unseen intermediate requests, not severity ground truth',
            'keep fault truth, quota telemetry, and injection plans outside detector inputs',
            'record shared-GPU background load and control drift; do not alter clocks or external workloads',
            'validation results stay sealed until feature and threshold selection is locked',
            'infrastructure failures and incomplete coverage remain in reports',
            'freeze collector, algorithm, threshold, baseline, and hardware mapping before the blind draw',
            'do not retry or replace final blind samples to improve the score',
        ],
    }
    validate(plan)
    return plan


def validate(plan):
    if plan['blind_schedule_generated'] or plan['blind_target']['schedule_generated']:
        raise ValueError('this entry point cannot create a blind schedule')
    if (plan['injection_mechanism'] != FAULT_MECHANISM
            or plan['mps_execution_mode'] != 'shared'):
        raise ValueError('protocol must select shared mps_quota mode')
    if plan['matrix_per_topology'] != MATRIX or plan['budget'] != BUDGET:
        raise ValueError('collection matrix or bounded budget changed')
    if any(s['split'] not in ('development', 'validation') for s in plan['sessions']):
        raise ValueError('this entry point cannot create a blind schedule')

    groups = plan['partitions_by_physical_slot']
    expected_sizes = {'development': 4, 'validation': 6,
                      'blind_only_fault_devices': 6}
    if any(len(groups[name]) != size for name, size in expected_sizes.items()):
        raise ValueError('physical-device partition sizes must be 4/6/6')
    flat = sum(groups.values(), [])
    if len(flat) != 16 or set(flat) != set(range(16)):
        raise ValueError('physical-device partitions overlap or omit devices')

    seeds = [s['workload_seed'] for s in plan['sessions']]
    ids = [s['session_id'] for s in plan['sessions']]
    if len(set(seeds)) != len(seeds) or len(set(ids)) != len(ids):
        raise ValueError('session identities or workload seeds overlap')
    if len(plan['sessions']) > BUDGET['max_training_sessions']:
        raise ValueError('training-session budget exceeded')

    for split in ('development', 'validation'):
        for topology in TOPOLOGIES:
            selected = [s for s in plan['sessions'] if s['split'] == split
                        and s['topology'] == topology]
            expected = MATRIX[split]
            healthy = [s for s in selected if s['condition'] == 'healthy']
            fault_sessions = [s for s in selected
                              if s['condition'] == 'mps_quota_fault']
            if len(healthy) != expected['healthy_sessions']:
                raise ValueError('incorrect healthy-session count')
            for session in healthy:
                if (session['healthy_windows'] != BUDGET['healthy_windows_per_session']
                        or session['fault_cases']
                        or session['healthy_mps_quota_percent'] != HEALTHY_MPS_QUOTA_PERCENT
                        or session['mps_mode'] != 'shared'):
                    raise ValueError('invalid healthy MPS control session')
            if len(fault_sessions) != sum(expected['faults_per_stratum']):
                raise ValueError('each fault case must have an independent session')
            cases = []
            for session in fault_sessions:
                if (session['healthy_windows'] != 0
                        or session['healthy_mps_quota_percent'] is not None
                        or session['mps_mode'] != 'shared'
                        or len(session['fault_cases']) != 1):
                    raise ValueError('each static MPS fault needs one independent session')
                case = session['fault_cases'][0]
                if (case['mechanism'] != FAULT_MECHANISM
                        or case['quota_scope'] != 'target_rank_process'
                        or case['apply'] != 'static_at_client_start'):
                    raise ValueError('fault must be a static target-rank MPS quota')
                if case['physical_slot'] not in groups[split]:
                    raise ValueError('fault-device holdout violated')
                stratum = case['fault_stratum']
                if case['requested_mps_quota_percent'] != MPS_QUOTA_REQUESTS[split][stratum]:
                    raise ValueError('MPS quota request mismatch')
                cases.append(case)
            for stratum, count in zip(FAULT_STRATA, expected['faults_per_stratum']):
                if sum(c['fault_stratum'] == stratum for c in cases) != count:
                    raise ValueError('incorrect fault-stratum allocation')

    if (len([s for s in plan['sessions'] if s['condition'] == 'healthy'])
            > BUDGET['max_healthy_sessions']
            or len([s for s in plan['sessions'] if s['condition'] == 'mps_quota_fault'])
            > BUDGET['max_fault_sessions']):
        raise ValueError('collection-session budget exceeded')
    active_seconds = sum(
        session['healthy_windows'] * BUDGET['healthy_window_seconds']
        + (BUDGET['fault_session_seconds'] if session['fault_cases'] else 0)
        for session in plan['sessions'])
    if active_seconds > BUDGET['max_active_collection_seconds']:
        raise ValueError('active collection-time budget exceeded')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = make_protocol(args.seed, Path(__file__).resolve().parents[1])
    # Controller-only plan. This file must never be passed to a trainer/detector.
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'sessions': len(plan['sessions']),
                      'fault_cases': sum(len(s['fault_cases']) for s in plan['sessions']),
                      'blind_generated': False}))


if __name__ == '__main__':
    main()
