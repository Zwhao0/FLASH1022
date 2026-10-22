"""Two-node automatic detection/profile/localization smoke and post-run scoring."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import secrets

from flash_v1.scripts.data16 import ROOT, NODES, run


def make_online_plan(session, seed):
    rng = random.Random(seed)
    ranks = [rng.randrange(8), rng.randrange(8, 16)]
    rng.shuffle(ranks)
    intervals, start = [], 1
    for name, length, target in [('healthy_baseline', 55, None), ('fault0', 35, ranks[0]),
                                 ('healthy_recovery', 45, None), ('fault1', 35, ranks[1]),
                                 ('healthy_final', 25, None)]:
        intervals.append(dict(id=name, start=start, end=start + length - 1,
                              record_start=start, target_ranks=[] if target is None else [target],
                              delay_ms=0 if target is None else round(rng.uniform(400, 650), 3)))
        start += length
    return dict(session_id=session, role='online_validation', online=True,
                profile_enabled=False, max_profiles=3, seq_len=16, seed=seed,
                fault_kind='simulated_gpu_stream_delay', intervals=intervals)


def score(local):
    local = Path(local)
    plan = json.loads((local / 'plan.json').read_text())
    events = [json.loads(x) for x in (local / 'host2/events.jsonl').read_text().splitlines()]
    training = [json.loads(x) for x in (local / 'host2/train.jsonl').read_text().splitlines()]
    detections = [x for x in events if x['event'] == 'detection']
    schedules = {x['epoch']: x for x in events if x['event'] == 'profile_schedule'}
    results = {x['epoch']: x for x in events if x['event'] == 'localization'}
    faults = [x for x in plan['intervals'] if x['target_ranks']]
    fingerprints, injected, raw_rows = None, [], []
    for index, (host, _, _) in enumerate(NODES):
        node = local / host
        manifest = json.loads((node / f'node{index}_manifest.json').read_text())
        drain = json.loads((node / 'drain_status.json').read_text())
        assert manifest['status'] == 'completed' and drain['status'] == 'complete'
        identity = {k: manifest[k] for k in ('configuration', 'source_sha256', 'probe_sha256',
                                             'snapshot_sha256', 'model_sha256')}
        assert fingerprints is None or fingerprints == identity, 'cross-node runtime mismatch'
        fingerprints = identity
        assert int.from_bytes((node / 'state.bin').read_bytes(), 'little') == 0
        for path in node.glob('injection_rank*.jsonl'):
            injected += [x for x in map(json.loads, path.read_text().splitlines()) if 'injected_cycles' in x]
        with gzip.open(node / 'raw_profile.jsonl.gz', 'rt') as stream:
            raw_rows += [json.loads(x) for x in stream]
    expected_injection = {(step, interval['target_ranks'][0]) for interval in faults
                          for step in range(interval['start'], interval['end'] + 1)}
    assert {(x['iteration'], x['rank']) for x in injected} == expected_injection
    for item in injected:
        interval = next(x for x in faults if x['start'] <= item['iteration'] <= x['end'])
        assert item['hook_calls'] > 0
        assert abs(item['calibrated_injected_ms'] - 2 * interval['delay_ms']) <= max(1, .1 * interval['delay_ms'])
    assert len(schedules) <= plan['max_profiles']
    for row in training:
        epoch = next((e for e, s in schedules.items() if s['start'] <= row['iteration'] <= s['end']), 0)
        assert row['profile_epoch'] == epoch, 'training/probe epoch differs from schedule'
    for raw in raw_rows:
        assert any(s['start'] <= raw['iteration'] <= s['end'] for s in schedules.values()), 'sampling outside requested window'
    windows = []
    for epoch, schedule in sorted(schedules.items()):
        assert schedule['end'] - schedule['start'] + 1 == 8
        rows = [x for x in raw_rows if schedule['start'] <= x['iteration'] <= schedule['end']]
        complete = all(any(x['iteration'] == step and x['rank'] == rank for x in rows)
                       for step in range(schedule['start'], schedule['end'] + 1) for rank in range(16))
        windows.append(dict(epoch=epoch, start=schedule['start'], end=schedule['end'],
                            complete_iteration_rank_coverage=complete, samples=len(rows),
                            predicted=results.get(epoch, {}).get('result', {}).get('predicted'),
                            localization_before_training_end=results.get(epoch, {}).get('wall_time', float('inf')) < training[-1]['walltime']))
    scored = []
    for interval in faults:
        matches = [d for d in detections if interval['start'] <= d['iteration'] <= interval['end']]
        decisions = [results[d['epoch']]['result']['predicted'] for d in matches if d['epoch'] in results]
        expected = f"rank:{interval['target_ranks'][0]}"
        scored.append(dict(id=interval['id'], target_rank=interval['target_ranks'][0],
                           delay_ms_per_backward=interval['delay_ms'], detected=bool(matches),
                           detection_delay_iterations=matches[0]['iteration'] - interval['start'] if matches else None,
                           predicted=decisions, correct=expected in decisions and all(x == expected for x in decisions)))
    healthy_detections = [d for d in detections if not any(x['start'] <= d['iteration'] <= x['end'] for x in faults)]
    report = dict(session=plan['session_id'], scope='real 16 GPU automatic closed-loop quick validation; two random single-GPU stream-delay faults',
                  fault_cases=scored, correct_faults=sum(x['correct'] for x in scored), fault_denominator=len(scored),
                  healthy_detection_count=len(healthy_detections), healthy_detections=healthy_detections,
                  healthy_iterations=sum(not any(x['start'] <= row['iteration'] <= x['end'] for x in faults)
                                         and row['iteration'] > 12 for row in training),
                  profile_windows=windows, localization_results=len(results),
                  ring_drops=0, sampling_disabled_at_finish=True, injection_verified=True,
                  inference_model_sha256=hashlib.sha256((ROOT / 'results/data16_models/wait_frozen.json').read_bytes()).hexdigest())
    (local / 'online_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if args.score_only:
        local = ROOT / 'results/data16_sessions' / args.session
    else:
        plan = make_online_plan(args.session, args.seed if args.seed is not None else secrets.randbits(32))
        local = run(plan, timeout=args.timeout)
    score(local)


if __name__ == '__main__':
    main()
