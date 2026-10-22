"""Read-only scoring of the final healthy, true-SGD ABBA monitor experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from flash_v1.scripts.monitor16_report import digest, jsonl, read_json, timing_statistics


ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ('monopt3_a1_baseline', 'monopt3_b1_online', 'monopt3_b2_online', 'monopt3_a2_baseline')


def score(sessions_root):
    paths = {name: Path(sessions_root) / name for name in SESSIONS}
    required = [path / file for path in paths.values() for file in ('plan.json', 'host2/train.jsonl', 'host2/run.json')]
    required += [path / host / f'node{node}_manifest.json'
                 for path in paths.values() for node, host in enumerate(('host2', 'host1'))]
    required += [paths[name] / host / file for name in SESSIONS if '_online' in name
                 for host in ('host2', 'host1') for file in ('drain_status.json', 'state.bin')]
    required += [paths[name] / 'host2/monitor.bin' for name in SESSIONS if '_online' in name]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='missing', missing=missing, sessions=list(SESSIONS),
                    qualified_total_overhead_point_estimate_below_one_percent=False)
    checks, runs, training, identities, online_evidence = {}, {}, {}, [], {}
    for name, path in paths.items():
        online = '_online' in name
        plan, run = read_json(path / 'plan.json'), read_json(path / 'host2/run.json')
        rows = jsonl(path / 'host2/train.jsonl')
        checks[f'{name}:exact_100_iterations'] = [row['iteration'] for row in rows] == list(range(1, 101))
        checks[f'{name}:true_sgd'] = (math.isfinite(float(run.get('lr', 0))) and float(run.get('lr', 0)) > 0
                                      and not run.get('monitor_benchmark', False))
        checks[f'{name}:healthy_plan'] = not any(row.get('target_ranks') or row.get('delay_ms', 0) for row in plan['intervals'])
        checks[f'{name}:no_profile_steps'] = all(row.get('profile_epoch', 0) == 0 for row in rows)
        checks[f'{name}:runtime_mode'] = run.get('online', False) == online
        training[name] = {row['iteration']: row for row in rows}
        runs[name] = dict(seed=plan['seed'], learning_rate=run.get('lr'), total_iterations=len(rows),
                          plan_sha256=hashlib.sha256((path / 'plan.json').read_bytes()).hexdigest(),
                          train_sha256=hashlib.sha256((path / 'host2/train.jsonl').read_bytes()).hexdigest())
        for node, host in enumerate(('host2', 'host1')):
            manifest = read_json(path / host / f'node{node}_manifest.json')
            identities.append((f'{name}/{host}', manifest))
            checks[f'{name}/{host}:completed'] = manifest.get('status') == 'completed'
            checks[f'{name}/{host}:mode'] = (manifest.get('online', False) == online
                                              and manifest.get('probe_enabled', True) == online)
        if online:
            events = [row for row in jsonl(path / 'host2/events.jsonl')
                      if row['event'] in ('detection', 'profile_schedule', 'localization')]
            checks[f'{name}:no_detection_or_profile_events'] = not events
            channel_data = (path / 'host2/monitor.bin').read_bytes()
            channel = None
            if len(channel_data) >= 56:
                magic, written, consumed, request, epoch, start, end = struct.unpack('<8s6Q', channel_data[:56])
                channel = dict(magic=magic.decode(errors='replace'), written=written, consumed=consumed,
                               request_epoch=request, schedule_epoch=epoch, start=start, end=end)
            checks[f'{name}:all_channel_rows_consumed'] = (channel is not None and channel['magic'] == 'FLASHM01'
                                                           and channel['written'] == channel['consumed'] == 100
                                                           and channel['request_epoch'] == channel['schedule_epoch'] == 0)
            drains = {}
            for host in ('host2', 'host1'):
                drain = read_json(path / host / 'drain_status.json')
                drains[host] = drain
                checks[f'{name}/{host}:zero_drops'] = drain.get('status') == 'complete' and drain.get('dropped') == []
                checks[f'{name}/{host}:sampling_disabled'] = int.from_bytes((path / host / 'state.bin').read_bytes(), 'little') == 0
            online_evidence[name] = dict(events=events, channel=channel, drains=drains)
    checks['same_seed'] = len({run['seed'] for run in runs.values()}) == 1
    checks['same_positive_learning_rate'] = len({run['learning_rate'] for run in runs.values()}) == 1
    alignment = {}
    for key in ('source_sha256', 'configuration', 'model_sha256', 'environment', 'snapshot_sha256'):
        alignment[key] = {name: digest(manifest.get(key)) for name, manifest in identities}
        checks[f'aligned:{key}'] = (all(key in manifest and manifest[key] is not None for _, manifest in identities)
                                    and len(set(alignment[key].values())) == 1)
    probes = {name: manifest.get('probe_sha256') for name, manifest in identities}
    online_probes = [value for name, value in probes.items() if '_online/' in name]
    checks['online_probe_same_nonnull'] = all(online_probes) and len(set(online_probes)) == 1
    checks['baseline_no_probe'] = all(value is None for name, value in probes.items() if '_baseline/' in name)
    timing, loss = None, None
    complete = all(checks[f'{name}:exact_100_iterations'] for name in SESSIONS)
    if complete:
        losses = [[float(training[name][i]['loss']) for name in SESSIONS] for i in range(1, 101)]
        finite = all(math.isfinite(value) for row in losses for value in row)
        difference = max(max(row) - min(row) for row in losses) if finite else None
        loss = dict(all_finite=finite, max_absolute_cross_run_difference=difference, tolerance=1e-4)
        checks['loss_matches'] = finite and difference < 1e-4
        try:
            timing = timing_statistics(*[[float(training[name][i]['wall_iteration_ms'])
                                         for i in range(13, 101)] for name in SESSIONS])
            checks['timing_valid'] = True
        except (ValueError, KeyError, TypeError) as error:
            checks['timing_valid'] = False
            timing = dict(error=repr(error))
    else:
        checks['loss_matches'] = checks['timing_valid'] = False
    # Accuracy belongs to a separate real-SGD fault run, outside this healthy timing denominator.
    accuracy_path = Path(sessions_root) / 'monopt2_b1_online/online_report.json'
    accuracy = read_json(accuracy_path) if accuracy_path.is_file() else None
    accuracy_reference = dict(session='monopt2_b1_online', path=str(accuracy_path),
                              available=accuracy is not None,
                              correct_faults=accuracy.get('correct_faults') if accuracy else None,
                              fault_denominator=accuracy.get('fault_denominator') if accuracy else None,
                              healthy_detection_count=accuracy.get('healthy_detection_count') if accuracy else None,
                              validated_two_of_two=bool(accuracy and accuracy.get('correct_faults') == 2
                                                        and accuracy.get('fault_denominator') == 2),
                              included_in_healthy_overhead_qualification=False)
    qualified = all(checks.values()) and timing.get('measured_aggregate_below_one_percent', False)
    return dict(status='complete' if all(checks.values()) else 'validation_failed', sessions=list(SESSIONS),
                checks=checks, failed_checks=[key for key, valid in checks.items() if not valid],
                runs=runs, alignment=alignment, probe_sha256=probes, online_evidence=online_evidence,
                loss_validation=loss, monitor_timing=timing, independent_accuracy_reference=accuracy_reference,
                qualified_total_overhead_point_estimate_below_one_percent=qualified,
                scope='Full online MONITOR versus no-FLASH baseline, healthy true-SGD ABBA; all positions 13..100 retained.',
                limitations=['The <1% qualification is a point estimate for these four healthy runs, not a long-term guarantee.',
                             'Shared-GPU serial run drift remains visible in paired and block statistics; no lowest-run selection.',
                             'Separate fault accuracy is referenced rather than inferred from the healthy timing experiment.',
                             'PROFILE overhead is not measured here.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions-root', type=Path, default=ROOT / 'results/data16_sessions')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/monitor_optimization/report_stage3.json')
    args = parser.parse_args()
    report = score(args.sessions_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
