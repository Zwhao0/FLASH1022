"""Score the predeclared ABBA monitor experiment; never launch training."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ('monopt_a1_baseline', 'monopt_b1_online', 'monopt_b2_online', 'monopt_a2_baseline')


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    location = (len(values) - 1) * fraction
    low = math.floor(location)
    return values[low] + (values[math.ceil(location)] - values[low]) * (location - low)


def timing_statistics(a1, b1, b2, a2):
    if any(len(values) != 88 for values in (a1, b1, b2, a2)):
        raise ValueError('exactly 88 positions (iterations 13..100) required')
    if any(not math.isfinite(value) or value <= 0 for values in (a1, b1, b2, a2) for value in values):
        raise ValueError('wall durations must be finite and positive')
    baseline = [(left + right) / 2 for left, right in zip(a1, a2)]
    online = [(left + right) / 2 for left, right in zip(b1, b2)]
    ratios = [on / base for on, base in zip(online, baseline)]
    blocks = [dict(start=13 + index, end=20 + index,
                   ratio=sum(online[index:index + 8]) / sum(baseline[index:index + 8]))
              for index in range(0, 88, 8)]
    aggregate = (sum(b1) + sum(b2)) / (sum(a1) + sum(a2))
    return dict(iterations=[13, 100], positions_per_run=88,
                aggregate_ratio=aggregate, aggregate_overhead_pct=100 * (aggregate - 1),
                online_sum_ms=sum(b1) + sum(b2), baseline_sum_ms=sum(a1) + sum(a2),
                paired_overhead_pct=dict(B1_A1=100 * (sum(b1) / sum(a1) - 1),
                                         B2_A2=100 * (sum(b2) / sum(a2) - 1)),
                iteration_ratio_median=statistics.median(ratios),
                iteration_ratio_p95=percentile(ratios, .95),
                iteration_overhead_median_pct=100 * (statistics.median(ratios) - 1),
                iteration_overhead_p95_pct=100 * (percentile(ratios, .95) - 1),
                eight_step_blocks=blocks,
                eight_step_ratio_range=[min(row['ratio'] for row in blocks), max(row['ratio'] for row in blocks)],
                measured_aggregate_below_one_percent=aggregate < 1.01)


def build_report(sessions_root):
    paths = {name: Path(sessions_root) / name for name in SESSIONS}
    required = [path / 'plan.json' for path in paths.values()]
    required += [path / host / f'node{node}_manifest.json'
                 for path in paths.values() for node, host in enumerate(('host2', 'host1'))]
    required += [path / 'host2/train.jsonl' for path in paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='missing', missing=missing, sessions=list(SESSIONS))
    checks, runs, manifests, training = {}, {}, [], {}
    for name, path in paths.items():
        plan = read_json(path / 'plan.json')
        nodes = [read_json(path / host / f'node{node}_manifest.json')
                 for node, host in enumerate(('host2', 'host1'))]
        rows = jsonl(path / 'host2/train.jsonl')
        first = [row for row in rows if int(row['iteration']) <= 100]
        checks[f'{name}:exact_first_100_sequence'] = [int(row['iteration']) for row in first] == list(range(1, 101))
        checks[f'{name}:completed_both_nodes'] = all(node.get('status') == 'completed' for node in nodes)
        checks[f'{name}:first_100_healthy_plan'] = not any(
            interval.get('target_ranks') and interval['start'] <= 100
            for interval in plan['intervals'])
        checks[f'{name}:expected_runtime_mode'] = all(
            node.get('online', False) == ('_online' in name)
            and node.get('probe_enabled', True) == ('_online' in name) for node in nodes)
        training[name] = {int(row['iteration']): row for row in first}
        manifests.extend((f'{name}/{host}', node) for host, node in zip(('host2', 'host1'), nodes))
        runs[name] = dict(seed=plan['seed'], total_iterations=len(rows),
                          manifest_sha256={host: hashlib.sha256((path / host / f'node{index}_manifest.json').read_bytes()).hexdigest()
                                           for index, host in enumerate(('host2', 'host1'))},
                          train_sha256=hashlib.sha256((path / 'host2/train.jsonl').read_bytes()).hexdigest())
    checks['same_seed'] = len({row['seed'] for row in runs.values()}) == 1
    alignment = {}
    for key in ('model_sha256', 'configuration', 'source_sha256', 'environment', 'snapshot_sha256'):
        fingerprints = {name: digest(manifest.get(key)) for name, manifest in manifests}
        checks[f'aligned:{key}'] = len(set(fingerprints.values())) == 1 and all(key in manifest for _, manifest in manifests)
        alignment[key] = fingerprints
    probe = {name: manifest.get('probe_sha256') for name, manifest in manifests}
    checks['online_probe_matches'] = len({value for name, value in probe.items() if '_online/' in name}) == 1 and all(
        value is not None for name, value in probe.items() if '_online/' in name)
    checks['baseline_probe_absent'] = all(value is None for name, value in probe.items() if '_baseline/' in name)
    prefix = {}
    for name in (SESSIONS[1], SESSIONS[2]):
        events = jsonl(paths[name] / 'host2/events.jsonl')
        detections = [row for row in events if row['event'] == 'detection' and row['iteration'] <= 100]
        schedules = [row for row in events if row['event'] == 'profile_schedule' and row['start'] <= 100 and row['end'] >= 1]
        profile_steps = [i for i, row in training[name].items() if row.get('profile_epoch', 0)]
        prefix[name] = dict(detections=detections, profile_schedules=schedules, profile_steps=profile_steps)
        checks[f'{name}:healthy_prefix_no_detection_or_profile'] = not (detections or schedules or profile_steps)
    complete = all(checks[f'{name}:exact_first_100_sequence'] for name in SESSIONS)
    loss = None
    statistics_result = None
    if complete:
        differences = [max(float(training[name][i]['loss']) for name in SESSIONS)
                       - min(float(training[name][i]['loss']) for name in SESSIONS) for i in range(1, 101)]
        loss = dict(max_absolute_cross_run_difference=max(differences), tolerance=1e-4,
                    all_finite=all(math.isfinite(float(training[name][i]['loss'])) for name in SESSIONS for i in range(1, 101)))
        checks['first_100_loss_matches'] = loss['all_finite'] and max(differences) < 1e-4
        statistics_result = timing_statistics(*[
            [float(training[name][i]['wall_iteration_ms']) for i in range(13, 101)] for name in SESSIONS])
    else:
        checks['first_100_loss_matches'] = False
    fault_report = None
    b1_events = jsonl(paths[SESSIONS[1]] / 'host2/events.jsonl')
    fault_evidence = dict(
        events=[row for row in b1_events if row['event'] in ('detection', 'profile_schedule', 'localization')],
        drains={host: read_json(paths[SESSIONS[1]] / host / 'drain_status.json')
                if (paths[SESSIONS[1]] / host / 'drain_status.json').is_file() else dict(status='missing')
                for host in ('host2', 'host1')})
    try:
        if not (paths[SESSIONS[1]] / 'host2/events.jsonl').is_file():
            raise FileNotFoundError('B1 fault validation requires host2/events.jsonl')
        existing = paths[SESSIONS[1]] / 'online_report.json'
        if existing.is_file():
            fault_report = read_json(existing)
        else:
            from flash_v1.scripts.online16 import score
            with contextlib.redirect_stdout(io.StringIO()):
                fault_report = score(paths[SESSIONS[1]])
        checks['fault_scoring_succeeded'] = fault_report.get('session') == SESSIONS[1]
    except (AssertionError, OSError, ValueError, KeyError) as error:
        checks['fault_scoring_succeeded'] = False
        fault_report = dict(error=repr(error))
    checks['two_faults_correct'] = (fault_report.get('fault_denominator') == 2
                                   and fault_report.get('correct_faults') == 2)
    checks['no_healthy_detections'] = fault_report.get('healthy_detection_count') == 0
    windows = fault_report.get('profile_windows', [])
    checks['two_complete_profile_windows_before_training_end'] = (len(windows) == 2 and all(
        row.get('complete_iteration_rank_coverage') is True
        and row.get('localization_before_training_end') is True
        and row.get('end', 0) - row.get('start', 0) == 7 for row in windows))
    checks['no_ring_drops'] = (fault_report.get('ring_drops') == 0 and all(
        row.get('status') == 'complete' and row.get('dropped') == []
        for row in fault_evidence['drains'].values()))
    checks['sampling_disabled_at_finish'] = fault_report.get('sampling_disabled_at_finish') is True
    report = dict(status='complete' if all(checks.values()) else 'validation_failed',
                  sessions=list(SESSIONS), run_order='A1 B1 B2 A2', checks=checks,
                  failed_checks=[key for key, passed in checks.items() if not passed],
                  runs=runs, alignment=alignment, probe_sha256=probe,
                  healthy_prefix=prefix, loss_validation=loss, monitor_timing=statistics_result,
                  fault_validation=fault_report, fault_evidence=fault_evidence,
                  target='All online healthy prefix positions 13..100; aggregate monitor overhead <1%.',
                  limitations=['Short serial runs on shared GPUs measure this experiment, not a long-term guarantee.',
                               'All four declared runs and all 88 positions are retained; no lowest-run selection.',
                               'Per-step wall intervals include the preceding step logging/publication tail.',
                               'PROFILE <5% is not inferred by comparing fault PROFILE steps against a healthy baseline.'])
    report['qualified_monitor_target_met'] = (all(checks.values()) and statistics_result is not None
                                              and statistics_result['measured_aggregate_below_one_percent'])
    return report


def main():
    global SESSIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions-root', type=Path, default=ROOT / 'results/data16_sessions')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/monitor_optimization/report.json')
    parser.add_argument('--prefix', default='monopt_', help='Unique version prefix for the four declared runs')
    args = parser.parse_args()
    SESSIONS = tuple(args.prefix + suffix for suffix in ('a1_baseline', 'b1_online', 'b2_online', 'a2_baseline'))
    report = build_report(args.sessions_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(args.output),
                          missing=report.get('missing'), failed_checks=report.get('failed_checks'),
                          monitor_timing=report.get('monitor_timing'),
                          qualified_monitor_target_met=report.get('qualified_monitor_target_met')), indent=2))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
