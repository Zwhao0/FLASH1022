"""Read-only scoring of mixed fault trials. Integrity and localization accuracy are separate."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
MODES = ('attention_bwd_sleep', 'experts_bwd_sleep', 'attention_fwd_sleep', 'attention_bwd_compute')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, errors, lines=False):
    path = Path(path)
    rows = []
    try:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            if not lines:
                return json.load(stream)
            rows = []
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except (ValueError, TypeError) as error:
                        errors.append(dict(path=str(path), line=number, error=str(error)))
            return rows
    except (OSError, ValueError, EOFError) as error:
        errors.append(dict(path=str(path), error=str(error)))
        return rows if lines else {}


def _summary(rows, key):
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))
              and math.isfinite(row[key]) and row[key] > 0]
    return dict(rows=len(rows), valid_rows=len(values), iterations=[row['iteration'] for row in rows],
                mean_ms=statistics.mean(values) if values else None,
                median_ms=statistics.median(values) if values else None)


def _strength(fault, training, intervals, warmup):
    previous = [row for row in intervals if not row.get('target_ranks') and row['end'] < fault['start']]
    baseline = max(previous, key=lambda row: row['end']) if previous else None
    result = dict(baseline_interval=baseline, timing={})
    for key in ('wall_iteration_ms', 'time_ms'):
        result['timing'][key] = {}
        for name, exclude in (('including_profile', False), ('excluding_profile', True)):
            eligible = [row for row in training if row.get('iteration', 0) > warmup
                        and (not exclude or not row.get('profile_epoch', 0))]
            before = [row for row in eligible if baseline and baseline['start'] <= row['iteration'] <= baseline['end']]
            during = [row for row in eligible if fault['start'] <= row['iteration'] <= fault['end']]
            left, right = _summary(before, key), _summary(during, key)
            ratio = right['mean_ms'] / left['mean_ms'] if left['mean_ms'] and right['mean_ms'] else None
            result['timing'][key][name] = dict(healthy=left, fault=right, ratio=ratio,
                degradation_pct=100 * (ratio - 1) if ratio is not None else None)
    return result


def _decision(predicted, target):
    if predicted == f'rank:{target}':
        return 'correct'
    if isinstance(predicted, str) and predicted.startswith('rank:'):
        return 'wrong_rank'
    return 'healthy' if predicted == 'healthy' else 'rejected'


def _score_session(local, *, frozen_sources=None, frozen_model=None, warmup=12):
    """Return evidence without writing files or launching GPU work.

    frozen_sources: path to JSON {source_path: sha256}, or that mapping.
    frozen_model: path to the frozen inference model, or its SHA256 string.
    Missing, malformed, failed and truncated runs remain in the returned report.
    """
    local, errors, checks = Path(local), [], {}
    plan = _read(local / 'plan.json', errors)
    intervals = plan.get('intervals', [])
    faults = [row for row in intervals if row.get('target_ranks')]
    events = _read(local / 'host2/events.jsonl', errors, True)
    training = _read(local / 'host2/train.jsonl', errors, True)
    detections = [row for row in events if row.get('event') == 'detection']
    schedules = [row for row in events if row.get('event') == 'profile_schedule']
    diagnoses = [row for row in events if row.get('event') == 'localization']
    manifests, drains, injections, raw = {}, {}, [], []
    for index, host in enumerate(('host2', 'host1')):
        node = local / host
        manifest = manifests[host] = _read(node / f'node{index}_manifest.json', errors)
        drain = drains[host] = _read(node / 'drain_status.json', errors)
        checks[f'{host}:completed'] = manifest.get('status') == 'completed'
        checks[f'{host}:online_probe_enabled'] = manifest.get('online') is True and manifest.get('probe_enabled') is True
        checks[f'{host}:no_ring_drops'] = drain.get('status') == 'complete' and drain.get('dropped') == []
        try:
            state = (node / 'state.bin').read_bytes()
            checks[f'{host}:state_off'] = len(state) > 0 and int.from_bytes(state, 'little') == 0
        except OSError as error:
            errors.append(dict(path=str(node / 'state.bin'), error=str(error)))
            checks[f'{host}:state_off'] = False
        for path in sorted(node.glob('injection_rank*.jsonl')):
            injections.extend(_read(path, errors, True))
        raw.extend(_read(node / 'raw_profile.jsonl.gz', errors, True))
    for key in ('configuration', 'source_sha256', 'probe_sha256', 'snapshot_sha256', 'model_sha256'):
        checks[f'cross_node:{key}'] = (all(row.get(key) is not None for row in manifests.values())
                                      and manifests['host2'].get(key) == manifests['host1'].get(key))
    freeze_path = ROOT / 'results/data16_models/freeze.json'
    freeze = _read(freeze_path, errors) if freeze_path.is_file() else {}
    if any(fault.get('fault_mode') == 'attention_bwd_compute' for fault in faults):
        checks['cross_node:compute_workload_sha256'] = (bool(manifests['host2'].get('compute_workload_sha256'))
            and manifests['host2'].get('compute_workload_sha256') == manifests['host1'].get('compute_workload_sha256'))
    frozen_sources = frozen_sources or plan.get('frozen_source_sha256') or freeze.get('inference_source_sha256')
    if isinstance(frozen_sources, (str, Path)):
        frozen_sources = _read(Path(frozen_sources), errors)
    checks['frozen_sources_supplied'] = bool(frozen_sources)
    if isinstance(frozen_sources, dict):
        for source, expected in frozen_sources.items():
            checks[f'frozen_source:{source}'] = all(
                row.get('source_sha256', {}).get(source) == expected for row in manifests.values())
            try:
                checks[f'local_frozen_source:{source}'] = sha256(ROOT / source) == expected
            except OSError:
                checks[f'local_frozen_source:{source}'] = False
    frozen_model = frozen_model or plan.get('inference_model_sha256') or freeze.get('model_file_sha256')
    if frozen_model and (isinstance(frozen_model, Path) or len(str(frozen_model)) != 64):
        try:
            frozen_model = sha256(frozen_model)
        except OSError as error:
            errors.append(dict(path=str(frozen_model), error=str(error)))
            frozen_model = None
    actual_model = ROOT / 'results/data16_models/wait_frozen.json'
    try:
        model_hash = sha256(actual_model)
    except OSError:
        model_hash = None
    checks['frozen_model'] = bool(frozen_model) and frozen_model == model_hash
    expected_steps = [step for row in intervals for step in range(row['start'], row['end'] + 1)]
    checks['training_complete_sequence'] = [row.get('iteration') for row in training] == expected_steps and bool(expected_steps)
    checks['diagnosis_budget'] = max(len(schedules), len(diagnoses)) <= plan.get('max_profiles', 0)
    checks['detection_schedule_linkage'] = all(sum(s.get('epoch') == d.get('epoch') for s in schedules) == 1 for d in detections)
    checks['diagnosis_schedule_linkage'] = all(sum(s.get('epoch') == d.get('epoch') for s in schedules) == 1 for d in diagnoses)
    checks['unique_schedule_epochs'] = len({row.get('epoch') for row in schedules}) == len(schedules)
    checks['profile_epoch_alignment'] = all(row.get('profile_epoch', 0) == next((s['epoch'] for s in schedules
        if s['start'] <= row['iteration'] <= s['end']), 0) for row in training)
    checks['raw_only_in_profile'] = all(any(s['start'] <= row.get('iteration', 0) <= s['end'] for s in schedules) for row in raw)
    windows = []
    for schedule in schedules:
        rows = [row for row in raw if schedule['start'] <= row.get('iteration', 0) <= schedule['end']]
        observed = {(row.get('iteration'), row.get('rank')) for row in rows if row.get('duration_us', 0) > 0}
        expected = {(step, rank) for step in range(schedule['start'], schedule['end'] + 1) for rank in range(16)}
        missing = sorted(expected - observed)
        decisions = [row for row in diagnoses if row.get('epoch') == schedule.get('epoch')]
        complete = schedule['end'] - schedule['start'] == 7 and not missing
        checks[f'profile:{schedule.get("epoch")}:complete_8x16_timed'] = complete
        before_end = bool(decisions) and bool(training) and all(
            row.get('wall_time', float('inf')) < training[-1].get('walltime', float('-inf')) for row in decisions)
        checks[f'profile:{schedule.get("epoch")}:diagnosed_before_end'] = before_end
        windows.append(dict(schedule=schedule, raw_rows=len(rows), missing_timed_iteration_ranks=missing,
                            complete_iteration_rank_coverage=complete, diagnoses=decisions,
                            localization_before_training_end=before_end))
    expected_injection = {(step, rank) for fault in faults for rank in fault['target_ranks']
                          for step in range(fault['start'], fault['end'] + 1)}
    injected = [row for row in injections if 'iteration' in row]
    observed_injection = Counter((row.get('iteration'), row.get('rank')) for row in injected)
    checks['injection_exact_once_coverage'] = (set(observed_injection) == expected_injection
                                             and all(count == 1 for count in observed_injection.values()))
    violations = []
    for row in injected:
        fault = next((fault for fault in faults if fault['start'] <= row.get('iteration', 0) <= fault['end']
                      and row.get('rank') in fault['target_ranks']), None)
        if fault is None:
            violations.append(dict(row=row, reason='unexpected_injection'))
            continue
        mode = fault.get('fault_mode')
        engine = 'cuda_fp32_fma' if mode == 'attention_bwd_compute' else 'cuda_sleep'
        nominal = 2 * fault.get('delay_ms', 0)
        estimate = row.get('calibrated_injected_ms')
        valid = (row.get('fault_mode') == mode and row.get('engine') == engine
                 and row.get('hook_calls', 0) > 0 and isinstance(estimate, (int, float))
                 and math.isfinite(estimate) and nominal > 0 and abs(estimate - nominal) <= .1 * nominal
                 and (row.get('injected_cycles') == 0 if engine == 'cuda_fp32_fma' else row.get('injected_cycles', 0) > 0))
        if not valid:
            violations.append(dict(row=row, reason='mode_engine_hooks_or_calibration_mismatch', nominal_ms=nominal))
    checks['injection_truth'] = not violations
    cases = []
    for fault in faults:
        target = fault['target_ranks'][0]
        matches = [row for row in detections if fault['start'] <= row.get('iteration', 0) <= fault['end']]
        epochs = {row.get('epoch') for row in matches}
        decisions = [row for row in diagnoses if row.get('epoch') in epochs]
        outcomes = [_decision(row.get('result', {}).get('predicted'), target) for row in decisions]
        correct = bool(outcomes) and all(value == 'correct' for value in outcomes)
        cases.append(dict(id=fault.get('id'), target_rank=target, physical_gpu=target, fault_mode=fault.get('fault_mode'),
            delay_ms=fault.get('delay_ms'), target_degradation_pct=fault.get('target_degradation_pct'),
            start=fault['start'], end=fault['end'], detected=bool(matches), correct=correct,
            wrong_rank='wrong_rank' in outcomes, rejected='rejected' in outcomes, judged_healthy='healthy' in outcomes,
            no_detection=not matches, detected_without_diagnosis=bool(matches) and not decisions,
            detection_delay_iterations=min((row['iteration'] - fault['start'] for row in matches), default=None),
            detections=matches, diagnoses=decisions, decision_outcomes=outcomes,
            measured_strength=_strength(fault, training, intervals, warmup)))
    healthy = [row for row in detections if not any(fault['start'] <= row.get('iteration', 0) <= fault['end'] for fault in faults)]
    checks['readable_evidence'] = not errors
    return dict(session=plan.get('session_id', local.name), path=str(local), parallel_layout=plan.get('parallel_layout', 'legacy'),
        integrity_passed=all(checks.values()), checks=checks, failed_checks=[key for key, passed in checks.items() if not passed],
        errors=errors, fault_denominator=len(cases), correct_faults=sum(case['correct'] for case in cases), fault_cases=cases,
        healthy_detection_count=len(healthy), healthy_detections=healthy,
        healthy_iterations=sum(row.get('iteration', 0) > warmup and not any(f['start'] <= row['iteration'] <= f['end'] for f in faults) for row in training),
        profile_windows=windows, all_detections=detections, all_localizations=diagnoses, all_profile_schedules=schedules,
        diagnosis_count=len(diagnoses), profile_budget=plan.get('max_profiles'), drains=drains,
        ring_drops=sum(row.get('missed', 0) for drain in drains.values() for row in drain.get('dropped', [])),
        sampling_disabled_at_finish=all(checks.get(f'{host}:state_off') for host in manifests),
        injections=injections, injection_violations=violations,
        injection_estimate_semantics='calibrated_injected_ms sums pre-calibrated CUDA event workload durations; it is not a live per-step duration measurement.',
        injection_missing=sorted(expected_injection - set(observed_injection)),
        injection_coverage=len(expected_injection & set(observed_injection)) / len(expected_injection) if expected_injection else None,
        inference_model_sha256=model_hash, manifests=manifests,
        interpretation='integrity_passed checks evidence completeness only; consult correct_faults and every diagnosis for accuracy.')


def score_session(local, *, frozen_sources=None, frozen_model=None, warmup=12):
    """Read-only scorer; preserve corrupt or failed sessions rather than discarding them.

    See _score_session for the frozen hash parameter formats.
    """
    try:
        return _score_session(local, frozen_sources=frozen_sources, frozen_model=frozen_model, warmup=warmup)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        local, errors = Path(local), []
        evidence = {str(path.relative_to(local)): _read(path, errors, path.suffix == '.jsonl')
                    for path in sorted(local.rglob('*.json*')) if path.suffix in ('.json', '.jsonl')}
        plan = evidence.get('plan.json', {})
        return dict(session=plan.get('session_id', local.name) if isinstance(plan, dict) else local.name,
                    path=str(local), parallel_layout=plan.get('parallel_layout', 'legacy') if isinstance(plan, dict) else 'unknown',
                    integrity_passed=False, checks={'schema_readable': False}, failed_checks=['schema_readable'],
                    errors=errors + [dict(error=repr(error))], fault_denominator=0, correct_faults=0,
                    fault_cases=[], evidence=evidence)


def build_report(sessions, *, sessions_root=ROOT / 'results/data16_sessions', **kwargs):
    runs = [score_session(Path(session) if Path(session).is_dir() else Path(sessions_root) / session, **kwargs)
            for session in sessions]
    checks = {}
    for layout in ('legacy', 'node_local_pp'):
        cases = [case for run in runs if run['parallel_layout'] == layout for case in run['fault_cases']]
        checks[f'{layout}:16_physical_gpus_once'] = Counter(case['target_rank'] for case in cases) == Counter(range(16))
        checks[f'{layout}:four_each_mode'] = Counter(case['fault_mode'] for case in cases) == Counter({mode: 4 for mode in MODES})
    checks['exact_two_layout_runs'] = Counter(run['parallel_layout'] for run in runs) == Counter(('legacy', 'node_local_pp'))
    checks['runs_integrity'] = all(run['integrity_passed'] for run in runs)
    cases = [case for run in runs for case in run['fault_cases']]
    accuracy = {key: sum(bool(case[key]) for case in cases) for key in
                ('detected', 'correct', 'wrong_rank', 'rejected', 'judged_healthy', 'no_detection', 'detected_without_diagnosis')}
    accuracy['denominator'] = len(cases)
    return dict(integrity_passed=all(checks.values()), checks=checks,
                failed_checks=[key for key, passed in checks.items() if not passed], runs=runs, accuracy=accuracy,
                interpretation='Integrity success does not mean localization success. Outcomes may overlap when repeated diagnoses disagree.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', required=True)
    parser.add_argument('--sessions-root', type=Path, default=ROOT / 'results/data16_sessions')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mixed32/report.json')
    parser.add_argument('--frozen-sources', type=Path)
    parser.add_argument('--frozen-model', type=Path)
    args = parser.parse_args()
    report = build_report(args.sessions, sessions_root=args.sessions_root,
                          frozen_sources=args.frozen_sources, frozen_model=args.frozen_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), integrity_passed=report['integrity_passed'], accuracy=report['accuracy'])))
    return 0 if report['integrity_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
