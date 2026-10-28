"""Private post-run explanations; never a detector or a replacement score."""
import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev

from flash_v4.health_centered_fusion import fingerprint, write_new


def segment(rows):
    result = {'samples': len(rows)}
    for key in ('time_ms', 'wall_iteration_ms'):
        values = [r[key] for r in rows if key in r]
        if values:
            result[key] = dict(mean=mean(values), median=median(values),
                               cv=pstdev(values) / mean(values))
    return result


def timing_diagnostics(rows, window, request_iteration=None, profile=None,
                       threshold=0.2, healthy=False):
    first, last = window['first'], window['last']
    baseline = segment([r for r in rows if max(4, first - 30) <= r['iteration'] < first])
    # Exclude PROFILE overhead when judging what MONITOR could have observed.
    stop = min(last + 1, profile[0]) if profile else last + 1
    active = segment([r for r in rows if first <= r['iteration'] < stop])
    changes = {}
    for key in ('time_ms', 'wall_iteration_ms'):
        if key in baseline and key in active:
            changes[key] = {stat: active[key][stat] / baseline[key][stat] - 1
                            for stat in ('mean', 'median')}
    flags = []
    if healthy:
        if request_iteration is not None:
            flags.append('healthy_monitor_request')
    elif request_iteration is None:
        observed = changes.get('time_ms')
        if observed and max(observed.values()) <= threshold:
            flags.append('rank0_step_signal_below_v1_degradation_gate')
        elif observed:
            flags.append('Rbeast_candidate_probability_or_filter_review_needed')
        else:
            flags.append('insufficient_timing_evidence')
    elif request_iteration < first:
        flags.append('monitor_requested_before_injection')
    overlap = (max(0, min(last, profile[1]) - max(first, profile[0]) + 1)
               if profile else 0)
    if profile and not healthy:
        if not overlap:
            flags.append('profile_missed_fault_episode')
        elif overlap != profile[1] - profile[0] + 1:
            flags.append('profile_only_partially_overlapped_fault_episode')
    return dict(baseline=baseline, fault_before_profile=active,
                relative_change=changes, planned_fault_profile_overlap_iterations=overlap,
                profile_wait_after_request_steps=(profile[0] - request_iteration
                    if profile and request_iteration is not None else None), flags=flags,
                interpretation='descriptive hypotheses only; not causal proof or a scoring override')


def diagnose_case(root, truth, quality, threshold):
    train = root / 'final/node0/training/train.jsonl'
    training = [json.loads(line) for line in train.read_text().splitlines()] if train.exists() else []
    requests = [json.loads(p.read_text()) for p in (root / 'final').glob('node*/control/profile_request.json')]
    request = requests[0] if len(requests) == 1 else {}
    timing = timing_diagnostics(training, truth['window'], request.get('iteration'),
        quality.get('window'), threshold, healthy=truth['target_rank'] is None)
    prediction_path = root / 'live/frozen_prediction.json'
    prediction = json.loads(prediction_path.read_text()) if prediction_path.exists() else {}
    actor = {}
    target = truth['target_rank']
    if target is not None:
        paths = list((root / 'final').glob(f'node*/actuation/stream_delay_rank_{target}.jsonl'))
        if len(paths) == 1:
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            fault = [r for r in records if r.get('fault_path_enabled')]
            delayed = [r for r in fault if r.get('requested_delay_us', 0) > 0]
            sums = {k: sum(r.get(k, 0) for r in fault) for k in
                    ('requested_delay_us', 'estimated_compute_us', 'delay_launches',
                     'delay_errors', 'event_error', 'cap_hits', 'time_budget_hits', 'cold',
                     'compute', 'estimated', 'nccl_excluded', 'unknown_excluded')}
            actor = dict(sums=sums, delayed_iterations=[r['iteration'] for r in delayed],
                median_requested_delay_ms=median(r['requested_delay_us'] / 1000 for r in delayed) if delayed else 0,
                phase_requested_ms=[sum(r['phase_requested_us'][p] for r in fault) / 1000 for p in range(3)],
                measured_physical_slowdown=False)
            if any(sums[k] for k in ('delay_errors', 'event_error', 'cap_hits', 'time_budget_hits')):
                timing['flags'].append('native_actor_error_or_budget_censoring')
            if quality.get('window'):
                first, last = quality['window']
                actor['actual_delayed_profile_iterations'] = sum(first <= r['iteration'] <= last for r in delayed)
    return dict(timing=timing, actor=actor, monitor_event=request.get('event'),
                alarm_rank=prediction.get('alarm_rank'),
                abstention_reasons=prediction.get('abstention_reasons', []),
                collection_quality_passed=quality.get('quality_passed'),
                prediction_during_training=quality.get('prediction_during_training', False))


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    private = Path(plan['private_execution'])
    if fingerprint(private)['sha256'] != plan['private_execution_receipt']['sha256']:
        raise ValueError('private plan changed after freeze')
    truth = json.loads(private.read_text())['cases']
    root = Path('results/v4/online') / plan['runtime_release']
    evaluation = json.loads((root / 'development_evaluation_private.json').read_text())
    cases = []
    for row in plan['development_GPU_sessions']:
        case = row['case']
        quality = json.loads((root / ('driver_' + case) / 'result.json').read_text())
        outcome = next(r['outcome'] for r in evaluation['cases'] if r['case'] == case)
        cases.append(dict(case=case, topology=row['topology'], target_rank=truth[case]['target_rank'],
            requested_ratio=truth[case]['ratio'], outcome=outcome,
            **diagnose_case(root / case, truth[case], quality, plan['paper_monitor_parameters']['degradation'])))
    write_new(root / 'quick_diagnostics_private.json', dict(cases=cases,
        plan=fingerprint(plan_path), source=fingerprint(Path(__file__)),
        detector_inputs_changed=False, evaluation_counts_unchanged=evaluation['counts']))
    print(json.dumps({'cases': len(cases), 'counts': evaluation['counts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
