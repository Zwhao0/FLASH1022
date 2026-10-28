"""Private, fixed-window strength calibration; never a detector input or score."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

from flash_v4.health_centered_fusion import fingerprint, write_new


def qualify(rows, window, contract, profile=None):
    first, last = window['first'], window['last']
    baseline_ids = list(range(first - contract['baseline_steps'], first))
    active_ids = list(range(first, last + 1))
    wanted = baseline_ids + active_ids
    selected = [r for r in rows if r.get('iteration') in wanted]
    by_id = {r['iteration']: r for r in selected}
    reasons = []
    if len(selected) != len(wanted) or set(by_id) != set(wanted):
        return dict(qualified=False, reasons=['missing_or_duplicate_timing'])
    if profile is not None:
        if (not isinstance(profile, (list, tuple)) or len(profile) != 2
                or any(type(i) is not int for i in profile) or profile[0] > profile[1]):
            return dict(qualified=False, reasons=['invalid_profile_window'])
        if max(baseline_ids[0], profile[0]) <= min(last, profile[1]):
            reasons.append('profile_contaminates_fixed_measurement_window')
    for row in selected:
        for key in ('time_ms', 'wall_iteration_ms'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                return dict(qualified=False, reasons=['invalid_timing'])
    metrics = {}
    for key in ('time_ms', 'wall_iteration_ms'):
        base = [by_id[i][key] for i in baseline_ids]
        active = [by_id[i][key] for i in active_ids]
        center = median(base)
        blocks = [median(active[i:i + contract['block_steps']]) / center - 1
                  for i in range(0, len(active), contract['block_steps'])]
        metrics[key] = dict(baseline_median_ms=center, baseline_cv=pstdev(base) / mean(base),
                            median_increase=median(active) / center - 1,
                            mean_increase=mean(active) / mean(base) - 1,
                            block_median_increases=blocks)
        if metrics[key]['baseline_cv'] > contract['max_baseline_cv']:
            reasons.append(key + '_unstable_baseline')
        if not all(contract['min_increase'] <= metrics[key][s] <= contract['max_increase']
                   for s in ('median_increase', 'mean_increase')):
            reasons.append(key + '_outside_primary_band')
        if min(blocks) < contract['min_block_increase']:
            reasons.append(key + '_not_sustained')
    wall_times = [by_id[i].get('walltime') for i in wanted]
    if (any(type(v) not in (float, int) or not math.isfinite(v) for v in wall_times)
            or any(a >= b for a, b in zip(wall_times, wall_times[1:]))):
        return dict(qualified=False, reasons=reasons + ['missing_or_invalid_wall_duration'],
                    metrics=metrics)
    # Fixed whole-episode wall duration, including inter-iteration/runtime gaps.
    before, end = by_id[first - 1].get('walltime'), by_id[last].get('walltime')
    if (type(before) not in (float, int) or type(end) not in (float, int)
            or not math.isfinite(before) or not math.isfinite(end) or end <= before):
        reasons.append('missing_or_invalid_wall_duration')
        duration = None
    else:
        duration = end - before
        if duration >= contract['max_fault_seconds']:
            reasons.append('episode_exceeds_time_budget')
    return dict(qualified=not reasons, reasons=reasons, metrics=metrics,
                measured_steps=len(active_ids), wall_seconds=duration,
                local_compute_deterioration_measured=False,
                interpretation='rank0 job-step proxy calibration; no measured physical GPU slowdown')


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    private_path = Path(plan['private_execution'])
    if fingerprint(private_path) != plan['private_execution_receipt']:
        raise ValueError('private calibration plan changed')
    root = Path('results/v4/online') / plan['runtime_release']
    receipt = json.loads((root / 'release_receipt.json').read_text())
    if fingerprint(plan_path) != receipt['plan']:
        raise ValueError('public calibration plan changed after staging')
    truth = json.loads(private_path.read_text())['cases']
    evaluation = json.loads((root / 'development_evaluation_private.json').read_text())
    results = []
    for case in plan['development_GPU_sessions']:
        name = case['case']
        quality = json.loads((root / ('driver_' + name) / 'result.json').read_text())
        timing = root / name / 'final/node0/training/train.jsonl'
        rows = [json.loads(s) for s in timing.read_text().splitlines()]
        result = qualify(rows, truth[name]['window'], plan['strength_contract'], quality.get('window'))
        if (not quality.get('manual_integration_only') or
                quality.get('window') != [case['force_profile_at'], case['force_profile_at'] + 7]):
            result['qualified'] = False
            result['reasons'].append('calibration_profile_window_not_confirmed')
        actor = next(r['actuation_quality'] for r in evaluation['cases'] if r['case'] == name)
        if not (quality.get('nodes_completed') and quality.get('quality_passed') and actor['passed']):
            result['qualified'] = False
            result['reasons'].append('collection_or_actuation_failed')
        results.append(dict(case=name, topology=case['topology'], target_rank=truth[name]['target_rank'],
                            requested_ratio=truth[name]['ratio'], timing=fingerprint(timing), **result))
    output = dict(cases=results, all_qualified=all(r['qualified'] for r in results),
                  plan=fingerprint(plan_path), source=fingerprint(Path(__file__)),
                  calibration_only=True, detector_inputs_changed=False,
                  accuracy_claimed=False, local_compute_deterioration_measured=False)
    write_new(root / 'episode_qualification_private.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
