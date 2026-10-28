"""Private prespecified timing checks excluding whole PROFILE-overlapping blocks."""
import math
from statistics import mean, median, pstdev


def check(rows, window, contract, profile=None):
    first, last = window['first'], window['last']
    baseline = list(range(first - contract['baseline_steps'], first))
    active = list(range(first, last + 1))
    wanted = baseline + active
    selected = [r for r in rows if r.get('iteration') in wanted]
    data = {r['iteration']: r for r in selected}
    if len(selected) != len(wanted) or set(data) != set(wanted):
        return dict(qualified=False, reasons=['missing_or_duplicate_timing'])
    for i in wanted:
        for key in ('time_ms', 'wall_iteration_ms', 'walltime'):
            value = data[i].get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                return dict(qualified=False, reasons=['invalid_timing'])
    if any(data[a]['walltime'] >= data[b]['walltime'] for a, b in zip(wanted, wanted[1:])):
        return dict(qualified=False, reasons=['invalid_wall_order'])
    if profile is not None and (not isinstance(profile, (list, tuple)) or len(profile) != 2
            or any(type(i) is not int for i in profile) or profile[0] > profile[1]):
        return dict(qualified=False, reasons=['invalid_profile_window'])
    if profile and max(baseline[0], profile[0]) <= min(first - 1, profile[1]):
        return dict(qualified=False, reasons=['profile_contaminated_baseline'])
    blocks = [active[i:i + contract['block_steps']] for i in range(0, len(active), contract['block_steps'])]
    clean = [b for b in blocks if len(b) == contract['block_steps']
             and (not profile or b[-1] < profile[0] or b[0] > profile[1])]
    reasons = []
    if len(clean) < contract['min_clean_blocks']:
        return dict(qualified=False, reasons=['insufficient_unprofiled_fixed_blocks'])
    metrics = {}
    for key in ('time_ms', 'wall_iteration_ms'):
        base = [data[i][key] for i in baseline]
        values = [data[i][key] for b in clean for i in b]
        changes = [median(data[i][key] for i in b) / median(base) - 1 for b in clean]
        m = dict(baseline_cv=pstdev(base) / mean(base), median_increase=median(values) / median(base) - 1,
                 mean_increase=mean(values) / mean(base) - 1, block_median_increases=changes)
        metrics[key] = m
        if m['baseline_cv'] > contract['max_baseline_cv']:
            reasons.append(key + '_unstable_baseline')
        if not all(contract['min_increase'] <= m[s] <= contract['max_increase'] for s in ('mean_increase', 'median_increase')):
            reasons.append(key + '_outside_primary_band')
        if min(changes) < contract['min_block_increase']:
            reasons.append(key + '_not_sustained')
    duration = data[last]['walltime'] - data[first - 1]['walltime']
    if duration >= contract['max_fault_seconds']:
        reasons.append('episode_exceeds_time_budget')
    return dict(qualified=not reasons, reasons=reasons, metrics=metrics,
                clean_blocks=[[b[0], b[-1]] for b in clean], wall_seconds=duration,
                measured_unprofiled_steps=sum(map(len, clean)),
                interpretation='fixed clean blocks only; not full-episode uninstrumented timing or measured physical slowdown')
