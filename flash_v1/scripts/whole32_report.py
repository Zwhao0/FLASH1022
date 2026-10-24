"""Read-only whole-compute evidence scoring; every formal trial stays in 32 slots."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from flash_v1.scripts import mixed32_report as base

ROOT = base.ROOT
LAYOUTS = ('legacy', 'node_local_pp')


def _finite(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (not positive or value > 0))


def _close(left, right):
    return _finite(left) and _finite(right) and math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)


def _family(op):
    name = op.split('.')[1] if '.' in op else op
    if name in ('mm', 'addmm', 'bmm', 'baddbmm', 'mv', 'dot', 'matmul', 'linear'):
        return 'gemm'
    return 'attention' if 'attention' in name else 'other_local_compute'


def _truth(run, local, plan):
    checks, errors, catalogs, truth, violations, files = {}, [], {}, [], [], []
    for host in ('host2', 'host1'):
        for path in sorted((local / host).glob('whole_compute_calibration_rank*.json')):
            catalog = base._read(path, errors)
            rank = catalog.get('rank')
            if rank in catalogs:
                checks[f'catalog:{rank}:unique'] = False
            catalogs[rank] = catalog
        for path in sorted((local / host).glob('whole_compute_truth_rank*.jsonl')):
            truth.extend(base._read(path, errors, True))
            files.append(dict(path=str(path), sha256=base.sha256(path)))
    checks['all_rank_healthy_catalogs'] = set(catalogs) == set(range(16))
    coverage = {}
    for rank, catalog in catalogs.items():
        shapes = catalog.get('shapes', {})
        digest = hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()
        checks[f'catalog:{rank}:hash'] = bool(shapes) and digest == catalog.get('catalog_sha256')
        checks[f'catalog:{rank}:loaded_reference'] = bool(catalog.get('reference_catalog_sha256')) and catalog.get('reference_catalog_sha256') == digest
        reference_session = plan.get('whole_compute_reference_session')
        host = 'host2' if rank < 8 else 'host1'
        reference_path = local.parent / str(reference_session) / host / f'whole_compute_calibration_rank{rank}.json'
        reference = base._read(reference_path, errors) if reference_session else {}
        checks[f'catalog:{rank}:pilot_reference'] = bool(reference) and reference.get('catalog_sha256') == digest and reference.get('shapes') == shapes
        reference_plan = base._read(reference_path.parents[1] / 'plan.json', errors) if reference else {}
        checks[f'catalog:{rank}:pilot_layout'] = bool(reference_plan) and reference_plan.get('parallel_layout') == plan.get('parallel_layout')
        checks[f'catalog:{rank}:timings'] = bool(shapes) and all(
            _finite(s.get('median_ms'), True) and s.get('samples_ms')
            and all(_finite(t, True) for t in s['samples_ms'])
            and s.get('phase') in ('fwd', 'bwd') for s in shapes.values())
        coverage[rank] = {phase: sorted({s.get('op') for s in shapes.values() if s.get('phase') == phase})
                          for phase in ('fwd', 'bwd')}
        checks[f'catalog:{rank}:forward_backward_compute'] = all(coverage[rank].values())
        for phase, ops in coverage[rank].items():
            names = {op.split('.')[1] if '.' in op else op for op in ops}
            attention = any('attention' in name for name in names) or (bool(names & {'bmm', 'baddbmm'}) and any('softmax' in name for name in names))
            checks[f'catalog:{rank}:{phase}:gemm_and_attention'] = any(_family(op) == 'gemm' for op in ops) and attention
        checks[f'catalog:{rank}:no_excluded_major_compute'] = not any(_family(op) in ('gemm', 'attention') for op in catalog.get('excluded_cuda_ops', {}))
    faults = [f for f in plan.get('intervals', []) if f.get('target_ranks')]
    expected = {(i, rank) for f in faults for rank in f['target_ranks'] for i in range(f['start'], f['end'] + 1)}
    counts = Counter((r.get('iteration'), r.get('rank')) for r in truth)
    checks['whole_truth_exact_once_coverage'] = set(counts) == expected and all(v == 1 for v in counts.values())
    checks['whole_compute_plan_only'] = bool(faults) and all(f.get('fault_mode') == 'whole_compute'
        and len(f.get('target_ranks', [])) == 1 and _finite(f.get('slowdown_extra_ratio'), True) for f in faults)
    fallback_count, total_calls, observed = 0, 0, {}
    previous_pending = {}
    for row in sorted(truth, key=lambda r: (r.get('rank', -1), r.get('iteration', -1))):
        rank, step = row.get('rank'), row.get('iteration')
        fault = next((f for f in faults if rank in f['target_ranks'] and f['start'] <= step <= f['end']), None)
        reasons = []
        catalog = catalogs.get(rank, {})
        ratio = fault.get('slowdown_extra_ratio') if fault else None
        if not fault or not _close(row.get('extra_ratio'), ratio):
            reasons.append('case_ratio')
        if row.get('catalog_sha256') != catalog.get('catalog_sha256') or not catalog:
            reasons.append('catalog_reference')
        shapes, fallback_rows = catalog.get('shapes', {}), row.get('fallback_shapes', [])
        fallback = {s.get('signature'): s for s in fallback_rows}
        operators = row.get('operators', [])
        if not operators or len({o.get('signature') for o in operators}) != len(operators):
            reasons.append('missing_or_duplicate_operators')
        requests, healthy, calls, buckets = 0., 0., 0, Counter()
        observed.setdefault(rank, set())
        for op in operators:
            key = op.get('signature')
            known, sample = key in shapes, shapes.get(key, fallback.get(key, {}))
            count, duration = op.get('calls'), op.get('healthy_gpu_ms')
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or not _finite(duration, True):
                reasons.append('invalid_operator_calls_or_timing')
                continue
            if not sample or op.get('op') != sample.get('op') or op.get('phase') != sample.get('phase'):
                reasons.append('unknown_shape_or_metadata')
            if not _close(duration, sample.get('median_ms')) or not _close(op.get('ratio'), ratio):
                reasons.append('operator_timing_or_ratio')
            if bool(op.get('fallback')) != (not known):
                reasons.append('fallback_flag')
            if not known:
                fallback_count += count
                ref = shapes.get(sample.get('reference_signature'), {})
                if (sample.get('fallback') != 'same_op_work_units_scaling'
                    or ref.get('op') != op.get('op') or ref.get('phase') != op.get('phase')
                    or not _finite(sample.get('work_ratio'), True)
                    or not _close(sample.get('median_ms'), ref.get('median_ms', 0) * sample.get('work_ratio', 0))
                    or not _close(sample.get('work_units'), ref.get('work_units', 0) * sample.get('work_ratio', 0))
                    or sample.get('calls') != count):
                    reasons.append('invalid_fallback_reference')
            requested = duration * count * ratio if _finite(ratio) else float('nan')
            if not _close(op.get('requested_extra_ms'), requested):
                reasons.append('per_operator_proportion')
            calls += count
            healthy += duration * count
            requests += requested
            buckets[f"{op.get('phase')}:{_family(op.get('op', ''))}"] += requested
            observed[rank].add((op.get('phase'), op.get('op')))
        total_calls += calls
        if set(fallback) != {o.get('signature') for o in operators if o.get('fallback')}:
            reasons.append('unaccounted_fallback_shapes')
        if not (row.get('compute_calls') == calls and _close(row.get('healthy_estimated_ms'), healthy)
                and _close(row.get('requested_extra_ms'), requests)):
            reasons.append('operator_totals')
        model = catalog.get('fma_model', {})
        slope, intercept = model.get('units_per_ms'), model.get('intercept_ms')
        total_injected, total_launches = 0., 0
        actual_buckets = row.get('buckets', {})
        if set(buckets) != set(actual_buckets):
            reasons.append('bucket_coverage')
        for key, detail in actual_buckets.items():
            rounds, launches = detail.get('launch_rounds'), detail.get('injected_launches')
            if (not _finite(slope, True) or not _finite(intercept)
                or not isinstance(rounds, int) or not isinstance(launches, int)
                or min(rounds, launches) < 0):
                reasons.append('fma_calibration')
                continue
            predicted = rounds / slope + intercept * launches
            if not (_close(detail.get('requested_extra_ms'), buckets[key])
                    and _close(detail.get('calibrated_injected_ms'), predicted)):
                reasons.append('bucket_fma_prediction')
            total_injected += predicted
            total_launches += launches
        before, after = row.get('pending_before_ms'), row.get('pending_extra_ms')
        pending = row.get('pending_buckets', {})
        quantum = catalog.get('quantum', {}).get('observed_ms')
        if not (_close(row.get('calibrated_injected_ms'), total_injected)
                and row.get('injected_launches') == total_launches
                and _close(requests, total_injected + after - before)
                and _close(after, sum(pending.values()))):
            reasons.append('fma_credit_conservation')
        if not _finite(quantum, True) or not all(_finite(v) and -.001 <= v < quantum + .001 for v in pending.values()):
            reasons.append('unbounded_pending_credit')
        if rank in previous_pending and not _close(before, previous_pending[rank]):
            reasons.append('pending_continuity')
        previous_pending[rank] = after
        if reasons:
            violations.append(dict(rank=rank, iteration=step, reasons=sorted(set(reasons))))
    checks['injection_truth'] = bool(truth) and not violations
    checks['injection_exact_once_coverage'] = checks['whole_truth_exact_once_coverage']
    strict_measured = bool(truth) and fallback_count == 0
    for rank, catalog in catalogs.items():
        required = {(s.get('phase'), s.get('op')) for s in catalog.get('shapes', {}).values()
                    if _family(s.get('op', '')) in ('gemm', 'attention')}
        checks[f'rank:{rank}:local_major_ops_covered'] = bool(required) and required <= observed.get(rank, set())
    summaries = [{k: v for k, v in row.items() if k not in ('operators', 'fallback_shapes')} for row in truth]
    catalog_summaries = {r: {k: v for k, v in c.items() if k != 'shapes'} for r, c in catalogs.items()}
    return dict(checks=checks, errors=errors, catalogs=catalog_summaries, whole_compute_truth=summaries, truth_files=files,
                strict_measured_shape_coverage=strict_measured, measured_shape_call_fraction=(total_calls - fallback_count) / total_calls if total_calls else None,
                operator_coverage=coverage, observed_operator_coverage={r: sorted(v) for r, v in observed.items()},
                fallback_calls=fallback_count, injection_violations=violations,
                injection_missing=sorted(expected - set(counts)), injection_coverage=len(expected & set(counts)) / len(expected) if expected else None)


def _score_session(local, **kwargs):
    local = Path(local)
    run = base.score_session(local, **kwargs)
    errors = run.setdefault('errors', [])
    plan = base._read(local / 'plan.json', errors)
    truth = _truth(run, local, plan)
    run['checks'].update(truth.pop('checks'))
    errors.extend(truth.pop('errors'))
    run.update(truth)
    manifests = run.get('manifests', {})
    hashes = [m.get('compute_workload_sha256') for m in manifests.values()]
    run['checks']['cross_node:compute_workload_sha256'] = len(hashes) == 2 and bool(hashes[0]) and hashes[0] == hashes[1]
    run['checks']['manifest_plan_matches'] = len(manifests) == 2 and all(m.get('plan') == plan for m in manifests.values())
    for host in ('host2', 'host1'):
        raw = base._read(local / host / 'raw_profile.jsonl.gz', errors, True)
        run['checks'][f'{host}:raw_drain_count'] = run.get('drains', {}).get(host, {}).get('samples') == len(raw)
        run['checks'][f'{host}:raw_valid'] = all(isinstance(r.get('rank'), int) and 0 <= r['rank'] < 16 and _finite(r.get('duration_us'), True) and isinstance(r.get('iteration'), int) for r in raw)
        keys = [(r.get('rank'), r.get('comm'), r.get('event_id')) for r in raw]
        run['checks'][f'{host}:raw_unique_records'] = len(keys) == len(set(keys)) and all(k[2] is not None for k in keys)
    epochs = Counter(d.get('epoch') for d in run.get('all_detections', []))
    run['checks']['unique_detection_epochs'] = all(n == 1 for n in epochs.values())
    for case in run.get('fault_cases', []):
        fault = next(f for f in plan['intervals'] if f.get('id') == case['id'])
        case['slowdown_extra_ratio'] = fault.get('slowdown_extra_ratio')
        measured = case['measured_strength']['timing']['wall_iteration_ms']['excluding_profile']['degradation_pct']
        case['strength_qualified_25pct'] = measured is not None and measured + 1e-9 >= 25
        case['strength_qualified_20pct'] = measured is not None and measured + 1e-9 >= 20
    run['checks']['readable_evidence'] = not errors
    run['failed_checks'] = [k for k, v in run['checks'].items() if not v]
    run['integrity_passed'] = not run['failed_checks']
    run['injection_estimate_semantics'] = 'Per-shape healthy CUDA medians times the case ratio; appended FMA duration uses the frozen calibration model. Live per-step duration is not measured.'
    return run


def score_session(local, **kwargs):
    """Read existing logs only. Missing or corrupt evidence is retained as failure."""
    try:
        return _score_session(local, **kwargs)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        run = base.score_session(local, **kwargs)
        run['checks']['whole_schema_readable'] = False
        run['errors'].append(dict(error=repr(error)))
        run['failed_checks'] = [k for k, v in run['checks'].items() if not v]
        run['integrity_passed'] = False
        return run


def build_report(sessions, *, sessions_root=ROOT / 'results/data16_sessions', cleanup=None, frozen_bundle=None, **kwargs):
    """Fixed 32-case denominator, independent of evidence validity and strength."""
    runs = [score_session(Path(s) if Path(s).is_dir() else Path(sessions_root) / s, **kwargs) for s in sessions]
    observed = [(run.get('parallel_layout'), c) for run in runs for c in run.get('fault_cases', [])]
    slots = []
    for layout in LAYOUTS:
        for rank in range(16):
            matches = [c for l, c in observed if l == layout and c.get('target_rank') == rank]
            if len(matches) == 1:
                case = dict(matches[0], parallel_layout=layout, evidence_case_count=1)
            else:
                case = dict(parallel_layout=layout, target_rank=rank, evidence_case_count=len(matches),
                            detected=False, correct=False, no_detection=True)
            slots.append(case)
    checks = {'exact_two_layout_runs': Counter(r.get('parallel_layout') for r in runs) == Counter(LAYOUTS),
              'runs_integrity': bool(runs) and all(r['integrity_passed'] for r in runs),
              'exact_32_observed_cases': len(observed) == 32}
    for layout in LAYOUTS:
        checks[f'{layout}:16_physical_gpus_once'] = Counter(c.get('target_rank') for l, c in observed if l == layout) == Counter(range(16))
    errors = []
    bundle = Path(frozen_bundle or ROOT / 'results/whole32/frozen')
    freeze = base._read(bundle / 'freeze_manifest.json', errors)
    artifacts = freeze.get('artifacts', {})
    checks['frozen_bundle_complete'] = set(artifacts) == {'design.json'} | {f'{role}_{layout}.json' for role in ('pilot', 'calibration', 'validation', 'formal') for layout in LAYOUTS}
    for name, digest in artifacts.items():
        path = bundle / name
        checks[f'frozen_artifact:{name}'] = path.is_file() and base.sha256(path) == digest
    for run in runs:
        layout = run.get('parallel_layout')
        expected = base._read(bundle / f'formal_{layout}.json', errors) if layout in LAYOUTS else {}
        actual = base._read(Path(run['path']) / 'plan.json', errors)
        checks[f"{run['session']}:frozen_formal_plan"] = bool(expected) and expected == actual
    cleanup = base._read(cleanup or ROOT / 'results/whole32/cleanup.json', errors) if not isinstance(cleanup, dict) else cleanup
    for host in ('host2', 'host1'):
        node = cleanup.get(host, {})
        checks[f'{host}:cleanup'] = node.get('owned_modules') == [] and node.get('shm') == [] and node.get('redis17422_open') is False
    keys = ('detected', 'correct', 'wrong_rank', 'rejected', 'judged_healthy', 'no_detection', 'detected_without_diagnosis')
    accuracy = {key: sum(bool(c.get(key)) for c in slots) for key in keys}
    accuracy.update(denominator=32, observed_fault_count=len(observed))
    by_layout = {layout: dict({key: sum(bool(c.get(key)) for c in slots if c['parallel_layout'] == layout) for key in keys}, denominator=16) for layout in LAYOUTS}
    strength = {f'at_least_{threshold}pct': dict(denominator=32, qualified=sum(bool(c.get(f'strength_qualified_{threshold}pct')) for c in slots),
        correct=sum(bool(c.get(f'strength_qualified_{threshold}pct')) and bool(c.get('correct')) for c in slots)) for threshold in (25, 20)}
    return dict(integrity_passed=all(checks.values()), checks=checks, failed_checks=[k for k, v in checks.items() if not v],
                runs=runs, cases=slots, accuracy=accuracy, accuracy_by_layout=by_layout, strength=strength, cleanup=cleanup, errors=errors,
                frozen_bundle=dict(path=str(bundle), manifest_sha256=base.sha256(bundle / 'freeze_manifest.json') if (bundle / 'freeze_manifest.json').is_file() else None, manifest=freeze),
                interpretation='All 32 slots remain in accuracy. Strength qualification is separate; all detections, diagnoses, unknown shapes, and evidence failures are retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', required=True)
    parser.add_argument('--sessions-root', type=Path, default=ROOT / 'results/data16_sessions')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cleanup', type=Path)
    parser.add_argument('--frozen-bundle', type=Path)
    parser.add_argument('--frozen-sources', type=Path)
    parser.add_argument('--frozen-model', type=Path)
    args = parser.parse_args()
    report = build_report(args.sessions, sessions_root=args.sessions_root, cleanup=args.cleanup, frozen_bundle=args.frozen_bundle,
                          frozen_sources=args.frozen_sources, frozen_model=args.frozen_model)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
    return 0 if report['integrity_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
