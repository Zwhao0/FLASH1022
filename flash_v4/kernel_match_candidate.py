"""Bounded development candidate: exact-shape fallback for unmodeled kernels.

Existing fitted families keep their model. Only otherwise unmodeled families
may use an H1 exact-key median, supported by >=2 ranks and >=6 iterations.
These are sampled launch envelopes, not pure kernel timings. No new hooks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import time

from flash_v4.batch_development_score import _development_sessions, FROZEN_MATRIX
from flash_v4.health_centered_fusion import fingerprint, write_new
from flash_v4.offline_localizer import (
    EXPECTED_RANKS, InputError, _kernel_family_and_features, _predict,
    _read_observations, _score_dataset,
)


def fit_exact(healthy, models):
    groups = defaultdict(lambda: defaultdict(list))
    iterations = defaultdict(set)
    for rank, content in healthy.items():
        for row in content['event_rows']:
            if row.get('kind') != 'kernel':
                continue
            key, sample, reason = _kernel_family_and_features(row)
            if reason is None and key[0] not in models:
                groups[key][rank].append(sample['elapsed_us'])
                iterations[key].add(row['iteration'])
    return {key: median(median(values) for values in by_rank.values())
            for key, by_rank in groups.items()
            if len(by_rank) >= 2 and len(iterations[key]) >= 6}


def score_kernel(data, models, exact):
    records, coverage = {}, {}
    for rank, content in data.items():
        per_iteration = defaultdict(lambda: [0.0, 0.0])
        counts = {'event_count': 0, 'regression_events': 0,
                  'exact_fallback_events': 0, 'unknown_events': 0}
        for row in content['event_rows']:
            if row.get('kind') != 'kernel':
                continue
            counts['event_count'] += 1
            key, sample, reason = _kernel_family_and_features(row)
            if reason is not None:
                counts['unknown_events'] += 1
                continue
            model = models.get(key[0])
            predicted = _predict(model, sample['features']) if model is not None else exact.get(key)
            if predicted is None:
                counts['unknown_events'] += 1
                continue
            counts['regression_events' if model is not None else 'exact_fallback_events'] += 1
            totals = per_iteration[row['iteration']]
            totals[0] += sample['elapsed_us'] * sample['sample_weight']
            totals[1] += predicted * sample['sample_weight']
        records[rank] = dict(per_iteration)
        counts['modeled_event_fraction'] = (
            (counts['regression_events'] + counts['exact_fallback_events'])
            / counts['event_count'] if counts['event_count'] else 0)
        coverage[str(rank)] = counts
    return records, coverage


def offsets_from_health(records):
    offsets = {}
    for rank in EXPECTED_RANKS:
        ratios = [actual / predicted for actual, predicted in records[rank].values()
                  if actual > 0 and predicted > 0]
        if len(ratios) != 8:
            raise InputError('candidate H1 lacks eight modeled iterations per rank')
        offsets[rank] = median(ratios)
    return offsets


def normalize(records, offsets, iterations):
    scores = {rank: {} for rank in EXPECTED_RANKS}
    for iteration in iterations:
        raw = {rank: records[rank][iteration][0] / records[rank][iteration][1] / offsets[rank]
               for rank in EXPECTED_RANKS if iteration in records[rank]
               and records[rank][iteration][0] > 0 and records[rank][iteration][1] > 0}
        if len(raw) != 16:
            raise InputError('candidate requires kernel evidence on every rank/iteration')
        drift = median(raw.values())
        for rank, value in raw.items():
            scores[rank][str(iteration)] = value / drift
    rows = [{'rank': rank, 'score_by_iteration': values,
             'score_median': median(values.values())} for rank, values in scores.items()]
    return sorted(rows, key=lambda r: (-r['score_median'], r['rank']))


def timed_score(data, models, exact):
    # Each baseline and candidate runs once on the same already-parsed input.
    start = time.process_time()
    _score_dataset(data, models, 'kernel')
    baseline_ms = (time.process_time() - start) * 1000
    start = time.process_time()
    records, coverage = score_kernel(data, models, exact)
    candidate_ms = (time.process_time() - start) * 1000
    return records, coverage, {'baseline_kernel_score_cpu_ms': baseline_ms,
                               'candidate_kernel_score_cpu_ms': candidate_ms}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    release, schedules = _development_sessions(FROZEN_MATRIX)
    raw_root = Path('results/v4/compat') / release
    v0 = Path('results/v4/exploration/r11_development_v0')
    sources = [fingerprint(FROZEN_MATRIX)]
    timings, references = [], {}
    for topology, schedule in schedules.items():
        ref = next(s for s in schedule if s['role'] == 'reference_h1')
        root = raw_root / ref['session_id'].lower()
        start = time.perf_counter()
        healthy = _read_observations(root)
        read_ms = (time.perf_counter() - start) * 1000
        model_path = v0 / f'{topology}_development_01.json'
        prior = json.loads(model_path.read_text())['branches']['kernel']['models']
        models = {tuple(m['family']): [m['coefficients']['intercept_us'],
                                      m['coefficients']['us_per_launched_thread']]
                  for m in prior}
        start = time.process_time()
        exact = fit_exact(healthy, models)
        fit_ms = (time.process_time() - start) * 1000
        records, coverage = score_kernel(healthy, models, exact)
        offsets = offsets_from_health(records)
        references[topology] = {
            'regression_families': len(models), 'fallback_exact_keys': len(exact),
            'healthy_coverage_by_rank': coverage, 'exact_reference_fit_cpu_ms': fit_ms,
            'reference_read_validate_wall_ms': read_ms,
        }
        sources.append(fingerprint(model_path))
        sources.extend(fingerprint(p) for p in sorted(root.glob('node*/observations/rank_*.jsonl')))
        conditions = [s for s in schedule if s['role'] != 'reference_h1']
        for ordinal, condition in enumerate(conditions, 1):
            key = f'{topology}_development_{ordinal:02d}'
            prior_path = v0 / (key + '.json')
            prior = json.loads(prior_path.read_text())
            sources.append(fingerprint(prior_path))
            if prior.get('status') == 'collection_incomplete':
                write_new(output / (key + '.json'), prior)
                continue
            root = raw_root / condition['session_id'].lower()
            start = time.perf_counter()
            data = _read_observations(root)
            read_ms = (time.perf_counter() - start) * 1000
            if data[0]['iterations'] != healthy[0]['iterations']:
                raise InputError('reference and condition iteration windows do not match')
            records, coverage, timing = timed_score(data, models, exact)
            rows = normalize(records, offsets, data[0]['iterations'])
            write_new(output / (key + '.json'), {
                'batch_case': prior['batch_case'],
                'branches': {'gemm': prior['branches']['gemm'],
                             'kernel': {'branch': 'kernel', 'rank_order': rows,
                                        'condition_model_coverage_by_rank': coverage}},
                'feature_method': 'regression_or_supported_exact_H1_fallback',
                'fault_labels_used': False,
            })
            timings.append({'case_key': key, 'read_validate_wall_ms': read_ms, **timing})
            sources.extend(fingerprint(p) for p in sorted(root.glob('node*/observations/rank_*.jsonl')))
    write_new(output / 'feature_receipt.json', {
        'schema': 'v4-exact-kernel-feature-receipt-1',
        'sources': sources,
        'source_code': [fingerprint(Path(__file__)),
                        fingerprint(Path(__file__).with_name('offline_localizer.py')),
                        fingerprint(Path(__file__).with_name('batch_development_score.py'))],
        'features': [fingerprint(p) for p in sorted(output.glob('*_development_*.json'))],
        'references': references, 'timings': timings,
        'validation_read': False, 'private_read': False, 'new_gpu_collection': False,
        'cost_scope': 'one local CPU pass per 16-rank window; excludes JSON parsing, fusion and GPU instrumentation',
        'counter_role': 'coverage QA only; counts never enter localization scores',
    })
    print(json.dumps({'output': str(output), 'scored_conditions': len(timings),
                      'fallback_keys': {t: r['fallback_exact_keys'] for t, r in references.items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()
