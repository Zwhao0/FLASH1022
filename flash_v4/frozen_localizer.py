"""Freeze the chosen development rule and apply it once to R11 validation.

Uses development H1 models and H2 calibration throughout. Validation health
windows are test observations, never references or calibration data. Neither
fault targets nor requested severity enter prediction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time

from flash_v4.batch_development_score import FROZEN_MATRIX
from flash_v4.health_centered_fusion import fingerprint, write_new
from flash_v4.health_noise_fusion import MIN_ITERATIONS, MIN_POSITIVE_FRACTION
from flash_v4.kernel_match_candidate import (
    fit_exact, normalize, offsets_from_health, score_kernel,
)
from flash_v4.offline_localizer import InputError, _read_observations, _score_dataset

V0 = Path('results/v4/exploration/r11_development_v0')
V3 = Path('results/v4/exploration/r11_development_v3')
FEATURES = Path('results/v4/exploration/r11_development_v3_features')


def verify(items):
    for item in items:
        if fingerprint(Path(item['path'])) != item:
            raise ValueError('frozen source or input fingerprint mismatch')


def models_from_branch(branch, names):
    return {tuple(m['family']): [m['coefficients'][name] for name in names]
            for m in branch['models']}


def encode_models(models):
    return [{'family': list(key), 'coefficients': value} for key, value in models.items()]


def decode_models(rows):
    return {tuple(row['family']): row['coefficients'] for row in rows}


def decode_exact(rows):
    return {(tuple(r['family']), (tuple(r['grid']), tuple(r['block']), r['shared'])): r['us']
            for r in rows}


def features(data, reference):
    start = time.process_time()
    gemm, _ = _score_dataset(data, decode_models(reference['gemm_models']), 'gemm')
    gemm_records = {rank: {i: [v['known_weighted_actual_us'], v['known_weighted_predicted_us']]
                           for i, v in records.items()}
                    for rank, records in gemm.items()}
    kernel, coverage = score_kernel(data, decode_models(reference['kernel_models']),
                                    decode_exact(reference['kernel_exact']))
    iterations = data[0]['iterations']
    result = {'gemm': {'rank_order': normalize(gemm_records, {
        int(k): v for k, v in reference['gemm_offsets'].items()}, iterations)},
              'kernel': {'rank_order': normalize(kernel, {
                  int(k): v for k, v in reference['kernel_offsets'].items()}, iterations)}}
    return result, coverage, (time.process_time() - start) * 1000


def decide(branches, params):
    ranks = {rank: {'rank': rank, 'evidence': {}} for rank in range(16)}
    for branch in ('gemm', 'kernel'):
        rows = branches[branch]['rank_order']
        if len(rows) != 16 or {r['rank'] for r in rows} != set(range(16)):
            raise ValueError('one score row per rank required')
        noise = params['noise'][branch]
        for row in rows:
            values = row['score_by_iteration'].values()
            z = [(math.log(v) - noise['center_by_rank'][str(row['rank'])]) / noise['scale']
                 for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
                 and math.isfinite(v) and v > 0]
            if len(z) > 8:
                raise ValueError('score window exceeds fixed budget')
            ranks[row['rank']]['evidence'][branch] = {
                'median_z': median(z) if len(z) >= MIN_ITERATIONS else None,
                'positive_fraction': sum(v > 0 for v in z) / 8,
                'valid_iterations': len(z)}
    for row in ranks.values():
        active = [(b, r) for b, r in row['evidence'].items() if r['median_z'] is not None]
        source, best = max(active, key=lambda item: item[1]['median_z']) if active else (
            None, {'median_z': 0, 'positive_fraction': 0})
        row.update(source=source, score=max(0, best['median_z']),
                   positive_fraction=best['positive_fraction'])
    ordered = sorted(ranks.values(), key=lambda r: (-r['score'], r['rank']))
    top, second = ordered[:2]
    gap = top['score'] - second['score']
    reasons = []
    if not top['score'] > params['strength_threshold']:
        reasons.append('within_healthy_strength_envelope')
    if not gap > params['margin_threshold']:
        reasons.append('ambiguous_rank_margin')
    if top['positive_fraction'] < MIN_POSITIVE_FRACTION:
        reasons.append('insufficient_persistence')
    return {'rank_order': ordered, 'alarm_rank': None if reasons else top['rank'],
            'abstention_reasons': reasons, 'top_two_margin': gap}


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    feature_receipt = json.loads((FEATURES / 'feature_receipt.json').read_text())
    for field in ('sources', 'source_code', 'features'):
        verify(feature_receipt[field])
    review = json.loads(Path('results/v4/evaluation/r11_development_v3/alarm_evaluation.json').read_text())
    if review['provisional_development_gate_passed'] is not True:
        raise ValueError('prespecified development gate failed')
    calibration = json.loads((V3 / 'calibration.json').read_text())
    raw = Path('results/v4/compat/v4_delay_20261027_r11')
    references, checks = {}, []
    for topology, prefix in (('legacy', 'dl'), ('node_local_pp', 'dp')):
        old = json.loads((V0 / f'{topology}_development_01.json').read_text())
        healthy = _read_observations(raw / (prefix + '_h1'))
        kernel_models = models_from_branch(old['branches']['kernel'],
                                            ('intercept_us', 'us_per_launched_thread'))
        exact = fit_exact(healthy, kernel_models)
        kernel_records, _ = score_kernel(healthy, kernel_models, exact)
        references[topology] = {
            'gemm_models': encode_models(models_from_branch(old['branches']['gemm'],
                          ('intercept_us', 'us_per_gflop', 'us_per_megabyte'))),
            'gemm_offsets': old['branches']['gemm']['health_time_multiplier_by_rank'],
            'kernel_models': encode_models(kernel_models),
            'kernel_offsets': {str(k): v for k, v in offsets_from_health(kernel_records).items()},
            'kernel_exact': [{'family': list(key[0]), 'grid': list(key[1][0]),
                              'block': list(key[1][1]), 'shared': key[1][2], 'us': value}
                             for key, value in exact.items()],
        }
        # Verify serialized inference recreates existing H2 features, without labels.
        serialized = json.loads(json.dumps(references[topology]))
        h2 = _read_observations(raw / (prefix + '_h2'))
        replay, _, _ = features(h2, serialized)
        prior = json.loads((FEATURES / f'{topology}_development_01.json').read_text())
        for branch in ('gemm', 'kernel'):
            expected = {r['rank']: r['score_by_iteration'] for r in prior['branches'][branch]['rank_order']}
            for row in replay[branch]['rank_order']:
                if any(not math.isclose(v, expected[row['rank']][i], rel_tol=1e-12, abs_tol=1e-12)
                       for i, v in row['score_by_iteration'].items()):
                    raise ValueError('serialized feature replay mismatch')
        checks.append({'topology': topology, 'H2_serialized_feature_replay': 'passed'})
    # Check the general inference adapter against every saved development decision.
    replay_count = 0
    for path in sorted(FEATURES.glob('*_development_*.json')):
        p = json.loads(path.read_text())
        if p.get('status') == 'collection_incomplete':
            continue
        topology = p['batch_case']['topology']
        result = decide(p['branches'], calibration['by_topology'][topology])
        previous = json.loads((V3 / path.name).read_text())
        if (result['alarm_rank'] != previous['alarm_rank']
                or result['abstention_reasons'] != previous['abstention_reasons']
                or any(not math.isclose(a['score'], b['score'], rel_tol=1e-12, abs_tol=1e-12)
                       or a['rank'] != b['rank'] for a, b in zip(
                           result['rank_order'], previous['rank_order']))):
            raise ValueError('general decision adapter replay mismatch')
        replay_count += 1
    write_new(output / 'model.json', {
        'schema': 'flash-v4-frozen-exact-kernel-offline-model-1',
        'references': references, 'calibration': calibration,
        'acceptance_status': 'frozen_for_single_independent_offline_validation_only',
        'validation_used_for_fitting': False, 'online_accepted': False,
    })
    code = ['frozen_localizer.py', 'offline_localizer.py', 'kernel_match_candidate.py',
            'health_noise_fusion.py', 'health_centered_fusion.py', 'batch_development_score.py']
    write_new(output / 'freeze_receipt.json', {
        'model': fingerprint(output / 'model.json'),
        'source_files': [fingerprint(Path(__file__).with_name(name)) for name in code],
        'provenance': [fingerprint(p) for p in (FEATURES / 'feature_receipt.json',
                      V3 / 'prediction_hashes.json', V3 / 'calibration.json',
                      Path('results/v4/evaluation/r11_development_v3/alarm_evaluation.json'))],
        'feature_replay': checks, 'saved_decision_replays_passed': replay_count,
        'validation_read_before_freeze': False,
        'validation_policy': 'one pass, all scheduled cases, failed windows preserved, no refitting from validation',
    })
    print(json.dumps({'frozen_model': str(output / 'model.json'), 'decision_replays_passed': replay_count}))


def predict(model_dir, output):
    # This directory is the once-only run marker, including on interruption.
    output.mkdir(parents=True, exist_ok=False)
    freeze_receipt = json.loads((model_dir / 'freeze_receipt.json').read_text())
    verify([freeze_receipt['model']] + freeze_receipt['source_files'] + freeze_receipt['provenance'])
    model = json.loads((model_dir / 'model.json').read_text())
    matrix = json.loads(FROZEN_MATRIX.read_text())
    selected = [s for s in matrix['session_schedule']['sessions'] if s['phase'] == 'validation']
    if len(selected) != 16:
        raise ValueError('expected the original sixteen validation sessions')
    allowed, counters = {}, {'legacy': 0, 'node_local_pp': 0}
    for session in selected:
        topology = session['topology']
        counters[topology] += 1
        key = f'{topology}_validation_{counters[topology]:02d}'
        allowed[key] = {'session_id': session['session_id'], 'topology': topology,
                        'condition': session['condition']}
    write_new(output / 'validation_whitelist.json', allowed)
    sources, results = [], []
    raw = Path('results/v4/compat') / matrix['release']
    for key, session in allowed.items():
        root = raw / session['session_id'].lower()
        paths = sorted(root.glob('node*/observations/rank_*.jsonl'))
        sources.extend(fingerprint(p) for p in paths)
        try:
            start = time.perf_counter()
            data = _read_observations(root)
            read_ms = (time.perf_counter() - start) * 1000
            branches, coverage, score_ms = features(data, model['references'][session['topology']])
            result = decide(branches, model['calibration']['by_topology'][session['topology']])
            result.update(status='predicted_once_with_frozen_development_model',
                          coverage_by_rank=coverage, read_validate_wall_ms=read_ms,
                          feature_score_cpu_ms=score_ms)
        except InputError as error:
            result = {'status': 'collection_incomplete', 'reason': str(error)}
        result.update(case_key=key, topology=session['topology'], fault_labels_used=False,
                      validation_calibration_used=False)
        write_new(output / (key + '.json'), result)
        results.append({'case_key': key, 'status': result['status'],
                        'alarm_rank': result.get('alarm_rank')})
    write_new(output / 'prediction_hashes.json', {
        'model_receipt': fingerprint(model_dir / 'freeze_receipt.json'),
        'model': fingerprint(model_dir / 'model.json'), 'sources': sources,
        'public_matrix': fingerprint(FROZEN_MATRIX),
        'prediction_files': [fingerprint(p) for p in sorted(output.glob('*.json'))],
        'private_labels_read': False, 'validation_refit': False,
    })
    print(json.dumps({'output': str(output), 'cases': results}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze').add_argument('--output', type=Path, required=True)
    p = commands.add_parser('predict-validation')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.output)
    else:
        predict(args.model_dir, args.output)
