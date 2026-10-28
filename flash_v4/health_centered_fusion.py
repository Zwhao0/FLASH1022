"""One development candidate: topology-specific healthy centers, shared noise.

Consumes the existing public observation-derived branch scores only. This is
our calibration exploration, not a GREYHOUND/FLASH paper localization rule.
No new GPU samples, hooks, diagnostic counters, or fault labels are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from flash_v4.health_noise_fusion import (
    BRANCHES, MIN_ITERATIONS, MIN_MARGIN_Z, MIN_POSITIVE_FRACTION,
    MIN_STRENGTH_Z, _logs,
)

TOPOLOGIES = ('legacy', 'node_local_pp')
RANKS = tuple(range(16))
CASE_PATTERN = re.compile(r'(legacy|node_local_pp)_development_\d{2}')


def _case(prediction):
    case = prediction.get('batch_case', {})
    if (case.get('reference_role') != 'development_H1'
            or case.get('topology') not in TOPOLOGIES
            or not CASE_PATTERN.fullmatch(case.get('case_key', ''))
            or not case['case_key'].startswith(case['topology'] + '_development_')
            or case.get('condition_role') not in ('healthy_repeat_h2', 'fault_condition')):
        raise ValueError('only R11 development predictions are allowed')
    return case


def _branch_rows(prediction, branch):
    rows = prediction['branches'][branch]['rank_order']
    if (len(rows) != len(RANKS) or any(type(r.get('rank')) is not int for r in rows)
            or {r['rank'] for r in rows} != set(RANKS)):
        raise ValueError('one branch row per rank 0 through 15 is required')
    result = {row['rank']: _logs(row) for row in rows}
    if any(len(values) > 8 for values in result.values()):
        raise ValueError('branch score exceeds the fixed eight-iteration budget')
    return result


def _mad(values, center):
    return 1.4826 * median(abs(value - center) for value in values)


def calibrate(healthy_predictions):
    """H2 estimates drift/noise; its fit is never independent FPR evidence."""
    by_topology = {}
    for prediction in healthy_predictions:
        case = _case(prediction)
        if case['condition_role'] != 'healthy_repeat_h2':
            raise ValueError('calibration requires healthy H2 only')
        topology = case['topology']
        if topology in by_topology:
            raise ValueError('candidate expects one fixed H2 run per topology')
        by_topology[topology] = prediction
    if set(by_topology) != set(TOPOLOGIES):
        raise ValueError('separate healthy calibration for both topologies is required')
    calibration = {'by_topology': {}, 'method': {
        'schema': 'topology-health-centered-shared-noise-1',
        'scope': 'one_development_candidate_not_frozen_for_acceptance',
        'healthy_center': 'median log H2/H1 score per topology, branch and rank',
        'noise': 'one shared scale per topology/branch; max(raw pooled MAD, centered residual MAD, 1e-6)',
        'rank_score': 'maximum positive median standardized healthy-centered branch deviation',
        'strength_floor': MIN_STRENGTH_Z, 'margin_floor': MIN_MARGIN_Z,
        'minimum_valid_iterations': MIN_ITERATIONS,
        'minimum_positive_fraction': MIN_POSITIVE_FRACTION,
        'thresholds_changed_from_v1': False,
        'development_failure_analysis_informed_design': True,
        'fault_labels_used_for_calibration_or_prediction': False,
        'independent_health_fpr_estimated': False,
        'population_false_alarm_guarantee': False,
        'unseen_topology_behavior': 'requires its own healthy reference and calibration',
        'runtime_sampling_changed': False,
    }}
    for topology, prediction in by_topology.items():
        params = {'noise': {}}
        for branch in BRANCHES:
            rows = _branch_rows(prediction, branch)
            if any(len(values) < MIN_ITERATIONS for values in rows.values()):
                raise ValueError('healthy calibration lacks sufficient iterations for every rank')
            centers = {str(rank): median(values.values()) for rank, values in rows.items()}
            pooled = [v for values in rows.values() for v in values.values()]
            residuals = [v - centers[str(rank)] for rank, values in rows.items()
                         for v in values.values()]
            pooled_mad = _mad(pooled, median(pooled))
            residual_mad = _mad(residuals, median(residuals))
            params['noise'][branch] = {
                'center_by_rank': centers,
                'scale': max(pooled_mad, residual_mad, 1e-6),
                'raw_pooled_mad': pooled_mad, 'centered_residual_mad': residual_mad,
                'observations': len(pooled),
            }
        calibration['by_topology'][topology] = params
        ranking = rank_evidence(prediction, calibration)
        params['strength_threshold'] = max(MIN_STRENGTH_Z, ranking[0]['score'])
        params['margin_threshold'] = max(MIN_MARGIN_Z, ranking[0]['score'] - ranking[1]['score'])
    return calibration


def rank_evidence(prediction, calibration):
    case = _case(prediction)
    params = calibration['by_topology'][case['topology']]
    by_rank = {rank: {} for rank in RANKS}
    for branch in BRANCHES:
        noise = params['noise'][branch]
        if not math.isfinite(noise['scale']) or noise['scale'] <= 0:
            raise ValueError('invalid healthy noise scale')
        for rank, values in _branch_rows(prediction, branch).items():
            center = noise['center_by_rank'][str(rank)]
            if not math.isfinite(center):
                raise ValueError('invalid healthy center')
            z = [(v - center) / noise['scale'] for v in values.values()]
            by_rank[rank][branch] = {
                'median_z': median(z) if len(z) >= MIN_ITERATIONS else None,
                'positive_fraction': sum(v > 0 for v in z) / 8,
                'valid_iterations': len(z),
            }
    ranks = []
    for rank, evidence in by_rank.items():
        available = [(b, item) for b, item in evidence.items() if item['median_z'] is not None]
        if available:
            source, item = max(available, key=lambda pair: pair[1]['median_z'])
            score, fraction = max(0.0, item['median_z']), item['positive_fraction']
        else:
            source, score, fraction = None, 0.0, 0.0
        ranks.append({'rank': rank, 'score': score, 'source': source,
                      'positive_fraction': fraction, 'evidence': evidence})
    return sorted(ranks, key=lambda row: (-row['score'], row['rank']))


def decide(prediction, calibration):
    ranks = rank_evidence(prediction, calibration)
    params = calibration['by_topology'][_case(prediction)['topology']]
    top, second = ranks[:2]
    margin = top['score'] - second['score']
    reasons = []
    if not top['score'] > params['strength_threshold']:
        reasons.append('within_healthy_strength_envelope')
    if not margin > params['margin_threshold']:
        reasons.append('ambiguous_rank_margin')
    if top['positive_fraction'] < MIN_POSITIVE_FRACTION:
        reasons.append('insufficient_persistence')
    return {
        'schema': 'health-centered-fusion-1',
        'status': 'development_candidate_not_frozen_or_accepted',
        'batch_case': prediction['batch_case'], 'rank_order': ranks,
        'alarm_rank': None if reasons else top['rank'],
        'abstention_reasons': reasons, 'top_two_margin': margin,
        'fault_labels_used': False,
    }


def fingerprint(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run(input_dir, output_dir):
    # A strict whitelist prevents validation traversal; no glob over raw runs.
    paths = sorted(p for p in input_dir.glob('*.json') if CASE_PATTERN.fullmatch(p.stem))
    expected = {f'{t}_development_{i:02d}' for t in TOPOLOGIES for i in range(1, 14)}
    if {p.stem for p in paths} != expected:
        raise ValueError('expected the 26 fixed development conditions, including failures')
    sources = {path.name: json.loads(path.read_text()) for path in paths}
    complete = []
    for name, prediction in sources.items():
        if prediction.get('status') == 'collection_incomplete':
            if prediction.get('case_key') != Path(name).stem:
                raise ValueError('incomplete development case identity mismatch')
        else:
            if _case(prediction)['case_key'] != Path(name).stem:
                raise ValueError('development case identity mismatch')
            complete.append(prediction)
    calibration = calibrate([p for p in complete
                             if _case(p)['condition_role'] == 'healthy_repeat_h2'])
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new(output_dir / 'calibration.json', calibration)
    summary = []
    for name, prediction in sources.items():
        if prediction.get('status') == 'collection_incomplete':
            result = prediction
            summary.append({'case_key': prediction['case_key'], 'status': 'collection_incomplete'})
        else:
            result = decide(prediction, calibration)
            summary.append({'case': result['batch_case'], 'top_rank': result['rank_order'][0]['rank'],
                            'alarm_rank': result['alarm_rank'],
                            'abstention_reasons': result['abstention_reasons']})
        write_new(output_dir / name, result)
    write_new(output_dir / 'summary.json', {'method': calibration['method'], 'cases': summary})
    # This receipt is saved before the separate evaluator accesses labels.
    write_new(output_dir / 'prediction_hashes.json', {
        'schema': 'v4-development-prediction-receipt-1',
        'input_files': [fingerprint(p) for p in paths],
        'prediction_files': [fingerprint(p) for p in sorted(output_dir.glob('*.json'))],
        'source_files': [fingerprint(Path(__file__)),
                         fingerprint(Path(__file__).with_name('health_noise_fusion.py'))],
        'validation_read': False, 'private_read': False,
    })
    return {'conditions': len(summary), 'output': str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.output_dir)))


if __name__ == '__main__':
    main()
