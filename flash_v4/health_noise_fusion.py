"""Development-only, health-calibrated evidence fusion; never reads fault labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

BRANCHES = ('gemm', 'kernel')
MIN_ITERATIONS = 6
MIN_POSITIVE_FRACTION = 0.75
MIN_STRENGTH_Z = 3.5
MIN_MARGIN_Z = 1.0


def _logs(row):
    result = {}
    for iteration, value in row.get('score_by_iteration', {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value) and value > 0:
                result[str(iteration)] = math.log(value)
    return result


def rank_evidence(prediction, calibration):
    """An unaffected branch cannot cancel sustained positive evidence elsewhere."""
    by_rank = {rank: {} for rank in range(16)}
    for branch in BRANCHES:
        params = calibration['noise'][branch]
        for row in prediction['branches'][branch]['rank_order']:
            values = _logs(row)
            scale = params['by_rank'][str(row['rank'])]['scale']
            z = {i: (value - params['center']) / scale
                 for i, value in values.items()}
            by_rank[row['rank']][branch] = {
                'median_z': median(z.values()) if len(z) >= MIN_ITERATIONS else None,
                'positive_fraction': sum(v > 0 for v in z.values()) / 8,
                'valid_iterations': len(z),
            }
    ranks = []
    for rank, evidence in by_rank.items():
        available = [(branch, item) for branch, item in evidence.items()
                     if item['median_z'] is not None]
        if available:
            source, strongest = max(available, key=lambda pair: pair[1]['median_z'])
            score = max(0.0, strongest['median_z'])
            persistence = strongest['positive_fraction']
        else:
            source, score, persistence = None, 0.0, 0.0
        ranks.append({'rank': rank, 'score': score, 'source': source,
                      'positive_fraction': persistence, 'evidence': evidence})
    ranks.sort(key=lambda row: (-row['score'], row['rank']))
    return ranks


def calibrate(healthy_predictions):
    if not healthy_predictions:
        raise ValueError('independent healthy calibration predictions required')
    calibration = {'noise': {}, 'healthy_calibration_windows': len(healthy_predictions)}
    for branch in BRANCHES:
        values = [value for prediction in healthy_predictions
                  for row in prediction['branches'][branch]['rank_order']
                  for value in _logs(row).values()]
        if not values:
            raise ValueError(f'no healthy evidence for {branch}')
        center = median(values)
        # A numerical floor, not a fault-magnitude threshold.
        scale = max(1.4826 * median(abs(v - center) for v in values), 1e-6)
        calibration['noise'][branch] = {
            'center': center, 'scale': scale, 'observations': len(values),
            'by_rank': {},
        }
        for rank in range(16):
            local = [v for prediction in healthy_predictions
                     for row in prediction['branches'][branch]['rank_order']
                     if row['rank'] == rank for v in _logs(row).values()]
            if not local:
                raise ValueError(f'no healthy calibration for {branch}/rank{rank}')
            local_center = median(local)
            local_mad = 1.4826 * median(abs(v - local_center) for v in local)
            # Normal GPU differences and drift are learned from healthy data;
            # rank is a calibration key, never a fault-identity feature.
            local_scale = max(scale, local_mad, abs(local_center - center))
            calibration['noise'][branch]['by_rank'][str(rank)] = {
                'scale': local_scale, 'healthy_drift': local_center - center,
                'observations': len(local),
            }
    null_rankings = [rank_evidence(p, calibration) for p in healthy_predictions]
    calibration['strength_threshold'] = max(MIN_STRENGTH_Z,
                                            *(r[0]['score'] for r in null_rankings))
    calibration['margin_threshold'] = max(MIN_MARGIN_Z,
                                          *(r[0]['score'] - r[1]['score']
                                            for r in null_rankings))
    calibration['method'] = {
        'noise': 'branch healthy log median; scale=max(pooled MAD, rank MAD, absolute rank healthy drift)',
        'rank_score': 'maximum positive branch median standardized deviation',
        'thresholds': 'healthy joint maxima with fixed conservative floors: strength 3.5 robust scales, gap 1 scale; not fitted to fault labels',
        'minimum_valid_iterations': MIN_ITERATIONS,
        'minimum_positive_fraction': MIN_POSITIVE_FRACTION,
        'calibration_false_alarm_is_not_validation': True,
        'population_false_alarm_guarantee': False,
        'fault_labels_used': False,
    }
    return calibration


def decide(prediction, calibration):
    ranks = rank_evidence(prediction, calibration)
    top, second = ranks[:2]
    margin = top['score'] - second['score']
    reasons = []
    if not top['score'] > calibration['strength_threshold']:
        reasons.append('within_healthy_strength_envelope')
    if not margin > calibration['margin_threshold']:
        reasons.append('ambiguous_rank_margin')
    if top['positive_fraction'] < MIN_POSITIVE_FRACTION:
        reasons.append('insufficient_persistence')
    return {
        'schema': 'health-noise-fusion-1',
        'status': 'development_candidate_not_frozen_or_accepted',
        'batch_case': prediction['batch_case'],
        'rank_order': ranks,
        'alarm_rank': None if reasons else top['rank'],
        'abstention_reasons': reasons,
        'top_two_margin': margin,
        'fault_labels_used': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    predictions, incomplete = [], []
    for path in sorted(args.input_dir.glob('*.json')):
        if path.name == 'summary.json':
            continue
        prediction = json.loads(path.read_text())
        if prediction.get('status') == 'collection_incomplete':
            if '_development_' not in prediction.get('case_key', ''):
                raise ValueError('only development records are allowed')
            incomplete.append((path.name, prediction))
            continue
        case = prediction.get('batch_case', {})
        if case.get('reference_role') != 'development_H1':
            raise ValueError('only development predictions are allowed')
        predictions.append((path.name, prediction))
    healthy = [p for _, p in predictions
               if p['batch_case']['condition_role'] == 'healthy_repeat_h2']
    calibration = calibrate(healthy)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'calibration.json').open('x') as f:
        json.dump(calibration, f, indent=2)
    summary = []
    for name, prediction in predictions:
        result = decide(prediction, calibration)
        with (args.output_dir / name).open('x') as f:
            json.dump(result, f, indent=2)
        summary.append({'case': result['batch_case'],
                        'top_rank': result['rank_order'][0]['rank'],
                        'score': result['rank_order'][0]['score'],
                        'alarm_rank': result['alarm_rank'],
                        'abstention_reasons': result['abstention_reasons']})
    for name, record in incomplete:
        with (args.output_dir / name).open('x') as f:
            json.dump(record, f, indent=2)
        summary.append({'case_key': record['case_key'],
                        'status': 'collection_incomplete'})
    with (args.output_dir / 'summary.json').open('x') as f:
        json.dump({'method': calibration['method'], 'cases': summary}, f, indent=2)


if __name__ == '__main__':
    main()
