"""Separate, once-only development evaluation using already opened dev labels.

Never opens private truth, validation observations, or the GPU collection tree.
All existing prediction, calibration and source hashes are checked before labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

from flash_v4.health_centered_fusion import CASE_PATTERN, fingerprint, write_new


def aggregate(cases):
    scored = [c for c in cases if c['evaluable']]
    alarms = [c for c in scored if c['alarm_rank'] is not None]
    silent = [c for c in scored if c['alarm_rank'] is None]
    return {
        'planned_fault_count': len(cases), 'complete_evaluable_fault_count': len(scored),
        'collection_incomplete_fault_count': len(cases) - len(scored),
        'candidate_top1_correct': sum(c['candidate_top1_correct'] for c in scored),
        'actual_alarm_count': len(alarms),
        'alarm_correct_rank_count': sum(c['alarm_correct'] for c in alarms),
        'alarm_wrong_rank_count': sum(c['alarm_wrong_rank'] for c in alarms),
        'fault_no_alarm_count': len(silent),
        'not_triggered_strength_only_count': sum(
            c['abstention_reasons'] == ['within_healthy_strength_envelope'] for c in silent),
        'rejected_ambiguous_margin_count': sum(
            'ambiguous_rank_margin' in c['abstention_reasons'] for c in silent),
        'insufficient_persistence_count_including_overlap': sum(
            'insufficient_persistence' in c['abstention_reasons'] for c in silent),
        'no_alarm_reason_patterns': dict(Counter(
            '|'.join(sorted(c['abstention_reasons'])) for c in silent)),
    }


def review(prediction_dir, output_dir):
    if output_dir.exists():
        raise ValueError('review output exists; refusing repeated evaluation')
    receipt_path = prediction_dir / 'prediction_hashes.json'
    receipt = json.loads(receipt_path.read_text())
    for field in ('prediction_files', 'input_files', 'source_files'):
        for item in receipt[field]:
            if fingerprint(Path(item['path'])) != item:
                raise ValueError('prediction, input or source changed before evaluation')
    # Existing evaluation contains development-only labels; no new private lookup.
    previous_path = Path('results/v4/evaluation/r11_development_v1/alarm_evaluation.json')
    previous = json.loads(previous_path.read_text())
    cases, transitions = [], []
    v0_dir = Path('results/v4/exploration/r11_development_v0')
    v1_dir = Path('results/v4/exploration/r11_development_v1')
    source_hashes = [fingerprint(previous_path)]
    diagnosis = []
    for old in previous['development_fault_cases']:
        key = old['case_key']
        if not CASE_PATTERN.fullmatch(key):
            raise ValueError('non-development label in previous evaluation')
        new = json.loads((prediction_dir / f'{key}.json').read_text())
        case = {k: old[k] for k in ('case_key', 'session_id', 'topology',
                                    'requested_extra_compute_time_ratio', 'evaluable')}
        if not case['evaluable']:
            if new['status'] != 'collection_incomplete':
                raise ValueError('incomplete case was replaced')
            cases.append(case)
            continue
        if (new['batch_case']['case_key'] != key or new['fault_labels_used'] is not False):
            raise ValueError('invalid case identity or fault label use')
        target, top, alarm = old['target_rank'], new['rank_order'][0]['rank'], new['alarm_rank']
        case.update(target_rank=target, candidate_top1_rank=top,
                    candidate_top1_correct=top == target, alarm_rank=alarm,
                    alarm_correct=alarm == target if alarm is not None else False,
                    alarm_wrong_rank=alarm != target if alarm is not None else False,
                    abstention_reasons=new['abstention_reasons'])
        cases.append(case)
        v0_path, v1_path = v0_dir / f'{key}.json', v1_dir / f'{key}.json'
        raw, v1 = json.loads(v0_path.read_text()), json.loads(v1_path.read_text())
        source_hashes.extend((fingerprint(v0_path), fingerprint(v1_path)))
        rows = {r['rank']: r for r in v1['rank_order']}
        kernel_top = raw['branches']['kernel']['rank_order'][0]['rank']
        extra_error = kernel_top == target and not old['candidate_top1_correct']
        transitions.append({'case_key': key, 'v1_top_correct': old['candidate_top1_correct'],
                            'v2_top_correct': top == target, 'v1_alarm': old['alarm_rank'],
                            'v2_alarm': alarm})
        coverage = {}
        for branch in ('gemm', 'kernel'):
            data = raw['branches'][branch]['condition_model_coverage_by_rank'][str(target)]
            coverage[branch] = {k: data[k] for k in (
                'event_count', 'known_event_count', 'unknown_event_count',
                'model_coverage_by_weight', 'unknown_reasons')}
        diagnosis.append({
            'case_key': key, 'target_rank': target, 'v0_kernel_top_rank': kernel_top,
            'v1_top_rank': old['candidate_top1_rank'],
            'v1_added_error_vs_kernel': extra_error,
            'v1_top_source': rows[old['candidate_top1_rank']]['source'],
            'target_branch_z': {b: rows[target]['evidence'][b]['median_z']
                                for b in ('gemm', 'kernel')},
            'v1_no_alarm_reasons': old['abstention_reasons'],
            'target_model_coverage': coverage,
        })
    healthy = []
    for topology in ('legacy', 'node_local_pp'):
        key = f'{topology}_development_01'
        p = json.loads((prediction_dir / f'{key}.json').read_text())
        healthy.append({'case_key': key, 'alarm_rank': p['alarm_rank'],
                        'scope': 'H2_calibration_fit_not_independent_FPR'})
    report = {
        'schema': 'flash-v4-development-candidate-review-1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prediction_receipt': fingerprint(receipt_path),
        'evaluator': fingerprint(Path(__file__)),
        'label_source': fingerprint(previous_path),
        'validation_read': False, 'new_private_label_lookups': 0,
        'candidate_count_this_round': 1, 'post_evaluation_parameter_changes': False,
        'thresholds_selected_using_faults': False,
        'design_informed_by_development_failure_analysis': True,
        'formal_acceptance_eligible': False, 'independent_health_fpr_computed': False,
        'overall': aggregate(cases),
        'by_topology': {t: aggregate([c for c in cases if c['topology'] == t])
                        for t in ('legacy', 'node_local_pp')},
        'by_topology_and_requested_strength': [
            {'topology': t, 'requested_strength': s,
             **aggregate([c for c in cases if c['topology'] == t
                          and c['requested_extra_compute_time_ratio'] == s])}
            for t in ('legacy', 'node_local_pp') for s in (0.25, 0.5, 1.0)],
        'calibration_health_not_FPR': healthy, 'transitions': transitions,
        'development_fault_cases': cases,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new(output_dir / 'alarm_evaluation.json', report)
    write_new(output_dir / 'v1_failure_diagnosis.json', {
        'schema': 'v4-development-v1-failure-diagnosis-1',
        'sources': source_hashes, 'cases': diagnosis,
        'v1_added_errors_vs_kernel': [c['case_key'] for c in diagnosis
                                      if c['v1_added_error_vs_kernel']],
        'kernel_raw_misranked_cases': [c['case_key'] for c in diagnosis
                                        if c['v0_kernel_top_rank'] != c['target_rank']],
        'v1_scale_receipt': fingerprint(v1_dir / 'calibration.json'),
        'scope': 'arithmetic_attribution_and_coverage; not causal proof of remaining misses',
    })
    return report['overall']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.predictions, args.output)))


if __name__ == '__main__':
    main()
