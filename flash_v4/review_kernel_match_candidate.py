"""Once-only development evaluation of the prespecified kernel fallback.

Verifies the complete feature/prediction provenance before opening the prior
development-only label report. Does not read private truth or validation.
"""
import argparse
import datetime
import json
from pathlib import Path
from statistics import median

from flash_v4.health_centered_fusion import CASE_PATTERN, fingerprint, write_new
from flash_v4.review_development_candidate import aggregate


def verify(items):
    for item in items:
        if fingerprint(Path(item['path'])) != item:
            raise ValueError('source/input/prediction changed before evaluation')


def span(values):
    return {'min': min(values), 'median': median(values), 'max': max(values)}


def review(predictions, features, output):
    if output.exists():
        raise ValueError('refusing repeated evaluation')
    plan_path = Path('results/v4/exploration/r11_development_v3_plan.json')
    plan = json.loads(plan_path.read_text())
    verify(plan['source_files'])
    receipt_path = predictions / 'prediction_hashes.json'
    receipt = json.loads(receipt_path.read_text())
    for field in ('input_files', 'prediction_files', 'source_files'):
        verify(receipt[field])
    feature_receipt_path = features / 'feature_receipt.json'
    feature_receipt = json.loads(feature_receipt_path.read_text())
    for field in ('sources', 'source_code', 'features'):
        verify(feature_receipt[field])
    previous_path = Path('results/v4/evaluation/r11_development_v2/alarm_evaluation.json')
    previous = json.loads(previous_path.read_text())
    cases, transitions, coverage = [], [], []
    for old in previous['development_fault_cases']:
        key = old['case_key']
        if not CASE_PATTERN.fullmatch(key):
            raise ValueError('development labels only')
        new = json.loads((predictions / (key + '.json')).read_text())
        row = {k: old[k] for k in ('case_key', 'session_id', 'topology',
                                   'requested_extra_compute_time_ratio', 'evaluable')}
        if not old['evaluable']:
            if new.get('status') != 'collection_incomplete':
                raise ValueError('failed collection was replaced')
            cases.append(row)
            continue
        if (new['batch_case']['case_key'] != key or new['fault_labels_used'] is not False):
            raise ValueError('case identity or label isolation failed')
        target, top, alarm = old['target_rank'], new['rank_order'][0]['rank'], new['alarm_rank']
        row.update(target_rank=target, candidate_top1_rank=top,
                   candidate_top1_correct=top == target, alarm_rank=alarm,
                   alarm_correct=alarm == target if alarm is not None else False,
                   alarm_wrong_rank=alarm != target if alarm is not None else False,
                   abstention_reasons=new['abstention_reasons'])
        cases.append(row)
        transitions.append({'case_key': key, 'baseline_top_correct': old['candidate_top1_correct'],
                            'candidate_top_correct': top == target,
                            'baseline_alarm_rank': old['alarm_rank'], 'candidate_alarm_rank': alarm,
                            'lost_correct_alarm': old['alarm_correct'] and alarm != target,
                            'new_correct_alarm': not old['alarm_correct'] and alarm == target})
        f = json.loads((features / (key + '.json')).read_text())
        coverage.extend(r['modeled_event_fraction'] for r in
                        f['branches']['kernel']['condition_model_coverage_by_rank'].values())
    overall = aggregate(cases)
    costs = feature_receipt['timings']
    cost = {k: span([r[k] for r in costs]) for k in (
        'baseline_kernel_score_cpu_ms', 'candidate_kernel_score_cpu_ms', 'read_validate_wall_ms')}
    healthy = [json.loads((predictions / f'{t}_development_01.json').read_text())
               for t in ('legacy', 'node_local_pp')]
    lost = [t['case_key'] for t in transitions if t['lost_correct_alarm']]
    eligible = (overall['alarm_wrong_rank_count'] == 0 and not lost
                and overall['actual_alarm_count'] >= previous['overall']['actual_alarm_count']
                and cost['candidate_kernel_score_cpu_ms']['max'] <= plan['cpu_score_budget_ms'])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'v4-development-exact-kernel-review-1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'provenance': [fingerprint(p) for p in (plan_path, receipt_path, feature_receipt_path,
                                               previous_path, Path(__file__))],
        'overall': overall,
        'by_topology': {t: aggregate([c for c in cases if c['topology'] == t])
                        for t in ('legacy', 'node_local_pp')},
        'by_strength': [{'requested_strength': s,
                         **aggregate([c for c in cases if c['requested_extra_compute_time_ratio'] == s])}
                        for s in (0.25, 0.5, 1.0)],
        'transitions': transitions, 'development_fault_cases': cases,
        'kernel_modeled_event_coverage_all_evaluable_fault_ranks': span(coverage),
        'local_processing_cost': cost,
        'cpu_score_budget_ms': plan['cpu_score_budget_ms'],
        'cost_scope': feature_receipt['cost_scope'],
        'calibration_health_not_FPR': [{'case_key': h['batch_case']['case_key'],
                                       'alarm_rank': h['alarm_rank']} for h in healthy],
        'provisional_development_gate_passed': eligible,
        'lost_correct_alarms': lost,
        'validation_read': False, 'new_private_label_lookups': 0,
        'independent_health_fpr_computed': False, 'online_overhead_measured': False,
        'candidate_count_this_round': 1, 'post_evaluation_parameter_changes': False,
        'formal_acceptance_eligible': False,
        'note': 'development gate is not generalization, FPR or online acceptance',
    }
    write_new(output / 'alarm_evaluation.json', report)
    print(json.dumps({'overall': overall, 'coverage': report[
        'kernel_modeled_event_coverage_all_evaluable_fault_ranks'],
        'local_cost': cost, 'development_gate_passed': eligible}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.predictions, args.features, args.output)
