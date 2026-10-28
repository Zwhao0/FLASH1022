"""Separate one-time validation labels/metrics, after frozen predictions exist."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from flash_v4.frozen_localizer import verify
from flash_v4.health_centered_fusion import fingerprint, write_new
from flash_v4.review_development_candidate import aggregate


def review(predictions, output):
    if output.exists():
        raise ValueError('validation evaluation exists; refusing a repeated pass')
    plan_path = Path('results/v4/frozen/r11_kernel_match_v3/validation_once_plan.json')
    plan = json.loads(plan_path.read_text())
    verify(plan['source_files'])
    verify([plan['freeze_receipt']])
    receipt_path = predictions / 'prediction_hashes.json'
    receipt = json.loads(receipt_path.read_text())
    verify(receipt['prediction_files'] + receipt['sources'] + [receipt['model'], receipt['model_receipt'], receipt['public_matrix']])
    freeze = json.loads(Path(receipt['model_receipt']['path']).read_text())
    verify(freeze['source_files'] + freeze['provenance'])
    matrix = json.loads(Path(receipt['public_matrix']['path']).read_text())
    allowed = json.loads((predictions / 'validation_whitelist.json').read_text())
    sessions = {s['session_id']: s for s in matrix['session_schedule']['sessions']
                if s['phase'] == 'validation'}
    if len(allowed) != 16 or {s['session_id'] for s in allowed.values()} != set(sessions):
        raise ValueError('only the original sixteen validation cases are allowed')
    # This is the first validation-label access, after all fingerprints and whitelist.
    private_path = Path('results/v4/private/gpu_delay_execution_r11.json')
    private = json.loads(private_path.read_text())
    if private['public_plan_sha256'] != receipt['public_matrix']['sha256']:
        raise ValueError('private truth is not bound to the original frozen plan')
    fault_cases, healthy_cases, lookups = [], [], []
    for key, case in allowed.items():
        session = sessions[case['session_id']]
        p = json.loads((predictions / (key + '.json')).read_text())
        if p['case_key'] != key or p['fault_labels_used'] is not False or p['validation_calibration_used'] is not False:
            raise ValueError('invalid prediction identity or validation fitting')
        row = {'case_key': key, 'session_id': case['session_id'], 'topology': case['topology'],
               'evaluable': p['status'] != 'collection_incomplete'}
        if not row['evaluable']:
            row['reason'] = p['reason']
        if session['condition'] == 'healthy':
            if row['evaluable']:
                row.update(alarm_rank=p['alarm_rank'], false_alarm=p['alarm_rank'] is not None,
                           abstention_reasons=p['abstention_reasons'])
            healthy_cases.append(row)
            continue
        row['requested_extra_compute_time_ratio'] = session['requested_extra_compute_time_ratio']
        if row['evaluable']:
            label = private['sessions'][case['session_id']]
            if label['seed'] != session['session_seed'] or label['ratio'] != session['requested_extra_compute_time_ratio']:
                raise ValueError('validation truth/session mismatch')
            target = label['target_rank']
            if type(target) is not int or not 0 <= target < 16:
                raise ValueError('invalid validation target rank')
            lookups.append(case['session_id'])
            alarm, top = p['alarm_rank'], p['rank_order'][0]['rank']
            row.update(target_rank=target, candidate_top1_rank=top,
                       candidate_top1_correct=top == target, alarm_rank=alarm,
                       alarm_correct=alarm == target if alarm is not None else False,
                       alarm_wrong_rank=alarm != target if alarm is not None else False,
                       abstention_reasons=p['abstention_reasons'])
        fault_cases.append(row)
    healthy_complete = [h for h in healthy_cases if h['evaluable']]
    false_alarms = sum(h['false_alarm'] for h in healthy_complete)
    verify(receipt['prediction_files'] + [receipt['model']])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'flash-v4-single-frozen-offline-validation-1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'provenance': [fingerprint(p) for p in (plan_path, receipt_path, private_path, Path(__file__))],
        'prediction_hashes_verified_before_labels_and_after_evaluation': True,
        'validation_target_lookup_session_ids': lookups,
        'validation_refit': False, 'post_validation_parameter_changes': False,
        'fault_overall': aggregate(fault_cases),
        'fault_by_topology': {t: aggregate([c for c in fault_cases if c['topology'] == t])
                              for t in ('legacy', 'node_local_pp')},
        'fault_by_strength': [{'requested_strength': s,
                              **aggregate([c for c in fault_cases if c['requested_extra_compute_time_ratio'] == s])}
                             for s in (0.375, 0.75)],
        'healthy_overall': {'planned_windows': len(healthy_cases),
                            'complete_windows': len(healthy_complete), 'false_alarm_windows': false_alarms,
                            'false_alarm_fraction': false_alarms / len(healthy_complete) if healthy_complete else None,
                            'unit': 'one complete 16-rank eight-iteration run',
                            'continuous_monitoring_FPR_measured': False,
                            'population_FPR_le_5_percent_proven': False},
        'fault_cases': fault_cases, 'healthy_cases': healthy_cases,
        'online_or_formal_blind_acceptance': False, 'blind_schedule_generated': False,
        'online_overhead_measured': False,
    }
    write_new(output / 'validation_evaluation.json', report)
    print(json.dumps({'faults': report['fault_overall'], 'healthy': report['healthy_overall']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    review(args.predictions, args.output)
