"""Post-run integration scoring. Private truth is never passed to inference."""
import argparse
import json
from pathlib import Path

from flash_v4.health_centered_fusion import fingerprint, write_new


def classify(quality, prediction, truth):
    """Keep acquisition, detection, abstention and localization denominators distinct."""
    if not quality.get('nodes_completed') or not quality.get('quality_passed') or quality.get('error'):
        return 'collection_failure'
    if quality.get('manual_integration_only'):
        return 'manual_integration_only'
    fault = truth['target_rank'] is not None
    if not quality.get('profile_triggered'):
        return 'not_triggered' if fault else 'healthy_no_trigger'
    if not prediction:
        return 'inference_failure'
    alarm = prediction.get('alarm_rank')
    if not fault:
        return 'healthy_false_alarm' if alarm is not None else 'healthy_trigger_abstained'
    if quality['window'][0] < truth['window']['first']:
        return 'pre_injection_trigger'
    if quality['window'][0] > truth['window']['last']:
        return 'fault_window_missed'
    if alarm is None:
        return 'abstained'
    return 'correct_alarm' if alarm == truth['target_rank'] else 'mislocalized'


def actuation_quality(root, truth, steps, warmup=3, budget=None):
    files = list(root.glob('node*/actuation/stream_delay_rank_*.jsonl'))
    seen, delayed, errors = set(), {}, 0
    for path in files:
        rank = int(path.stem.rsplit('_', 1)[-1])
        seen.add(rank)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        iterations = [r for r in records if r['kind'] == 'iteration']
        summaries = [r for r in records if r['kind'] == 'summary']
        configurations = [r for r in records if r['kind'] == 'configuration']
        if budget is not None:
            expected_ratio = truth['ratio'] if rank == truth['target_rank'] else 0
            expected_config = dict(ratio=expected_ratio,
                max_fault_iterations=budget.get('max_fault_iterations', 32),
                max_step_us=budget.get('max_step_us', 200000), max_fault_seconds=30)
            if (len(configurations) != 1 or any(configurations[0].get(k) != v
                    for k, v in expected_config.items())):
                errors += 1
        # Both frozen v1 trainers execute their default three warmup schedules.
        if (len(summaries) != 1 or len(iterations) != steps + warmup
                or {r['iteration'] for r in iterations} != set(range(1, steps + warmup + 1))):
            errors += 1
        for r in iterations:
            errors += sum(r.get(k, 0) for k in ('delay_errors', 'event_error', 'cap_hits', 'time_budget_hits'))
            if r.get('requested_delay_us', 0) > 0:
                delayed.setdefault(rank, []).append(r['iteration'])
        for r in summaries:
            errors += sum(r.get(k, 0) for k in ('unclosed_steps', 'log_errors', 'calibration_failed'))
    target = truth['target_rank']
    expected = ({} if target is None else
                {target: list(range(truth['window']['first'], truth['window']['last'] + 1))})
    return dict(passed=len(files) == 16 and seen == set(range(16)) and delayed == expected and errors == 0,
                delayed_iterations_by_rank=delayed, errors=errors, strength_verified=False)


def evaluate(plan_path):
    plan = json.loads(plan_path.read_text())
    root = Path('results/v4/online') / plan['runtime_release']
    # Only score after every finite driver has a terminal result. Never dispatch here.
    qualities = {row['case']: json.loads((root / ('driver_' + row['case']) / 'result.json').read_text())
                 for row in plan['development_GPU_sessions']}
    private_path = Path(plan['private_execution'])
    if fingerprint(private_path)['sha256'] != plan['private_execution_receipt']['sha256']:
        raise ValueError('private execution plan changed after freeze')
    truth = json.loads(private_path.read_text())['cases']
    rows = []
    for row in plan['development_GPU_sessions']:
        case = row['case']
        prediction_path = root / case / 'live/frozen_prediction.json'
        prediction = json.loads(prediction_path.read_text()) if prediction_path.exists() else None
        quality = qualities[case]
        requests = [json.loads(p.read_text()) for p in
                    (root / case / 'final').glob('node*/control/profile_request.json')]
        request_iteration = requests[0]['iteration'] if len(requests) == 1 else None
        outcome = classify(quality, prediction, truth[case])
        actuation = actuation_quality(root / case / 'final', truth[case], row['steps'],
                                      budget=plan.get('actor_budget', {}))
        if not actuation['passed'] and outcome != 'collection_failure':
            outcome = 'injection_failure'
        if (truth[case]['target_rank'] is not None and request_iteration is not None
                and request_iteration < truth[case]['window']['first']
                and outcome not in ('collection_failure', 'inference_failure', 'injection_failure')):
            outcome = 'pre_injection_trigger'
        rows.append(dict(case=case, topology=row['topology'], outcome=outcome,
                         target_rank=truth[case]['target_rank'],
                         actuation_quality=actuation,
                         alarm_rank=prediction.get('alarm_rank') if prediction else None,
                         profile_window=quality.get('window'), request_iteration=request_iteration,
                         request_delay_iterations=(request_iteration - truth[case]['window']['first']
                             if request_iteration is not None and truth[case]['target_rank'] is not None else None),
                         prediction_during_training=quality.get('prediction_during_training', False)))
    counts = {outcome: sum(r['outcome'] == outcome for r in rows)
              for outcome in sorted({r['outcome'] for r in rows})}
    result = dict(scope=plan['scope'], cases=rows, counts=counts, total_sessions=len(rows),
                  fault_sessions=sum(t['target_rank'] is not None for t in truth.values()),
                  healthy_sessions=sum(t['target_rank'] is None for t in truth.values()),
                  formal_acceptance_claimed=False, plan=fingerprint(plan_path),
                  scorer=fingerprint(Path(__file__)))
    write_new(root / 'development_evaluation_private.json', result)
    print(json.dumps({'total_sessions': len(rows), 'counts': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    evaluate(parser.parse_args().plan)
