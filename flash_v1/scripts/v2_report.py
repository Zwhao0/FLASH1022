"""CPU-only v2 session integrity and truth-free evidence packaging."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def inspect_session(path):
    path = Path(path)
    plan = json.loads((path / 'plan.json').read_text())
    checks, manifests, evidence = {}, [], []
    hashes = {}
    for node, host in enumerate(('host2', 'host1')):
        directory = path / host
        manifest = json.loads((directory / f'node{node}_manifest.json').read_text())
        manifests.append(manifest)
        checks[f'{host}:completed'] = manifest.get('status') == 'completed'
        for f in sorted(directory.glob('evidence_rank*.jsonl')):
            rows = jsonl(f)
            evidence.extend(rows)
            hashes[str(f.relative_to(path))] = hashlib.sha256(f.read_bytes()).hexdigest()
        drain = directory / 'drain_status.json'
        if drain.exists():
            value = json.loads(drain.read_text())
            checks[f'{host}:zero_ring_drops'] = value.get('status') == 'complete' and value.get('dropped') == []
    for key in ('configuration', 'source_sha256', 'model_sha256', 'probe_sha256', 'snapshot_sha256', 'evidence_model_sha256'):
        checks[f'aligned:{key}'] = manifests[0].get(key) == manifests[1].get(key)
    if plan.get('online'):
        for host in ('host2','host1'):
            checks[f'{host}:sampling_disabled_at_finish']=int.from_bytes((path/host/'state.bin').read_bytes(),'little')==0
    train = jsonl(path / 'host2/train.jsonl')
    checks['complete_iteration_sequence'] = [x['iteration'] for x in train] == list(range(1, plan['intervals'][-1]['end'] + 1))
    markers = [r for r in evidence if r.get('record_type') == 'iteration_complete']
    evidence = [r for r in evidence if r.get('record_type') != 'iteration_complete']
    counts = Counter((r['rank'], r['iteration'], r['module_kind'], r['phase']) for r in evidence)
    checks['unique_rank_iteration_kind_phase'] = all(n == 1 for n in counts.values())
    checks['all_16_ranks_observed'] = set(r['rank'] for r in evidence) == set(range(16)) if plan.get('evidence_enabled') else not evidence
    if plan.get('profile_backend') == 'semantic_only':
        checks['native_coverage_not_applicable'] = all(
            m.get('native_profile_coverage') == 'not_applicable_semantic_only' and
            not m.get('probe_enabled') for m in manifests)
        from flash_v1.evidence import interval_residual
        union_ok = True
        for row in evidence:
            try:
                union_ok &= row.get('coverage_complete') is True and math.isfinite(row['duration_us'])
                union_ok &= row['duration_us'] >= 0 and math.isfinite(row['module_elapsed_us'])
                union_ok &= all(math.isfinite(v[field]) for v in row['communication_intervals']
                                for field in ('start_us','end_us'))
                residual = interval_residual(row['module_elapsed_us'], [
                    (v['start_us'], v['end_us']) for v in row['communication_intervals']])
                union_ok &= abs(residual-row['duration_us']) < .01
                union_ok &= len(row['communication_intervals']) == (
                    1 if row['phase']=='forward' else 3 if row['module_kind']=='attention' else 2)
            except (ValueError, KeyError, TypeError):
                union_ok = False
        checks['complete_communication_union'] = union_ok
        observed = Counter((r['rank'],r['iteration']) for r in evidence)
        roles = {}
        for row in evidence:
            roles.setdefault((row['rank'],row['iteration']),set()).add((row['module_kind'],row['phase']))
        expected_roles = {(kind,phase) for kind in ('attention','experts') for phase in ('forward','backward')}
        marker_counts = Counter((r['rank'],r['iteration']) for r in markers)
        checks['complete_four_phase_markers'] = (set(observed)==set(marker_counts) and
            all(n==4 for n in observed.values()) and all(n==1 for n in marker_counts.values()) and
            all(value==expected_roles for value in roles.values()) and
            all(r.get('rows_count')==4 and r.get('expected_rows')==4 for r in markers))
        if not plan.get('online') and plan.get('profile_enabled', True):
            required={(rank,step) for rank in range(16) for step in range(1,plan['intervals'][-1]['end']+1)}
            checks['all_offline_profile_rank_iterations'] = set(observed)==required
    report = dict(session=plan['session_id'], integrity_passed=all(checks.values()), checks=checks,
                  failed_checks=[k for k,v in checks.items() if not v],
                  completion_markers=len(markers), evidence_rows=len(evidence), valid_rows=sum(r.get('coverage_complete') is True and r.get('duration_us') is not None for r in evidence),
                  slot_phase_rows={f'{k}:{p}':v for (k,p),v in Counter((r['module_kind'],r['phase']) for r in evidence).items()},
                  evidence_sha256=hashes, planned_steps=plan['intervals'][-1]['end'])
    if plan.get('profile_backend') == 'semantic_only':
        report['native_profile_coverage'] = 'not_applicable_semantic_only'
    destination = path / 'v2_integrity.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    return report


def export_window(path, start, end, destination):
    """Output only measurement schema; plan/truth is never passed to inference."""
    path, destination = Path(path), Path(destination)
    allowed = {'schema_version', 'rank', 'iteration', 'epoch', 'layer', 'phase', 'slot',
               'module_kind', 'module_name', 'occurrence', 'shape', 'dtype', 'measurement',
               'stream_id', 'coverage_complete', 'cpu_enqueue_us', 'module_elapsed_us',
               'duration_us', 'communication_intervals', 'event_pairs', 'invalid_reason'}
    rows = [r for host in ('host2','host1') for f in sorted((path/host).glob('evidence_rank*.jsonl'))
            for r in jsonl(f) if start <= r['iteration'] <= end and r.get('record_type') != 'iteration_complete']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with destination.open('w') as stream:
        for r in rows:
            stream.write(json.dumps({k:v for k,v in r.items() if k in allowed}) + '\n')
    return len(rows)


def score_online(path):
    """Separate post-run scorer: fault truth is read here, never by infer()."""
    path = Path(path)
    integrity = inspect_session(path)
    p = json.loads((path/'plan.json').read_text())
    events = jsonl(path/'host2/events.jsonl')
    train = jsonl(path/'host2/train.jsonl')
    detections = [r for r in events if r['event']=='detection']
    localizations = {r['epoch']:r for r in events if r['event']=='localization'}
    schedules = {r['epoch']:r for r in events if r['event']=='profile_schedule'}
    faults = [r for r in p['intervals'] if r.get('target_ranks')]
    cases = []
    for fault in faults:
        matches = [r for r in detections if fault['start']<=r['iteration']<=fault['end']]
        decisions = [localizations[r['epoch']]['result'] for r in matches if r['epoch'] in localizations]
        expected = f"rank:{fault['target_ranks'][0]}"
        predictions = [r.get('predicted','inconclusive') for r in decisions]
        cases.append(dict(case_id=fault['id'],mode=fault['fault_mode'],target_rank=fault['target_ranks'][0],
            detected=bool(matches),correct=expected in predictions and all(x==expected for x in predictions),
            predictions=predictions,detection_delay_iterations=min((r['iteration']-fault['start'] for r in matches),default=None),
            detected_without_diagnosis=bool(matches) and not decisions))
    cohort=[]
    all_evidence=[r for h in ('host2','host1') for f in (path/h).glob('evidence_rank*.jsonl') for r in jsonl(f)]
    excluded={'target_ranks','target_rank','fault_mode','delay_ms','slowdown_extra_ratio'}
    no_truth=not any(set(r)&excluded for r in all_evidence)
    scheduled={(epoch,step) for epoch,s in schedules.items() for step in range(s['start'],s['end']+1)}
    scoped=all((r['epoch'],r['iteration']) in scheduled for r in all_evidence)
    observed_steps={(r['epoch'],r['iteration']) for r in all_evidence}
    training_profile_steps={(int(r['profile_epoch']),r['iteration']) for r in train if r.get('profile_epoch',0)>0}
    scoped &= observed_steps==training_profile_steps==scheduled
    for epoch, schedule in schedules.items():
        selected=[r for r in all_evidence if r['epoch']==epoch and schedule['start']<=r['iteration']<=schedule['end']]
        markers=[r for r in selected if r.get('record_type')=='iteration_complete']
        measurements=[r for r in selected if r.get('record_type')=='measurement']
        identities=Counter((r['rank'],r['iteration']) for r in markers)
        expected={(rank,step) for rank in range(16) for step in range(schedule['start'],schedule['end']+1)}
        complete=(schedule['end']-schedule['start']==7 and set(identities)==expected and all(n==1 for n in identities.values()))
        cohort.append(dict(epoch=epoch,start=schedule['start'],end=schedule['end'],complete_8x16_markers=complete,
            measurements=len(measurements),valid_measurements=sum(r['coverage_complete'] for r in measurements),
            localized_before_training_end=epoch in localizations and localizations[epoch]['wall_time']<train[-1]['walltime']))
    report=dict(session=p['session_id'],integrity_passed=integrity['integrity_passed'] and no_truth and scoped and all(x['complete_8x16_markers'] for x in cohort),
                no_truth_in_inference_records=no_truth,cases=cases,correct=sum(r['correct'] for r in cases),
                denominator=len(faults),detected=sum(r['detected'] for r in cases),cohorts=cohort,
                healthy_detection_count=sum(not any(f['start']<=r['iteration']<=f['end'] for f in faults) for r in detections),
                healthy_iterations=sum(r['iteration']>12 and not any(f['start']<=r['iteration']<=f['end'] for f in faults) for r in train))
    report['only_scheduled_profile_evidence'] = scoped
    (path/'v2_online_score.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    args=parser.parse_args()
    print(json.dumps(inspect_session(args.session),indent=2))


if __name__=='__main__':
    main()
