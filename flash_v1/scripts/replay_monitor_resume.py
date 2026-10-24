"""CPU shadow replay on recorded control trace; no simulated GPU evidence."""
import json
from pathlib import Path
from flash_v1.analysis import WindowDetector

ROOT = Path(__file__).resolve().parents[2]

def replay(rows, events, resume):
    detector = WindowDetector(warmup=12)
    by_epoch = {e['epoch']: e for e in events if e['event'] == 'localization'}
    blocks = []
    for e in events:
        if e['event'] == 'detection':
            end_time = by_epoch[e['epoch']]['wall_time']
            end = max((r['iteration'] for r in rows if r['walltime'] <= end_time), default=e['iteration'])
            blocks.append((e['iteration'], end))
    detections, skipped, backlogged = [], [], []
    for row in rows:
        i = row['iteration']
        busy = next(((a,b) for a,b in blocks if a < i <= b), None)
        if not resume and busy:
            skipped.append(i)
            if i == busy[1]:
                detector = WindowDetector(warmup=i)
            continue
        if row.get('profile_epoch'):
            if resume:
                detector.samples.clear()
            continue
        if resume and busy:
            backlogged.append(i)
        event = detector.add(i, row['wall_iteration_ms'])
        if event:
            # Processing may be delayed until the recorded drain completes.
            event['request_eligible_iteration'] = busy[1] if busy else i
            detections.append(event)
    return dict(detections=detections, dropped_pending_rows=skipped,
                retained_pending_nonprofile_rows=backlogged)

def main():
    report = json.loads((ROOT/'results/v2_semantic/blind64/round1_report.json').read_text())
    output = {'scope':'development shadow replay, fixed recorded PROFILE/control trace; no counterfactual rank accuracy',
              'candidate':'monitor_resume=contiguous; retain last_cp, replay backlog; clear sample deque at PROFILE gap for unit-spaced BEAST', 'layouts':{}}
    for layout in ('legacy','node_local_pp'):
        base=ROOT/f'results/data16_sessions/v2s_scoped32_round1_{layout}_02/host2'
        rows=[json.loads(x) for x in (base/'train.jsonl').read_text().splitlines()]
        events=[json.loads(x) for x in (base/'events.jsonl').read_text().splitlines()]
        cases=[c for c in report['case_records'] if c['layout']==layout]
        original=[e['iteration'] for e in events if e['event']=='detection']
        results={}
        for name,resume in [('legacy_reset',False),('contiguous_resume',True)]:
            result=replay(rows,events,resume)
            detections=result['detections']
            hits=[]
            for c in cases:
                found=[e['request_eligible_iteration'] for e in detections if c['fault_iteration_start'] <= e['request_eligible_iteration'] <= c['fault_iteration_end']]
                hits.append(dict(case_id=c['case_id'], triggers=found,delay=min(found)-c['fault_iteration_start'] if found else None))
            healthy=[e['request_eligible_iteration'] for e in detections if not any(c['fault_iteration_start']<=e['request_eligible_iteration']<=c['fault_iteration_end'] for c in cases)]
            result.update(trigger_iterations=[e['iteration'] for e in detections],cases=hits,
                          fault_cases_triggered=sum(bool(h['triggers']) for h in hits),
                          duplicate_fault_triggers=sum(max(0,len(h['triggers'])-1) for h in hits),
                          healthy_triggers=healthy, original_trigger_iterations=original,
                          baseline_matches_original=[e['iteration'] for e in detections]==original)
            results[name]=result
        output['layouts'][layout]=results
    dest=ROOT/'results/localize_v2_explore/schedule_resume/replay.json'
    dest.write_text(json.dumps(output,indent=2)+'\n')
    for layout, variants in output['layouts'].items():
        for name,r in variants.items():
            print(layout,name, 'faults',r['fault_cases_triggered'],'healthy',r['healthy_triggers'],'duplicate',r['duplicate_fault_triggers'],'triggers',r['trigger_iterations'],'baseline_exact',r['baseline_matches_original'])

if __name__=='__main__': main()
