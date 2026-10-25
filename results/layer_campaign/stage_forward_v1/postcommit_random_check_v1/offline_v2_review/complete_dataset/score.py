"""Labels are opened only after the new complete prediction freeze is verified."""
from replay import *
def run_score():
    freeze=read(HERE/'prediction_freeze.json');assert sha(HERE/'predictions_without_truth.json')==freeze['predictions_sha256']
    d=read(HERE/'predictions_without_truth.json')
    index={(s['session'],w['epoch']):w for s in d['sessions'] if s for w in s['windows'] if not w.get('incomplete_tail')}
    report=dict(status='historical_development_offline_raw_replay',prediction_sha256=freeze['predictions_sha256'],scopes={},healthy={},errors=d['errors'])
    ledger=[]
    scope_sources={}
    cohort_sessions={s['session'] for s in d['sessions'] if s}
    for path in sorted((ROOT/'results/layer_campaign').rglob('cases.json')):
        t=read(path); mappings={}
        for layout in t.get('layouts',{}):
            snapshot=path.with_name(layout+'_predictions_without_truth.json')
            if snapshot.exists():
                session=read(snapshot).get('session')
                if session in cohort_sessions:mappings[layout]=dict(session=session)
        if mappings and set(mappings)==set(t.get('layouts',{})):
            name=str(path.parent.relative_to(ROOT/'results/layer_campaign'));scope_sources[name]=name;d['scopes'][name]=mappings
    for scope,relative in scope_sources.items():
        truth=read(ROOT/'results/layer_campaign'/relative/'cases.json');cases=[];seen=set()
        for layout,group in truth['layouts'].items():
            session=d['scopes'][scope][layout]['session']
            for case in group['cases']:
                identity=(session,case['case_id']);assert identity not in seen;seen.add(identity)
                epoch=case.get('detection_epoch');w=index.get((session,epoch));r=w.get('results',{}).get('fixed3',{}) if w else {}
                public=r.get('public');aux=r.get('aux');target=case['rank']
                result=dict(session=session,case_id=case['case_id'],layout=layout,epoch=epoch,expected_rank=target,replay_coverage=r.get('replay_coverage','no_complete_window'),public=verdict(public,target),aux=verdict(aux,target))
                cases.append(result);ledger.append(dict(scope=scope,**result))
        report['scopes'][scope]=dict(classification='invalid_scope_confounded_historical_raw_only_not_component_validation' if scope=='stage_forward_v1/peer_v1/l3' else 'previously_blind_historical_now_known' if scope.endswith('/blind8') else 'historical_development_or_known_regression',fixed_denominator=truth['fixed_denominator'],unique_case_keys=len(seen),public=count(c['public'] for c in cases),aux=count(c['aux'] for c in cases),cases=cases)
    healthy=[];unclassified=[]
    for session in d['sessions']:
        if not session:continue
        directory=ROOT/'results/data16_sessions'/session['session'];pp=directory/'plan.json';plan=read(pp if pp.exists() else directory/'host2/plan.json')
        for w in session['windows']:
            intervals=[i for i in plan.get('intervals',[]) if i.get('start',0)<=w['start'] and i.get('end',0)>=w['end']]
            healthy_interval=bool(intervals) and all(not i.get('target_ranks') and not i.get('delay_ms') and not i.get('slowdown_extra_ratio') for i in intervals)
            native=any(c.get('start',0)<=w['end'] and c.get('end',0)>=w['start'] and c.get('extra_ms',0)>0 for c in plan.get('native_module_fma_cases',[]))
            if healthy_interval and not native:healthy.append(dict(w,session=session['session']))
            elif not intervals:unclassified.append(dict(session=session['session'],epoch=w['epoch'],start=w['start']))
    for name,windows in [('historical_raw_healthy_tiles',healthy),('fit_calibration_tiles',d['canonical_health'])]:
        complete=[w for w in windows if not w.get('incomplete_tail')];rs=[w.get('results',{}).get('fixed3',{}) for w in complete]
        aux=[r['aux'] for r in rs if r.get('aux')];public=[r['public'] for r in rs if r.get('public')]
        report['healthy'][name]=dict(all_tiles=len(windows),complete_tiles=len(complete),incomplete_tails=len(windows)-len(complete),aux_results=len(aux),aux_structurally_scored=sum('coverage' in r for r in aux),aux_localized=sum(r.get('status')=='localized' for r in aux),aux_rejection_reasons=dict(collections.Counter(r.get('reason') for r in aux)),public_results=len(public),public_unavailable=len(complete)-len(public),public_localized=sum(r.get('status')=='localized' for r in public),public_localized_details=[dict(session=w.get('session'),epoch=w.get('epoch'),start=w['start'],end=w['end'],result=w['results']['fixed3']['public']) for w in complete if w.get('results',{}).get('fixed3',{}).get('public',{} ) is not None and w.get('results',{}).get('fixed3',{}).get('public',{}).get('status')=='localized'])
    all_windows=[w for s in d['sessions'] if s for w in s['windows']];rs=[w['results']['fixed3'] for w in all_windows if 'results' in w]
    report['healthy']['historical_raw_healthy_tiles']['collection_context']='Mixed historical fit/calibration, development, and prior regression/failure-producer collections; not an unconditional healthy rate of the current formal production collector. All 1/26 retained.'
    report['deduplicated_case_keys']=len({(c['session'],c['case_id']) for c in ledger})
    report['scorer_sha256']=sha(__file__)
    report.update(cohort=dict(sessions=len(d['sessions']),raw_tiles=len(all_windows),complete_tiles=len(rs),incomplete_tails=sum(bool(w.get('incomplete_tail')) for w in all_windows),aux_accepted=sum(r['aux'].get('status')=='localized' for r in rs),canonical_tiles=len(d['canonical_health'])),unclassified_windows=unclassified,replay_coverage=freeze['replay_coverage'],limitations=['All predictions frozen before this score; historical labels were previously known during original development. No new blind experiment.', '135 is a historical accepted-window count, not the exhaustive cohort size or independent case denominator.', 'Scope ledgers are separate and can overlap; no aggregate accuracy across scopes.', 'Public production result is scored only with matching raw core plus marker evidence; unavailable is retained as no result.', 'Offline healthy PROFILE localization is distinct from continuous BEAST detection false positives. No BEAST rate inferred.', 'Canonical fit/calibration tiles overlap development sources and are reported separately.'])
    save(HERE/'score_audit.json',report);save(HERE/'case_ledger.json',ledger)
    brief={k:report[k] for k in ('status','prediction_sha256','cohort','deduplicated_case_keys','replay_coverage','limitations')}
    brief['scopes']={k:{key:v[key] for key in ('classification','fixed_denominator','unique_case_keys','public','aux')} for k,v in report['scopes'].items()}
    brief['healthy']={k:dict(v,public_localized_details=[dict({key:detail[key] for key in ('session','epoch','start','end')},core_rank=detail['result'].get('rank'),core_reason=detail['result'].get('reason'),aux_status=detail['result'].get('stage_forward_aux',{}).get('status'),aux_reason=detail['result'].get('stage_forward_aux',{}).get('reason')) for detail in v['public_localized_details']]) for k,v in report['healthy'].items()}
    brief['healthy_coverage_audit']='health_coverage_audit.json'
    brief['healthy_gap_summary']='42=38 complete+4 tails; 38=26 production public+12 lacking both evidence.json and evidence_markers.json (fit 8, calibration 4). 34 structurally scored aux +4 invalid aux; all four invalid are legacy fit tiles. The public healthy rank1 failure is retained, distinct from the previously recorded sparse4 profile221-228 rank4 failure.'
    brief['coverage_boundary']='L1 2/2 and L2 4/4 are small matched replay groups, not repeats of original L1/L2 32-case campaigns.'
    save(HERE/'score_brief.json',brief)
    print(json.dumps(dict(scopes=len(brief['scopes']),case_keys=brief['deduplicated_case_keys'],complete_tiles=brief['cohort']['complete_tiles'],healthy_public_localized=brief['healthy']['historical_raw_healthy_tiles']['public_localized'])))
def verdict(r,target):
    if r is None:return dict(status='unavailable',rank=None,correct=False)
    return dict(status=r.get('status'),rank=r.get('rank'),reason=r.get('reason'),correct=r.get('status')=='localized' and r.get('rank')==target)
def count(rows):
    rows=list(rows);return dict(total=len(rows),correct=sum(r['correct'] for r in rows),wrong=sum(r['status']=='localized' and not r['correct'] for r in rows),rejected=sum(r['status'] not in ('localized','unavailable') for r in rows),unavailable=sum(r['status']=='unavailable' for r in rows))
if __name__=='__main__':run_score()
