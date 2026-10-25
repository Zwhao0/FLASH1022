#!/usr/bin/env python3
"""Post-hoc existing online-log audit; no new online execution."""
from partial_replay import *
HERE=None
def enrich(layouts, data, scorer):
    import statistics
    frozen_cases = {r["fault_interval"]["case_id"]: r for r in read(HERE / "contract.json")["cases"]}
    for layout, result in layouts.items():
        plan, session, events = (data[layout][k] for k in ("plan", "session", "events"))
        train = scorer.read_jsonl(session / "host2/train.jsonl")
        by_iter = {int(r["iteration"]): r for r in train}
        faults = [r for r in plan["intervals"] if r.get("target_ranks")]
        fault_steps = {i for f in faults for i in range(f["start"], f["end"] + 1)}
        schedules = {r["epoch"]: r for r in events if r.get("event") == "profile_schedule"}
        healthy = result["healthy_trigger_verdicts"]
        for row in healthy:
            profile = schedules.get(row["epoch"])
            row["profile_start"] = profile["start"] if profile else None
            row["profile_end"] = profile["end"] if profile else None
            row["profile_overlaps_fault"] = bool(profile and any(i in fault_steps for i in range(profile["start"], profile["end"] + 1)))
            row["overlapping_faults"] = [{"case_id": f["case_id"], "rank": f["target_ranks"][0], "start": f["start"], "end": f["end"]} for f in faults if profile and max(f["start"], profile["start"]) <= min(f["end"], profile["end"])]
            row["public_rank_matches_overlapping_fault"] = any(row["rank"] == f["rank"] for f in row["overlapping_faults"])
            row["profile_fully_in_fault"] = bool(profile and any(f["start"] <= profile["start"] and profile["end"] <= f["end"] for f in faults))
            row["pure_healthy_profile"] = bool(profile and profile["start"] > 12 and not row["profile_overlaps_fault"])
        result["healthy_detection_candidates"] = len(healthy)
        result["healthy_detection_profile_overlaps_fault_count"] = sum(r["profile_overlaps_fault"] for r in healthy)
        result["pure_healthy_profile_count"] = sum(r["pure_healthy_profile"] for r in healthy)
        result["pure_healthy_profile_localized_count"] = sum(r["pure_healthy_profile"] and r["verdict"] == "localized" for r in healthy)
        result["healthy_monitor_steps_without_profile"] = sum(i > 12 and i not in fault_steps and not int(r.get("profile_epoch", 0) or 0) for i, r in by_iter.items())
        result["detection_source_counts"] = dict(scorer.Counter(r.get("source", "unspecified") for r in events if r.get("event") == "detection"))
        for case in result["cases"]:
            frozen = frozen_cases[case["case_id"]]
            case.update(tier=frozen["tier"], drawn_family=frozen.get("family"), frozen_dose=frozen["dose"], frozen_slowdown_extra_ratio=frozen["fault_interval"]["slowdown_extra_ratio"])
            start, end = case["fault_start"], case["fault_end"]
            before = next((p for p in reversed(plan["intervals"]) if p.get("segment") == "healthy" and p["end"] < start), None)
            after = next((p for p in plan["intervals"] if p.get("segment") == "healthy" and p["start"] > end), None)
            def valid_iters(a, b):
                return [i for i in range(max(13, a), b + 1) if i in by_iter and not int(by_iter[i].get("profile_epoch", 0) or 0)]
            hi_before = valid_iters(before["start"], before["end"])[-16:] if before else []
            hi_after = valid_iters(after["start"], after["end"])[:16] if after else []
            fi = valid_iters(start, end)
            ht = [float(by_iter[i]["wall_iteration_ms"]) for i in hi_before + hi_after]
            ft = [float(by_iter[i]["wall_iteration_ms"]) for i in fi]
            hm, fm = (statistics.median(x) if x else None for x in (ht, ft))
            case.update(healthy_reference_before_steps=hi_before, healthy_reference_after_steps=hi_after, healthy_reference_nonprofile_steps=len(ht), fault_nonprofile_steps=len(ft), healthy_nonprofile_median_ms=hm, fault_nonprofile_median_ms=fm, observed_nonprofile_wall_degradation_pct=100 * (fm / hm - 1) if hm and fm else None)
            case["degradation_interpretation"] = "host2 rank0 merged train wall_iteration_ms; median of all valid fault non-PROFILE steps vs pooled nearest up-to-16 valid non-PROFILE steps in each adjacent healthy interval, excluding warmup; fixed rule before truth, no window selection; not PROFILE overhead or causal isolated compute cost"
            case["extra_in_fault_detections"] = max(0, case["detection_count_in_fault"] - 1)
            case["in_fault_detection_source_counts"] = dict(scorer.Counter(r.get("source", "unspecified") for r in events if r.get("event") == "detection" and start <= r.get("iteration", -1) <= end))

def main():
 global HERE
 freeze=read(OUT/'partial_prediction_freeze.json');assert freeze['predictions_sha256']==sha(OUT/'partial_predictions_without_truth.json')
 for path,digest in read(OUT/'partial_input_freeze.json')['sources'].items():assert sha(ROOT/path)==digest,path
 pred=read(OUT/'partial_predictions_without_truth.json')
 scorer=load(ROOT/'results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_reverted_baseline/score.py','partial_existing_scorer')
 candidate=load(ROOT/'results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_sparse3_gc_lifetime_v2/score_candidate.py','partial_sparse3_scorer')
 candidate.install_sparse_auxiliary_check(scorer);candidate.install_sparse_layout_scorer(scorer)
 audits={};labels={};views={}
 # Labels/truth are first accessed only after verifying every prediction/input freeze.
 for name,d in ATTEMPTS.items():
  contract=read(d/'contract.json');plan_info=contract['plans']['legacy'];plan=read(ROOT/plan_info['path']);actual=ROOT/'results/data16_sessions'/SESSIONS[name]
  view=OUT/'score_inputs'/name;h=view/'host2';h.mkdir(parents=True,exist_ok=True);views[name]=view
  for host in ['host1','host2']:
   target=view/host;target.mkdir(exist_ok=True)
   for p in (actual/host).glob('whole_compute_truth_rank*.jsonl'):
    labels[str(p.relative_to(ROOT))]=sha(p);link=target/p.name
    if not link.exists():link.symlink_to(p)
  for fn in ['train.jsonl','topology.json']:
   p=actual/'host2'/fn;link=h/fn
   if not link.exists():link.symlink_to(p)
  records=[];aux=[]
  for w in pred['windows']:
   if w['attempt']!=name or not w['input_complete']:continue
   original=actual/'host2'/f"online_profile_{w['schedule']['epoch']}"
   records+=read(original/'evidence.json')+read(original/'evidence_markers.json');aux+=read(original/'stage_forward_aux.json')
   link=h/original.name
   if not link.exists():link.symlink_to(OUT/'derived_public'/name/original.name)
  (h/'evidence_rank_all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
  (h/'stage_forward_aux_rank_all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in aux))
  design=[]
  for c in contract['cases']:
   if c['layout']!='legacy':continue
   f=c['fault_interval'];design.append(dict(case_id=f['case_id'],parent_case_id=f['case_id'],regression_family={'L1':'whole_compute','L2':'backward_stage' if c.get('phase')=='bwd' else 'forward_stage','L3':'component','L4':'operator'}[c['tier']],rank=c['rank'],phase=c.get('phase'),fault_target=c['fault_target'],operator_family=c.get('family')))
  data=dict(session=view,plan=plan,events=pred['public_events'][name],prediction_sha256=freeze['predictions_sha256'],plan_sha256=plan_info['sha256'])
  # Existing online public outcomes stay unchanged; separately attach fresh replay predictions.
  result=scorer.score_layout('legacy',data,dict(design=dict(layouts=dict(legacy=design))))
  HERE=d;enrich(dict(legacy=result),dict(legacy=data),scorer)
  train=jl(actual/'host2/train.jsonl');iters={r['iteration'] for r in train};last=max(iters,default=0)
  for row in result['cases']:
   full=all(i in iters for i in range(row['fault_start'],row['fault_end']+1))
   row['full_fault_executed_trace']=full;row['observed_last_iteration']=last
   row['raw_scorer_injection_valid']=row['injection_valid'];row['fma_validation_status']='verified_valid' if row['injection_valid'] else 'unknown_missing_truth' if row['truth_rows']==0 else 'recorded_invalid';row['assessment_status']='trace_complete_provisional_planned_target_only' if full and row['truth_rows']==0 else 'evaluated_existing_online_log' if full else 'not_evaluated_infra_truncated' if row['fault_start']<=last else 'not_evaluated_not_reached'
   w=next((w for w in pred['windows'] if w['attempt']==name and w['schedule']['epoch']==row['detection_epoch']),None)
   row['offline_replay_public']=None if not w else {k:w['result'].get(k) for k in ['status','rank','reason','source']}
   row['offline_replay_matches_logged_public']=None if not w else (row['public_rank']==w['result'].get('rank') and (row['public_outcome']=='correct_rank')==(w['result'].get('status')=='localized' and w['result'].get('rank')==row['rank']))
  audits[name]=result
 save(OUT/'partial_label_sources_after_prediction_freeze.json',dict(sources=labels,prediction_freeze_sha256=sha(OUT/'partial_prediction_freeze.json')))
 # Predeclared latest-attempt priority. No falling back to an older success.
 planned=read(BASE/'contract.json')['cases'];ledger=[]
 for c in planned:
  cid=c['fault_interval']['case_id'];row=next((r for r in audits['recovery']['cases'] if r['case_id']==cid),None)
  if row is None:
   ledger.append(dict(case_id=cid,layout=c['layout'],tier=c['tier'],rank=c['rank'],fault_start=c['fault_interval']['start'],fault_end=c['fault_interval']['end'],frozen_dose=c['dose'],assessment_status='not_evaluated_layout_never_started',independent_count=False))
  else:
   row=dict(row,chosen_attempt='recovery',independent_count=row['full_fault_executed_trace']);ledger.append(row)
 evaluated=[r for r in ledger if r['independent_count']]
 brief=dict(schema='available_partial_existing_online_log_review_v1',planned_cases=6,evaluated_independent_cases=len(evaluated),unassessed_cases=6-len(evaluated),detected_in_fault=sum(r['detected_in_fault'] for r in evaluated),public_correct=sum(r['public_outcome']=='correct_rank' for r in evaluated),core_correct=sum(r['core_outcome']=='correct_rank' for r in evaluated),strict_success=sum(r['end_to_end_success'] for r in evaluated),injection_valid=sum(r['injection_valid'] for r in evaluated),cases=ledger,health={name:{k:r[k] for k in ['healthy_false_triggers','healthy_wrong_gpu_localizations','healthy_nonfault_nonwarmup_steps','healthy_monitor_steps_without_profile','healthy_detection_candidates','healthy_detection_profile_overlaps_fault_count','pure_healthy_profile_count','pure_healthy_profile_localized_count','healthy_trigger_verdicts']} for name,r in audits.items()},health_independent_attempt='recovery; original is consistency only',interpretation='Existing real online logs post-hoc audit; offline replay is conditional on collected PROFILE; no new blind/full N6 claim')
 brief.update(trace_complete_provisional_cases=len(evaluated),physically_verified_cases=sum(r['injection_valid'] for r in evaluated),formal_accuracy_denominator=sum(r['injection_valid'] for r in evaluated),strict_success_assessment='unassessable_missing_truth; raw existing scorer false retained, not evidence of algorithm failure',raw_existing_scorer_strict_success=brief['strict_success'],planned_target_hits_provisional=brief['public_correct'],planned_target_hit_denominator_provisional=len(evaluated))
 save(OUT/'partial_full_audit.json',audits)
 for row in brief['cases']:
  if 'semantic_evidence_integrity' in row:row['semantic_evidence_passed']=(row.pop('semantic_evidence_integrity') or {}).get('passed')
  row.pop('auxiliary_audit',None)
 save(OUT/'partial_score_brief.json',brief)
 print(json.dumps({k:brief[k] for k in ['planned_cases','evaluated_independent_cases','unassessed_cases','detected_in_fault','public_correct','core_correct','strict_success','injection_valid']}))
if __name__=='__main__':main()
