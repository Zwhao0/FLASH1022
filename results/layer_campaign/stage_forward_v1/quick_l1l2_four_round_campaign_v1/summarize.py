#!/usr/bin/env python3
"""CPU final fixed-24 aggregation only; never reads injection truth or predicts."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def counts(rows):
 assessed=[r for r in rows if r['assessment_status']=='scored']
 return dict(fixed_denominator=len(rows),assessed=len(assessed),unassessed=len(rows)-len(assessed),detected=sum(r.get('detected_in_fault') is True for r in assessed),not_triggered=sum(r.get('public_outcome')=='not_triggered' for r in assessed),public_correct=sum(r.get('public_outcome')=='correct_rank' for r in assessed),public_wrong=sum(r.get('public_outcome')=='wrong_rank' for r in assessed),public_rejected=sum(r.get('public_outcome')=='rejected' for r in assessed),core_correct=sum(r.get('core_outcome')=='correct_rank' for r in assessed),peer_correct=sum(r.get('peer_outcome')=='correct_rank' for r in assessed),injection_valid=sum(r.get('injection_valid') is True for r in assessed),missing_truth=sum(r.get('truth_rows',0)==0 for r in assessed),strict_success=sum(r.get('end_to_end_success') is True for r in assessed),first_actual_accepted_correct=sum(r.get('first_actual_accepted_outcome')=='correct_rank' for r in assessed),first_actual_accepted_wrong=sum(r.get('first_actual_accepted_outcome')=='wrong_rank' for r in assessed))
def main():
 batch=read(HERE/'batch_ready.json');assert batch['fixed_cases']==24 and len(batch['goals'])==24
 rows=[];health=[];sources={};seen_sessions=set();round_status=[]
 for r in batch['rounds']:
  folder=ROOT/r['directory'];brief_path=folder/'score_brief.json';completion=read(folder/'execution_complete.json') if (folder/'execution_complete.json').exists() else {}
  brief=read(brief_path) if brief_path.exists() else None
  if brief:
   assert completion.get('status')=='complete' and set(completion.get('sessions',{}))=={'group1','group2','group3'}
   assert brief['fixed_denominator']==6
   sources[str(brief_path.relative_to(ROOT))]=sha(brief_path)
  scoremap={c['case_id']:c for c in brief.get('cases',[])} if brief else {}
  assert not scoremap or len(scoremap)==6
  missing_status='failed_or_incomplete_round_unassessed' if (folder/'execution_failure.json').exists() else 'round_result_missing_unassessed'
  round_status.append(dict(round=r['round'],status='scored' if brief else missing_status,execution_complete=completion.get('status')=='complete',original_failure_record_preserved=(folder/'execution_failure.json').exists()))
  for goal in [g for g in batch['goals'] if g['round']==r['round']]:
   row=dict(goal,assessment_status='scored' if goal['case_id'] in scoremap else missing_status)
   if goal['case_id'] in scoremap:
    result=scoremap[goal['case_id']];assert result['group']==goal['group'] and result['rank']==goal['rank'] and result['layout']==goal['layout'];row.update(result)
   rows.append(row)
  if brief:
   for group,h in brief.get('health_by_group',{}).items():
    entry=completion['sessions'][group];sid=entry['session_id'];assert sid not in seen_sessions;seen_sessions.add(sid)
    pure=h['pure_healthy_profile_count'];wrong=h['pure_healthy_profile_localized_count']
    health.append(dict(round=r['round'],group=group,session_id=sid,layout=r['groups'][group]['layout'],healthy_valid_monitor_steps=h['healthy_monitor_steps_without_profile'],healthy_nonwarm_nonfault_steps=h['healthy_nonfault_nonwarmup_steps'],healthy_detection_candidates=h['healthy_detection_candidates'],mixed_profile_candidates=h['healthy_detection_profile_overlaps_fault_count'],pure_healthy_profile_count=pure,pure_healthy_wrong_gpu=wrong,pure_healthy_wrong_gpu_ratio=wrong/pure if pure else None))
 assert len(rows)==24 and len({(r['round'],r['group'],r['case_id']) for r in rows})==24
 summaries={}
 for field in ['round','tier','phase','layout']:
  summaries[field]={str(value):counts([r for r in rows if r.get(field)==value]) for value in sorted({r.get(field) for r in rows},key=str)}
 pure=sum(h['pure_healthy_profile_count'] for h in health);wrong=sum(h['pure_healthy_wrong_gpu'] for h in health)
 out=dict(protocol='fixed_four_rounds_12_groups_24_new_case_campaign',overall=counts(rows),by=summaries,round_status=round_status,cases=rows,health_completed_independent_sessions=health,healthy_totals=dict(completed_sessions=len(health),valid_monitor_steps=sum(h['healthy_valid_monitor_steps'] for h in health),healthy_detection_candidates=sum(h['healthy_detection_candidates'] for h in health),mixed_profile_candidates=sum(h['mixed_profile_candidates'] for h in health),pure_healthy_profile_count=pure,pure_healthy_wrong_gpu=wrong,pure_healthy_wrong_gpu_ratio=wrong/pure if pure else None),sources=sources,batch_ready_sha256=sha(HERE/'batch_ready.json'),boundaries=['Fixed24 all goals retained; failed/missing rounds remain unassessed in denominator, never omitted.', 'Prior four-case quick test and prior partial/old core-only data excluded.', 'Healthy denominator includes only independently completed, scored sessions; absent health evidence is unavailable, not invented zero.', 'Startup failures/no-fault attempts are metadata only and never extra cases or healthy denominators.', 'Healthy PROFILE conditional localization ratio is null when denominator0; separate from continuous monitor candidates.', 'Short24-step injection and8-10-step tail are limited checks, not population accuracy.'])
 save(HERE/'score_brief.json',out);print(json.dumps(dict(overall=out['overall'],healthy=out['healthy_totals'])))
if __name__=='__main__':main()
