#!/usr/bin/env python3
"""Fixed complete L1/L2 original cohorts, CPU latest-v2 core fallback replay."""
import hashlib,json,sys,shutil,importlib.util
from pathlib import Path
ROOT=Path('/Users/henry/Desktop/FLASH1022');sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
SCOPE=ROOT/'results/layer_campaign'
def read(p):return json.loads(p.read_text())
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
class Redis:
 def __init__(self,rows):self.rows=rows
 def lrange(self,key,start,end):return [json.dumps(r).encode() for r in self.rows if r['rank']//8==int(key[-1])][start:None if end==-1 else end+1]
def predict():
 from flash_v1.online_v2 import diagnose
 if (HERE/'prediction_freeze.json').exists():raise FileExistsError('preserve frozen replay')
 prior=read(HERE.parent/'complete_dataset/predictions_without_truth.json');prior_sessions={s['session'] for s in prior['sessions'] if s}
 inputs={};inventory=[]
 for tier in ['l1','l2']:
  label=SCOPE/tier/'cases.json';inputs[str(label.relative_to(ROOT))]=sha(label) # Hash only; no labels opened.
  for layout in ['legacy','node_local_pp']:
   binding=SCOPE/tier/f'{layout}_predictions_without_truth.json';session=read(binding)['session'];assert session not in prior_sessions
   host=ROOT/'results/data16_sessions'/session/'host2';cfg=read(host/'configuration.json');assert not cfg.get('stage_forward_aux_enabled',False)
   profiles=sorted(host.glob('online_profile_*'),key=lambda p:int(p.name.rsplit('_',1)[1]))
   sources=[binding,host/'configuration.json',host/'topology.json',host/'events.jsonl',host/'train.jsonl']+[p/f for p in profiles for f in ['evidence.json','evidence_markers.json']]
   for p in sources:inputs[str(p.relative_to(ROOT))]=sha(p)
   inventory.append(dict(tier=tier,layout=layout,session=session,fixed_cases=16,profiles=len(profiles),group='latest_v2_core_only_legacy_aux_disabled',new_session_vs_prior44=True))
 for rel,digest in read(HERE.parent.parent/'source_freeze.json')['files'].items():assert sha(ROOT/rel)==digest;inputs[rel]=digest
 save(HERE/'inventory_freeze.json',dict(fixed_cases=64,cohorts=inventory,input_sha256=inputs,scorer_plan='all 64 complete planned cases retained, no-trigger is miss; no selection by PROFILE/acceptance; 68 existing PROFILE incl health',label_files_hashed_not_read=True,source_sha256=sha(Path(__file__)),new_full_fusion_cases=0,aux='historically disabled; production original-config core fallback, no synthesized aux',history='known original L1/L2 cohorts; no new blind data; exclude all prior44/upper6 sessions'))
 model=read(ROOT/'results/v2_semantic/linear_candidate_model.json');pred=[]
 for cohort in inventory:
  h=ROOT/'results/data16_sessions'/cohort['session']/'host2';derived=HERE/'production_replay'/cohort['session'];derived.mkdir(parents=True,exist_ok=True)
  for fn in ['topology.json','configuration.json']:shutil.copyfile(h/fn,derived/fn)
  events=[r for r in jl(h/'events.jsonl') if r.get('event') in {'detection','profile_schedule','localization','quota_exhausted'}];windows=[]
  for schedule in [r for r in events if r.get('event')=='profile_schedule']:
   p=h/f"online_profile_{schedule['epoch']}"
   if not (p/'evidence.json').exists() or not (p/'evidence_markers.json').exists():
    windows.append(dict(schedule=schedule,result=None,reason='missing_profile_evidence_or_markers',rows=0,markers=0));continue
   rows=read(p/'evidence.json');markers=read(p/'evidence_markers.json')
   result=diagnose(Redis(rows+markers),derived,schedule,model)
   windows.append(dict(schedule=schedule,result=result,rows=len(rows),markers=len(markers)))
  pred.append(dict(cohort,windows=windows,events=events))
 save(HERE/'predictions_without_truth.json',dict(cohorts=pred))
 save(HERE/'prediction_freeze.json',dict(status='all_L1_L2_original_cohort_predictions_frozen_before_label_score',predictions_sha256=sha(HERE/'predictions_without_truth.json'),inventory_sha256=sha(HERE/'inventory_freeze.json'),windows=sum(len(c['windows']) for c in pred),fixed_cases=64,new_full_fusion_cases=0,truth_read=False))
 print(json.dumps(dict(status='frozen',cases=64,windows=sum(len(c['windows']) for c in pred))))
if __name__=='__main__':predict()
