#!/usr/bin/env python3
"""CPU available-partial replay: freeze ALL public predictions before labels/truth."""
import argparse,hashlib,importlib.util,json,sys,shutil
from pathlib import Path
ROOT=Path('/Users/henry/Desktop/FLASH1022');sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
ATTEMPTS={'original':BASE,'recovery':BASE/'recovery_infra_v1'}
SESSIONS={'original':'v2s_postcommit_random_legacy_d67fffcc','recovery':'v2s_postcommit_random_legacy_d67fffcc_infra_recovery1'}
def read(p):return json.loads(p.read_text())
def jl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
class FakeRedis:
 def __init__(self,records,aux):self.records,self.aux=records,aux
 def lrange(self,key,start,end):
  node=int(key[-1]);rows=self.aux if key.startswith('online_stage_forward_aux') else self.records
  rows=[json.dumps(r).encode() for r in rows if r['rank']//8==node]
  return rows[start:None if end==-1 else end+1]
def predict():
 from flash_v1 import online_v2
 if (OUT/'partial_prediction_freeze.json').exists():raise FileExistsError('preserve predictions')
 protocol=dict(schema='available_partial_v2_review_v1',all_six_planned_cases_retained=True,independent_case_priority=['recovery','original'],old_attempt_only_consistency=True,case_eligibility='full fault32 actually executed and continuous train/monitor coverage through fault end; no PROFILE requirement for entry; no-trigger is miss',partial_fault_or_unstarted='not_evaluated_infra_interrupted',strict_rule='unchanged original scorer; full PROFILE evidence/FMA/aux gates, in-fault detection',all_available_profile_windows_replayed_before_labels=True,health='continuous healthy monitor trigger count/step denominator separate from pure healthy PROFILE wrong-GPU/reject and mixed overlap',scope='no new blind population claim; no retraining, retuning or GPU')
 save(OUT/'available_data_protocol.json',protocol)
 sources={};windows=[]
 for name,d in ATTEMPTS.items():
  session=ROOT/'results/data16_sessions'/SESSIONS[name];h=session/'host2';snap=d/'interrupted_public_snapshot/legacy_partial_public_events.jsonl'
  paths=[snap,h/'configuration.json',h/'topology.json']
  paths+=sorted(p for p in h.glob('online_profile_*/*.json') if p.name in {'evidence.json','evidence_markers.json','stage_forward_aux.json'})
  for p in paths:
   if p.exists():sources[str(p.relative_to(ROOT))]=sha(p)
 for rel,digest in read(BASE/'source_freeze.json')['files'].items():
  assert sha(ROOT/rel)==digest,(rel,'production source mismatch')
  sources[rel]=digest
 save(OUT/'partial_input_freeze.json',dict(sources=sources,protocol_sha256=sha(OUT/'available_data_protocol.json'),script_sha256=sha(Path(__file__)),no_truth_read=True))
 for name,d in ATTEMPTS.items():
  session=ROOT/'results/data16_sessions'/SESSIONS[name];h=session/'host2'
  events=jl(d/'interrupted_public_snapshot/legacy_partial_public_events.jsonl')
  derived=OUT/'derived_public'/name;derived.mkdir(parents=True,exist_ok=True)
  for fn in ['topology.json','configuration.json']:shutil.copyfile(h/fn,derived/fn)
  cfg=read(h/'configuration.json');model=read(ROOT/'results/v2_semantic/linear_candidate_model.json')
  for schedule in [e for e in events if e.get('event')=='profile_schedule']:
   p=h/f"online_profile_{schedule['epoch']}";files=[p/f for f in ['evidence.json','evidence_markers.json','stage_forward_aux.json']]
   entry=dict(attempt=name,session=SESSIONS[name],schedule=schedule,input_complete=all(f.exists() for f in files))
   if entry['input_complete']:
    rows,markers,aux=map(read,files);result=online_v2.diagnose(FakeRedis(rows+markers,aux),derived,schedule,model)
    entry.update(result=result,core=read(derived/f"online_profile_{schedule['epoch']}/original_dc89.json"),rows=len(rows),markers=len(markers),aux_rows=len(aux))
   else:entry.update(result=None,reason='missing_available_profile_input')
   windows.append(entry)
 save(OUT/'partial_predictions_without_truth.json',dict(protocol=protocol,windows=windows,public_events={name:jl(d/'interrupted_public_snapshot/legacy_partial_public_events.jsonl') for name,d in ATTEMPTS.items()}))
 save(OUT/'partial_prediction_freeze.json',dict(status='all_available_partial_replayed_predictions_frozen_before_truth',predictions_sha256=sha(OUT/'partial_predictions_without_truth.json'),inputs_sha256=sha(OUT/'partial_input_freeze.json'),protocol_sha256=sha(OUT/'available_data_protocol.json'),window_count=len(windows),truth_read=False))
 print(json.dumps(dict(status='partial_predictions_frozen_before_truth',windows=len(windows),by_attempt={n:sum(w['attempt']==n for w in windows) for n in ATTEMPTS})))
if __name__=='__main__':predict()
