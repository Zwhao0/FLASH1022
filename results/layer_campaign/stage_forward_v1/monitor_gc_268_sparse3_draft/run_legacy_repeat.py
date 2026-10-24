#!/usr/bin/env python3
"""Run the one predeclared legacy ABBA repeat; no adaptive retry."""
from __future__ import annotations
import argparse, base64, hashlib, importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
LAYOUT='legacy'
SESSION='v2s_sparse3_gc_monitor268_legacy_repeat_01'
PUBLIC_EVENTS={'detection','profile_schedule','localization','quota_exhausted'}
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def check():
 from flash_v1.startup_gc import configure_plan,validate_plan,POLICY
 from flash_v1.stage_forward_sampling import CONFIG
 plan_path=HERE/'repeat_legacy_plan.json';plan=read(plan_path)
 amend=read(HERE/'repeat_legacy_amendment.json');freeze=read(HERE/'freeze.json')
 supplement=read(HERE/'tmp_supplement/contract.json')
 original=read(HERE/'legacy_plan.json');configured=configure_plan(original)
 expected=dict(configured,session_id=SESSION)
 if plan!=expected: raise ValueError('repeat plan differs from frozen production configuration beyond session_id')
 if (plan.get('session_id')!=SESSION or plan.get('seed')!=1348533921 or plan.get('steps')!=268
     or plan.get('startup_gc_fixed_model') is not True or plan.get('startup_gc_policy')!=POLICY
     or plan.get('stage_forward_aux_sampling')!=CONFIG or validate_plan(plan) is not True):
  raise ValueError('repeat startup-GC/sparse3 configuration invalid')
 if plan.get('cost_harness_files')!=freeze.get('cost_harness_files') or plan.get('parallel_layout')!='legacy':
  raise ValueError('repeat source/layout contract mismatch')
 if (amend.get('repeat_plan_sha256')!=sha(plan_path)
     or amend.get('supplement_contract_sha256')!=sha(HERE/'tmp_supplement/contract.json')
     or supplement.get('repeat_sessions',{}).get('legacy')!=SESSION
     or amend.get('original_session')!='v2s_sparse3_gc_monitor268_legacy_recovery_02'):
  raise ValueError('repeat amendment/aggregate contract mismatch')
 if sha(plan_path)!=amend.get('repeat_plan_sha256') or sha(HERE/'freeze.json')!=amend.get('source_freeze_sha256'):
  raise ValueError('repeat hash binding mismatch')
 S=ROOT/'results/data16_sessions'/SESSION
 if S.exists() or any((HERE/n).exists() for n in ('repeat_legacy_start.json','repeat_legacy_complete.json')):
  raise FileExistsError('repeat session/marker already exists; never retry')
 return plan,freeze

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 plan,freeze=check()
 if not a.execute:
  print(json.dumps({'status':'CPU_repeat_preflight_passed_no_GPU','session_id':SESSION,'plan_sha256':sha(HERE/'repeat_legacy_plan.json')},indent=2));return
 from results.layer_campaign.continuation_v2.gates import owned_runner as owner
 owner.D=HERE;owner.freeze=freeze;owner.rendezvous_port=30520
 owner.check_sources()
 source_check=HERE/'remote_source_check.json'
 # Reinstall only the isolated, frozen cost harness package, never production code.
 backend=(HERE/'isolated_backend.tar').read_bytes();backend_sha=hashlib.sha256(backend).hexdigest()
 encoded=base64.b64encode(backend).decode('ascii')
 extraction="import base64,io,tarfile;tarfile.open(fileobj=io.BytesIO(base64.b64decode(__import__('sys').stdin.read()))).extractall('.')"
 for host,container,sudo in owner.NODES:
  owner.rpc(host,owner.dock(host,container,sudo,['python','-c',extraction]),input=encoded)
 owner.check_sources()
 # Confirm the fresh session does not exist in either owned container.
 code="from pathlib import Path;p=Path('results/data16_sessions/')/"+repr(SESSION)+";print('exists' if p.exists() else 'absent')"
 remote={host:owner.rpc(host,owner.dock(host,container,sudo,['python','-c',code])).strip() for host,container,sudo in owner.NODES}
 if any(v!='absent' for v in remote.values()): raise FileExistsError(f'remote repeat session already exists: {remote}')
 save(HERE/'repeat_legacy_start.json',{'status':'starting_one_predeclared_repeat','session_id':SESSION,
  'plan_sha256':sha(HERE/'repeat_legacy_plan.json'),'freeze_sha256':sha(HERE/'freeze.json'),
  'amendment_sha256':sha(HERE/'repeat_legacy_amendment.json'),'remote_session_absence':remote,
  'isolated_backend_tar_sha256':backend_sha,'no_automatic_retry':True})
 try:
  S=owner.detached_run(plan)
  # owned_runner returns the experiment root with host payloads below host1/host2.
  train_path=S/'host2/train.jsonl';train=[json.loads(x) for x in train_path.read_text().splitlines() if x.strip()]
  if len(train)!=268 or train[-1].get('iteration')!=268: raise ValueError('repeat rank-zero train stream incomplete')
  for host,node,mf in [('host1',1,'node1_manifest.json'),('host2',0,'node0_manifest.json')]:
   manifest=read(S/host/mf);actual_plan=read(S/host/'plan.json')
   if manifest.get('status')!='completed' or manifest.get('session_id')!=SESSION or actual_plan!=plan:
    raise ValueError(f'{host} completion/plan mismatch')
   cfg=manifest.get('configuration',{})
   if cfg.get('startup_gc_fixed_model') is not True or cfg.get('startup_gc_policy')!=plan['startup_gc_policy']:
    raise ValueError(f'{host} startup-GC plan attestation mismatch')
  counts_path=S/'host2/cost_controller_counts.json';counts=read(counts_path)
  events_path=S/'host2/events.jsonl';online_log=S/'host2/online.log'
  if events_path.exists():
   public=[json.loads(x) for x in events_path.read_text().splitlines() if x.strip()]
   public=[x for x in public if x.get('event') in PUBLIC_EVENTS];events_hash=sha(events_path)
   event_source={'path':str(events_path.relative_to(ROOT)),'exists':True,'sha256':events_hash}
  else:
   if (counts.get('natural_candidates')!=0 or counts.get('manual_requests_sent')!=[]
       or counts.get('ignored_child_candidates')!=0 or any(x.get('profile_epoch',0)!=0 for x in train)
       or not online_log.exists() or online_log.stat().st_size!=0):
    raise ValueError('events file absent despite public-event/profile evidence')
   public=[];events_hash=None
   event_source={'path':str(events_path.relative_to(ROOT)),'exists':False,
    'online_log_sha256':sha(online_log),'online_log_bytes':0,
    'absence_crosschecks':{'natural_candidates':0,'manual_requests_sent':[],
     'ignored_child_candidates':0,'profile_epoch_nonzero_steps':0}}
  pred_path=HERE/'repeat_legacy_predictions_without_truth.json'
  snapshot={'protocol':'monitor_gc_268_fixed_repeat_public_v1','layout':'legacy','session_id':SESSION,
   'prediction_before_truth':True,'plan_sha256':sha(HERE/'repeat_legacy_plan.json'),
   'source_freeze_sha256':sha(HERE/'freeze.json'),'events_sha256':events_hash,
   'event_source':event_source,'public_events':public,'train_rows':len(train),
   'train_sha256':sha(train_path),'controller_counts_sha256':sha(counts_path),
   'source_files':{'host1_manifest_sha256':sha(S/'host1/node1_manifest.json'),
    'host2_manifest_sha256':sha(S/'host2/node0_manifest.json')}}
  save(pred_path,snapshot)
  complete={'status':'complete','layout':'legacy','session_id':SESSION,
   'session_path':str(S.relative_to(ROOT)),'plan_sha256':sha(HERE/'repeat_legacy_plan.json'),
   'prediction_sha256':sha(pred_path),'events_sha256':events_hash,'events_file_absent':not events_path.exists(),
   'train_rows':len(train),'source_freeze_sha256':sha(HERE/'freeze.json'),'prediction_before_truth':True}
  save(HERE/'repeat_legacy_complete.json',complete)
  save(HERE/'repeat_legacy_owner_metadata.json',{'actual_gpu_owner':'extended64_protocol','helper_label':'layer_campaign','session_id':SESSION})
  active=HERE/'active.json'
  if active.exists():
   x=read(active);x.update(status='completed',gpu_owner='extended64_protocol',session=SESSION);save(active,x)
  print(json.dumps({'completed':SESSION,'prediction_sha256':sha(pred_path),'train_rows':len(train)},indent=2))
 except BaseException as exc:
  save(HERE/'repeat_legacy_failure.json',{'status':'failed_no_retry','session_id':SESSION,'error':repr(exc),
       'active_session':owner.active,'no_automatic_retry':True})
  raise
if __name__=='__main__':main()
