"""Online peer-priority upper6 regression after blind8; original cases/doses, no timeonly."""
import base64,gzip,hashlib,json,os,shlex,subprocess,sys,time,uuid
from pathlib import Path
from results.layer_campaign.continuation_v2.gates import owned_runner as r
B=Path(__file__).resolve().parent;F=B.parent/'source_freeze.json';r.freeze=r.read(F)
read=r.read;write=r.write;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
resume_l2='--resume-l2' in sys.argv
telemetry=[];owner=uuid.uuid4().hex[:12]
def record(**v):write(B/'pipeline_active.json',v);print(json.dumps(v),flush=True)
def run(path,args=(),log=None):
 with open(log or path.parent/'command.log','a') as out:subprocess.run([sys.executable,str(path),*args],cwd=r.ROOT,env=dict(os.environ,PYTHONPATH=str(r.ROOT)),stdout=out,stderr=subprocess.STDOUT,check=True)
def verify_ready(D):
 j=read(D/'plan_ready.json')
 for name,h in j['artifacts'].items():
  p=F if name=='source_freeze.json' else D/name
  assert sha(p)==h,(name,str(p))
 for p,h in r.freeze['files'].items():assert sha(r.ROOT/p)==h,p
try:
 previous=B.parent/'blind8'
 record(status='waiting_blind8')
 while not (previous/'telemetry_metadata.json').exists():
  if (previous/'execution_failure.json').exists():raise RuntimeError('Blind8 failed; upper regression remains unexecuted')
  time.sleep(15)
 state=read(previous/'active.json')
 assert state.get('status')=='complete' and state.get('strict')==8,state
 record(status='deploying_frozen_guard',source_freeze_sha256=sha(F))
 r.D=B
 try:r.check_sources();already_deployed=True
 except (AssertionError,RuntimeError):already_deployed=False
 payload={p:(r.ROOT/p).read_text() for p in r.freeze['files']}
 compressed=base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
 for h,c,s in r.NODES:
  code='from pathlib import Path;import json,sys,gzip,base64\nfor name,source in json.loads(gzip.decompress(base64.b64decode(sys.stdin.read()))).items():\n p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)'
  if not already_deployed:r.rpc(h,r.dock(h,c,s,['python','-c',code]),input=compressed)
  code='import hashlib,json;from pathlib import Path;print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in '+repr(list(r.freeze['binary_files']))+'}))'
  assert json.loads(r.rpc(h,r.dock(h,c,s,['python','-c',code])))==r.freeze['binary_files']
  script=f'/tmp/flash_guard_reval_{owner}.py';stop=f'/tmp/flash_guard_reval_{owner}_STOP';log=f'/tmp/flash_guard_reval_{owner}.jsonl'
  r.rpc(h,['python3','-c','from pathlib import Path;import sys;Path('+repr(script)+').write_text(sys.stdin.read())'],input=(r.ROOT/'results/layer_campaign/continuation_v1/gates/telemetry.py').read_text())
  pid=int(r.rpc(h,['sh','-c',f'nohup env FLASH_TELEMETRY_STOP={shlex.quote(stop)} python3 -u {shlex.quote(script)} >{shlex.quote(log)} 2>&1 </dev/null & echo $!']).strip());telemetry.append((h,pid,stop,log))
 for index,stage in enumerate(['l1','l2']):
  if resume_l2 and stage=='l1':
   prior=read(B/'l1/summary.json');assert prior['strict_success']==2 and prior['gate_passed']
   assert sum(v['healthy_localized_false_alarms'] for v in prior['by_layout'].values())==0
   continue
  D=B/stage;r.D=D;verify_ready(D);r.check_sources()
  record(stage=stage,status='executing_original_fixed_cases',denominator=[2,4][index])
  for li,layout in enumerate(['legacy','node_local_pp']):
   plan=read(D/f'formal_{layout}_plan.json');assert plan['freeze_manifest_sha256']==sha(F)
   assert plan.get('stage_forward_aux_enabled') is True and plan['stage_forward_aux_model'].endswith('peer_v1/model_scoped.json')
   assert not plan.get('controlled_profile_requests') and not plan.get('integration_node_module')
   r.rendezvous_port=30160+index*4+li
   S=r.detached_run(plan)
   events=[json.loads(x) for x in (S/'host2/events.jsonl').read_text().splitlines() if x]
   write(D/f'{layout}_predictions_without_truth.json',dict(layout=layout,session=S.name,events=[e for e in events if e['event'] in ('detection','profile_schedule','localization')]))
  write(D/'execution_complete.json',dict(status='complete',fixed_denominator=[2,4][index],no_timeonly_repeat=True))
  for h,p,stop,log in telemetry:(D/f'telemetry_{h}.jsonl').write_text(r.rpc(h,['cat',log]))
  run(r.ROOT/'results/layer_campaign/stage_forward_v1/cleanup_peer.py',[str(D),f'v2s_peer_priority_upper6_{stage}_'],D/'cleanup.log');assert read(D/'cleanup.json')['all_clear']
  run(D/'score.py',log=D/'scorer.log');summary=read(D/'summary.json')
  required=[2,4][index];assert summary['strict_success']>=required and summary['gate_passed'],f'{stage} strict gate failed'
  assert sum(v['healthy_localized_false_alarms'] for v in summary['by_layout'].values())==0,f'{stage} wrong healthy GPU'
  record(stage=stage,status='passed' if stage!='l2' else 'complete',strict=summary['strict_success'],public_sources=summary.get('public_localization_sources'),healthy_triggers=summary.get('healthy_false_triggers'))
except BaseException as error:
 record(status='stopped_for_review',error=repr(error),active_session=r.active);raise
finally:
 if r.active:
  for h,c,s in r.NODES:
   try:r.rpc(h,r.dock(h,c,s,['touch',f'results/data16_sessions/{r.active}/STOP']))
   except Exception:pass
 for h,p,stop,log in telemetry:
  try:r.rpc(h,['touch',stop]);time.sleep(1);(B/f'telemetry_{h}.jsonl').write_text(r.rpc(h,['cat',log]))
  except Exception:pass
 write(B/'telemetry_metadata.json',dict(owner=owner,period_s=5,inference_feature=False))
