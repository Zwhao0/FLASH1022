"""Sequential fixed stage, detached training and native 5s telemetry; no blind early scoring."""
import hashlib,json,shlex,subprocess as sp,sys,time,uuid
from pathlib import Path
from results.layer_campaign.continuation_v2.gates import owned_runner as r
B=Path(__file__).resolve().parent;stage=sys.argv[1];assert stage in ('l1','l2');D=B/stage;r.D=D;F=B.parent/'source_freeze.json';r.freeze=r.read(F)
telemetry=[];owner=uuid.uuid4().hex[:12]
def ready(name):
 j=r.read(D/name)
 for f,h in j['artifacts'].items():
  p=F if f=='source_freeze.json' else D/f
  assert hashlib.sha256(p.read_bytes()).hexdigest()==h,f
 assert hashlib.sha256(F.read_bytes()).hexdigest()==j['source_freeze_sha256']
 for f,h in r.freeze['files'].items():assert hashlib.sha256((r.ROOT/f).read_bytes()).hexdigest()==h,f
try:
 ready('timeonly_ready.json');r.check_sources()
 for host,container,sudo in r.NODES:
  script=f'/tmp/flash_peerupper_{stage}_{owner}.py';stop=f'/tmp/flash_peerupper_{stage}_{owner}_STOP';log=f'/tmp/flash_peerupper_{stage}_{owner}.jsonl'
  r.rpc(host,['python3','-c','from pathlib import Path;import sys;Path('+repr(script)+').write_text(sys.stdin.read())'],input=(r.ROOT/'results/layer_campaign/continuation_v1/gates/telemetry.py').read_text())
  pid=int(r.rpc(host,['sh','-c',f'nohup env FLASH_TELEMETRY_STOP={shlex.quote(stop)} python3 -u {shlex.quote(script)} >{shlex.quote(log)} 2>&1 </dev/null & echo $!']).strip());telemetry.append((host,pid,stop,log))
 time.sleep(10)
 resources={h:[json.loads(x) for x in r.rpc(h,['cat',l]).splitlines() if x] for h,p,s,l in telemetry};r.write(D/'initial_resources.json',resources)
 for kind in ('timeonly','formal'):
  if kind=='formal':
   for host,pid,stop,log in telemetry:(D/f'telemetry_{host}.jsonl').write_text(r.rpc(host,['cat',log]))
   sp.run([sys.executable,str(B/'timeonly_gate.py'),stage],check=True,cwd=r.ROOT)
   if not r.read(D/'timeonly_gate.json')['stable_for_formal']:raise RuntimeError('Healthy stability gate failed; no formal and fixed design retained')
   sp.run([sys.executable,str(B/'prepare.py'),'finalize',stage],check=True,cwd=r.ROOT);ready('plan_ready.json')
  for layout in ('legacy','node_local_pp'):
   plan=r.read(D/f'{kind}_{layout}_plan.json')
   assert plan.get('stage_forward_aux_enabled') is (kind=='formal')
   assert not plan.get('controlled_profile_requests')
   if kind=='formal':
    assert plan['frozen_model_sha256']==r.freeze['model_sha256']
    assert plan['stage_forward_aux_model'].endswith('peer_v1/model_scoped.json')
    assert plan['stage_forward_aux_context_binding'].endswith('peer_v1/model_scoped_context_binding.json')
   r.rendezvous_port=(29970 if stage=='l1' else 29980)+(0 if kind=='timeonly' else 2)+(1 if layout=='legacy' else 2)
   r.write(D/f'{kind}_{layout}_launcher.json',dict(session=plan['session_id'],rendezvous_port=r.rendezvous_port,redis_port=17422,supervisor='owned detached',telemetry_period_s=5))
   S=r.detached_run(plan)
   if kind=='formal':
    events=[json.loads(x) for x in (S/'host2/events.jsonl').read_text().splitlines() if x]
    r.write(D/f'{layout}_predictions_without_truth.json',dict(layout=layout,session=S.name,events=[e for e in events if e['event'] in ('detection','profile_schedule','localization')]))
 r.write(D/'execution_complete.json',dict(status='complete',fixed_denominator=2 if stage=='l1' else 4))
 r.write(D/'active.json',dict(status='complete_pending_independent_score',gpu_owner=None))
except BaseException as error:
 r.write(D/'execution_failure.json',dict(error=repr(error),active_session=r.active,no_automatic_training_retry=True));raise
finally:
 if r.active:
  for host,container,sudo in r.NODES:
   try:r.rpc(host,r.dock(host,container,sudo,['touch',f'results/data16_sessions/{r.active}/STOP']))
   except Exception:pass
 for host,pid,stop,log in telemetry:
  try:
   r.rpc(host,['touch',stop]);time.sleep(1);(D/f'telemetry_{host}.jsonl').write_text(r.rpc(host,['cat',log]))
  except Exception:pass
 r.write(D/'telemetry_metadata.json',dict(owner=owner,collectors=[dict(host=h,pid=p,stop=s,log=l) for h,p,s,l in telemetry],period_s=5,inference_feature=False))
