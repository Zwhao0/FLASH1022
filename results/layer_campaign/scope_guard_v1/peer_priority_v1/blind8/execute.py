"""Prelocked blind8 after both full qualification gates; no redraw or early score."""
import hashlib,json,shlex,subprocess as sp,sys,time,uuid
from pathlib import Path
from results.layer_campaign.continuation_v2.gates import owned_runner as r
B=Path(__file__).resolve().parent;D=B;r.D=D;F=B.parent/'source_freeze.json';r.freeze=r.read(F)
resume_formal='--resume-formal' in sys.argv
telemetry=[];owner=uuid.uuid4().hex[:12]
def ready(name):
 j=r.read(D/name)
 for f,h in j['artifacts'].items():
  p=F if f=='source_freeze.json' else D/f
  assert hashlib.sha256(p.read_bytes()).hexdigest()==h,f
 assert hashlib.sha256(F.read_bytes()).hexdigest()==j['source_freeze_sha256']
 for f,h in r.freeze['files'].items():assert hashlib.sha256((r.ROOT/f).read_bytes()).hexdigest()==h,f
try:
 r.write(D/'active.json',dict(status='waiting_full_l1_l2',gpu_owner=None))
 print(json.dumps(dict(status='waiting_full_l1_l2',design_locked=True)),flush=True)
 while not (B.parent/'telemetry_metadata.json').exists():
  state=r.read(B.parent/'pipeline_active.json')
  if state.get('status')=='stopped_for_review':raise RuntimeError('Prior qualification gate failed; blind8 remains unexecuted')
  time.sleep(15)
 state=r.read(B.parent/'pipeline_active.json')
 assert state.get('stage')=='l2' and state.get('status')=='complete' and state.get('strict')==32,state
 if not resume_formal:sp.run([sys.executable,str(B/'prepare.py'),'authorize-timeonly'],check=True,cwd=r.ROOT)
 else:assert r.read(D/'timeonly_execution_complete.json')['status']=='complete'
 ready('timeonly_ready.json');r.check_sources()
 for host,container,sudo in r.NODES:
  script=f'/tmp/flash_blind8_{owner}.py';stop=f'/tmp/flash_blind8_{owner}_STOP';log=f'/tmp/flash_blind8_{owner}.jsonl'
  r.rpc(host,['python3','-c','from pathlib import Path;import sys;Path('+repr(script)+').write_text(sys.stdin.read())'],input=(r.ROOT/'results/layer_campaign/continuation_v1/gates/telemetry.py').read_text())
  pid=int(r.rpc(host,['sh','-c',f'nohup env FLASH_TELEMETRY_STOP={shlex.quote(stop)} python3 -u {shlex.quote(script)} >{shlex.quote(log)} 2>&1 </dev/null & echo $!']).strip());telemetry.append((host,pid,stop,log))
 time.sleep(10)
 resources={h:[json.loads(x) for x in r.rpc(h,['cat',l]).splitlines() if x] for h,p,s,l in telemetry};r.write(D/'initial_resources.json',resources)
 for kind in (('formal',) if resume_formal else ('timeonly','formal')):
  if kind=='formal':
   if not resume_formal:
    r.write(D/'timeonly_execution_complete.json',dict(status='complete',fixed_denominator=8))
    for host,pid,stop,log in telemetry:(D/f'telemetry_{host}.jsonl').write_text(r.rpc(host,['cat',log]))
    sp.run([sys.executable,str(B/'timeonly_gate.py')],check=True,cwd=r.ROOT)
   if not r.read(D/'timeonly_gate.json')['stable_for_formal']:raise RuntimeError('Healthy stability gate failed; no formal and fixed design retained')
   if not (D/'plan_ready.json').exists():sp.run([sys.executable,str(B/'prepare.py'),'finalize'],check=True,cwd=r.ROOT)
   ready('plan_ready.json')
  for layout in ('legacy','node_local_pp'):
   plan=r.read(D/f'{kind}_{layout}_plan.json')
   assert plan.get('stage_forward_aux_enabled') is (kind=='formal')
   assert not plan.get('controlled_profile_requests')
   if kind=='formal':
    assert plan['frozen_model_sha256']==r.freeze['model_sha256']
    assert plan['stage_forward_aux_model'].endswith('peer_v1/model_scoped.json')
    assert plan['stage_forward_aux_context_binding'].endswith('peer_v1/model_scoped_context_binding.json')
   r.rendezvous_port=30140+(0 if kind=='timeonly' else 2)+(0 if layout=='legacy' else 1)
   r.write(D/f'{kind}_{layout}_launcher.json',dict(session=plan['session_id'],rendezvous_port=r.rendezvous_port,redis_port=17422,supervisor='owned detached',telemetry_period_s=5))
   S=r.detached_run(plan)
   if kind=='formal':
    events=[json.loads(x) for x in (S/'host2/events.jsonl').read_text().splitlines() if x]
    r.write(D/f'{layout}_predictions_without_truth.json',dict(layout=layout,session=S.name,events=[e for e in events if e['event'] in ('detection','profile_schedule','localization')]))
 r.write(D/'execution_complete.json',dict(status='complete',fixed_denominator=8))
 sp.run([sys.executable,str(B.parent.parent.parent/'stage_forward_v1/cleanup_peer.py'),str(D),'v2s_peer_blind8_'],check=True,cwd=r.ROOT)
 assert r.read(D/'cleanup.json')['all_clear']
 sp.run([sys.executable,str(B/'score.py')],check=True,cwd=r.ROOT)
 result=r.read(D/'summary.json');assert result['strict_success']==8 and result['gate_passed']
 assert sum(v['healthy_wrong_gpu_localizations'] for v in result['by_layout'].values())==0
 r.write(D/'active.json',dict(status='complete',strict=8,gpu_owner=None))
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
