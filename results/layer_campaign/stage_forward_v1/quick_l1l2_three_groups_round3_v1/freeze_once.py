"""CPU-only one draw from healthy calibration; three independent short groups, six L1/L2 cases."""
import copy,hashlib,json,os,random,secrets,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
from flash_v1.startup_gc import configure_plan,validate_plan
from results.layer_campaign.continuation_v2.gates import prepare as base
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 index=read(ROOT/'tmp/v2_github_tree_elements.json');submitted={r['path']:r['sha256'] for r in index['tree_elements']}
 source_path=HERE.parent/'postcommit_random_check_v1/source_freeze.json';f=read(source_path)
 for path,digest in f['files'].items():assert sha(ROOT/path)==digest,path
 for path,digest in f['submitted_runtime_and_model_files_verified'].items():assert sha(ROOT/path)==digest and submitted[path]==digest,path
 fd=os.open(HERE/'seed_seal.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as stream:
  token=secrets.token_hex(16);seal=dict(seed=int(token,16),seed_hex=token,created_unix=time.time(),one_draw_only=True,submitted_commit='5cc7cb8347d1efb91ef37a143e1428e82b15086d');json.dump(seal,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 rng=random.Random(seal['seed']);training_seed=rng.randrange(1,2**31);phases=['fwd','bwd',rng.choice(['fwd','bwd'])];layouts=['legacy','node_local_pp',rng.choice(['legacy','node_local_pp'])];groups=list(zip(layouts,phases));rng.shuffle(groups)
 old=read(ROOT/base.OLD_DESIGN_PATH);att,_=base.attestation_inputs(old);cases=[];plans={}
 (HERE/'source_freeze.json').write_bytes(source_path.read_bytes());source_sha=sha(HERE/'source_freeze.json')
 for li,(layout,stage_phase) in enumerate(groups):
  group=f'group{li+1}'
  baseline=base.old_whole_baseline(old,layout);plan=copy.deepcopy(read(HERE.parent/'production_no_diag_smoke3'/f'formal_{layout}_plan.json'));order=['L1','L2'];rng.shuffle(order);intervals=[dict(id='warmup',start=1,end=12,record_start=1,target_ranks=[],fault_mode=None,fault_scope=None,slowdown_extra_ratio=0.,segment='warmup')];previous=12
  for ci,tier in enumerate(order):
   rank=rng.randrange(16);fraction=rng.uniform(.50,.70);phase=None if tier=='L1' else stage_phase;dose,_=base.dose_for(layout,rank,'whole' if tier=='L1' else phase,fraction,baseline,att,rng);target=dict(kind='whole') if tier=='L1' else dict(kind='stage',phase=phase);onset=44+rng.randrange(2) if ci==0 else previous+31+rng.randrange(3)
   intervals.append(dict(id=f'healthy{ci}',start=previous+1,end=onset-1,record_start=previous+1,target_ranks=[],fault_mode=None,fault_scope=None,slowdown_extra_ratio=0.,segment='healthy'))
   fault=dict(id=f'quick-r3-{group}-{layout}-{ci}',case_id=f'quick-r3-{group}-{layout}-{ci}',start=onset,end=onset+23,record_start=onset,target_ranks=[rank],fault_mode='whole_compute',fault_scope=target,slowdown_extra_ratio=dose['slowdown_extra_ratio'],segment='fault');intervals.append(fault);previous=onset+23
   cases.append(dict(group=group,tier=tier,layout=layout,rank=rank,phase=phase,seed=training_seed,dose=dose,fault_target=target,fault_interval=fault,preflight_valid=True))
  tail=rng.randrange(9,11);intervals.append(dict(id='tail',start=previous+1,end=previous+tail,record_start=previous+1,target_ranks=[],fault_mode=None,fault_scope=None,slowdown_extra_ratio=0.,segment='healthy'))
  plan.update(session_id=f'v2s_quick3_r3_{group}_{layout}_{token[:8]}',protocol='quick_l1l2_three_groups_v1',submitted_commit=seal['submitted_commit'],seed=training_seed,intervals=intervals,steps=previous+tail,fixed_denominator=2,controlled_profile_requests=[],gc_callbacks_diagnostic=False,monitor_cost_timing=False,freeze_manifest_path=str((HERE/'source_freeze.json').relative_to(ROOT)),freeze_manifest_sha256=source_sha)
  plan=configure_plan(plan);assert configure_plan(plan)==plan and validate_plan(plan) is True and 130<=plan['steps']<=134
  path=HERE/f'formal_{group}_plan.json';save(path,plan);plans[group]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),steps=plan['steps'],session_id=plan['session_id'],layout=layout,L2_phase=stage_phase)
 c=dict(protocol='quick_l1l2_three_groups_v1',submitted_commit=seal['submitted_commit'],draw_seed=seal['seed'],fixed_cases=6,preflight_valid=True,cases=cases,plans=plans,source_freeze_sha256=source_sha,production_binding_sha256=sha(HERE.parent/'sparse3_gc_production_draft/source_binding.json'),manual_profile_requests_forbidden=True,one_draw_no_retuning=True,short_test_boundary='24-step faults and 8-10-step tail; not original32/fullhealth-recovery campaign',causal_output_rule='earliest in-fault detection primary; first accepted public verdict never replaced by later correct window; no-trigger remains miss; both layouts public frozen before truth',launcher='metadata_only_cost_v1.owned_runner.detached_run(configure_plan(plan))',source_snapshot_archive=False)
 save(HERE/'contract.json',c);save(HERE/'ready.json',dict(status='quick_l1l2_three_groups_frozen_for_execution',contract_sha256=sha(HERE/'contract.json'),source_freeze_sha256=source_sha,runner_sha256=sha(HERE/'run.py'),seed_seal_sha256=sha(HERE/'seed_seal.json'),fixed_cases=6,CPU_only=True,GPU_executed=False))
 print(json.dumps(dict(status='CPU_frozen_once',cases=6,steps={k:v['steps'] for k,v in plans.items()},source_freeze_sha256=source_sha)))
if __name__=='__main__':main()
