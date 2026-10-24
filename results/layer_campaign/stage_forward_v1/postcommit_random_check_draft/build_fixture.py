"""CPU construction smoke with an explicit synthetic seed/fake commit, /tmp only."""
import argparse,copy,hashlib,importlib.util,json,random,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
from flash_v1.startup_gc import configure_plan,validate_plan
from results.layer_campaign.continuation_v2.gates import prepare as base

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def healthy_inputs():
 blind=module('randomcheck_blind_helpers','results/layer_campaign/scope_guard_v1/peer_priority_v1/blind8/prepare.py')
 l4=module('randomcheck_l4_helpers','results/layer_campaign/scope_guard_v1/peer_priority_v1/l4/prepare.py')
 old=read(ROOT/base.OLD_DESIGN_PATH);att,_=base.attestation_inputs(old);inputs={};census={}
 for layout in('legacy','node_local_pp'):
  refs=blind._catalog_refs_by_rank(layout);cats={}
  for rank,ref in refs.items():
   path=ROOT/ref['catalog_path'];assert sha(path)==ref['catalog_file_sha256'];cats[rank]=read(path)
   assert cats[rank]['catalog_sha256']==ref['catalog_sha256']
   assert sha(ROOT/ref['proof_path'])==ref['proof_sha256']
  components=blind._component_pools(cats);operators={}
  for rank,cat in cats.items():
   filtered,_=l4.filtered_fwd_operator_catalog(cat);operators[rank]=defaultdict(list)
   for op in filtered['operators']:operators[rank][l4.families.operator_family(op)].append(op)
  census[layout]={str(rank):{'component':{family:len(sites)for family,sites in components[rank].items()},'operator':{family:len(sites)for family,sites in operators[rank].items()}}for rank in range(16)}
  inputs[layout]=dict(refs=refs,catalogs=cats,components=components,operators=operators,baseline=base.old_whole_baseline(old,layout))
 return inputs,census,att,l4.families

def build(seed,commit):
 inputs,census,att,families=healthy_inputs();rng=random.Random(seed);n=rng.choice([4,5,6]);legacy=rng.randint(2,n-2)
 tiers=['L1','L2','L3','L4']+[rng.choice(['L1','L2','L3','L4'])for _ in range(n-4)];rng.shuffle(tiers)
 training_seed=rng.randrange(1,2**31);cases=[];plans={}
 for layout,count in [('legacy',legacy),('node_local_pp',n-legacy)]:
  data=inputs[layout];selected=[]
  for index in range(count):
   tier=tiers.pop(0);rank=rng.randrange(16);fraction=rng.uniform(.50,.70);phase=rng.choice(['fwd','bwd'])if tier=='L2'else None if tier=='L1'else'fwd';case=dict(tier=tier,layout=layout,rank=rank,seed=training_seed,phase=phase,preflight_valid=True)
   try:
    if tier in('L1','L2'):
     phase='whole'if tier=='L1'else phase;dose,_=base.dose_for(layout,rank,phase,fraction,data['baseline'],att,rng);target={'kind':'whole'}if tier=='L1'else{'kind':'stage','phase':phase}
    else:
     pool=data['components'if tier=='L3'else'operators'][rank]
     # Family membership is declared before draw. Empty drawn family stays invalid.
     names=['attention','routed_expert_mlp','moe_block_router','normalization']if tier=='L3'else['matrix_multiply','elementwise_activation','routing_index','reduction_normalization','softmax','fused_sdpa_attention']
     family=rng.choice(names);sites=pool.get(family,[]);case['family']=family
     if not sites:raise ValueError('drawn rank/family has no eligible nonzero healthy site')
     site=rng.choice(sites);dose=families.dose_from_catalog_site(site,data['baseline'],fraction,reference_steps=3)
     case['site']=site;case['scope_catalog_ref']=data['refs'][rank];target={'kind':'component'if tier=='L3'else'operator','phase':'fwd','slot_id':site['slot_id']}
    case.update(dose=dose,fault_target=target)
   except(ValueError,RuntimeError)as error:
    case.update(preflight_valid=False,invalid_reason=str(error));selected.append(case);cases.append(case);continue
   selected.append(case);cases.append(case)
  template=read(HERE.parent/'production_no_diag_smoke3'/f'formal_{layout}_plan.json');plan=copy.deepcopy(template)
  intervals=[dict(id='warmup',start=1,end=12,record_start=1,target_ranks=[],fault_mode=None,fault_scope=None,slowdown_extra_ratio=0.,segment='warmup')];previous=12
  for index,case in enumerate(selected):
   onset=(43 if index==0 else previous+31)+rng.randrange(16)
   intervals.append(dict(id=f'healthy{index}',start=previous+1,end=onset-1,record_start=previous+1,target_ranks=[],fault_mode=None,fault_scope=None,slowdown_extra_ratio=0.,segment='healthy'))
   fault=dict(id=f'random-{layout}-{index}',case_id=f'random-{layout}-{index}',start=onset,end=onset+31,record_start=onset,target_ranks=[case['rank']],fault_mode='whole_compute',fault_scope=case.get('fault_target'),slowdown_extra_ratio=case.get('dose',{}).get('slowdown_extra_ratio',0),segment='fault')
   if case.get('scope_catalog_ref'):fault['scope_catalog_ref']=case['scope_catalog_ref']
   intervals.append(fault);case['fault_interval']=fault;previous=onset+31
  intervals.append(dict(id='tail',start=previous+1,end=previous+30,record_start=previous+1,target_ranks=[],fault_mode=None,fault_scope=None,slowdown_extra_ratio=0.,segment='healthy'))
  plan.update(session_id=f'SYNTHETIC_ONLY_{layout}',protocol='postcommit_compute_v2_random_4to6_v1',submitted_commit=commit,seed=training_seed,intervals=intervals,steps=previous+30,fixed_denominator=count,controlled_profile_requests=[],gc_callbacks_diagnostic=False,monitor_cost_timing=False)
  plan['design_sha256']=hashlib.sha256(json.dumps(dict(seed=seed,commit=commit,cases=selected),sort_keys=True).encode()).hexdigest()
  plan=configure_plan(plan);assert validate_plan(plan)is True
  faults=[row for row in intervals if row['segment']=='fault']
  assert 43<=faults[0]['start']<=58 and plan['steps']-faults[-1]['end']>=30
  assert all(31<=right['start']-left['end']<=46 for left,right in zip(faults,faults[1:]))
  assert all(row['end']-row['start']+1==32 for row in faults)
  for key in('evidence_model','stage_forward_aux_model','stage_forward_aux_context_binding'):assert(ROOT/plan[key]).is_file()
  assert all(len(x['target_ranks'])==1 for x in intervals if x['segment']=='fault')
  plans[layout]=plan
 return dict(synthetic_only=True,seed=seed,commit=commit,N=n,cases=cases,plans=plans,eligibility_census=census,preflight_valid=all(c['preflight_valid']for c in cases),no_actual_seed_drawn=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic-seed',type=int,default=12345);p.add_argument('--output',default='/tmp/flash_postcommit_synthetic.json');a=p.parse_args();output=Path(a.output).resolve()
 if not(output==Path('/tmp')or Path('/tmp').resolve()in output.parents):p.error('synthetic fixture output must remain under /tmp')
 result=build(a.synthetic_seed,'SYNTHETIC_FAKE_COMMIT_NOT_SUBMITTED');output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(N=result['N'],preflight_valid=result['preflight_valid'],cases=[{k:c.get(k)for k in('tier','layout','rank','family','phase','preflight_valid','invalid_reason')}for c in result['cases']],output=str(output),no_GPU=True)))
if __name__=='__main__':main()
