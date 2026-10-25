"""One post-commit seed seal and CPU-only formal plan construction; never launches."""
import hashlib,importlib.util,json,os,secrets,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
COMMIT='5cc7cb8347d1efb91ef37a143e1428e82b15086d';TREE='6c25b838566475db02cdfba22d720ca2b9637d30'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(p)

def main():
 index=read(ROOT/'tmp/v2_github_tree_elements.json');payload=read(ROOT/'tmp/v2_upload_payload.json');contents={r['path']:r['content']for r in payload['tree']};verified={}
 for row in index['tree_elements']:
  path=row['path']
  if path.startswith('flash_v1/')and path.endswith(('.py','.cpp','.cu'))or path.endswith(('model_scoped.json','model_scoped_context_binding.json','linear_candidate_model.json')):
   raw=(ROOT/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'] and contents[path].encode()==raw,path;verified[path]=row['sha256']
 source=read(HERE.parent/'production_no_diag_smoke3/source_freeze.json')
 for path,expected in source['files'].items():assert sha(ROOT/path)==expected,path
 marker=HERE/'seed_seal.json'
 # Create exclusively before the only entropy call; failed builds retain the seal.
 fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w')as stream:
  seed_hex=secrets.token_hex(16);seal=dict(submitted_commit=COMMIT,submitted_tree=TREE,seed_hex=seed_hex,seed=int(seed_hex,16),created_unix=time.time(),one_draw_only=True,payload_sha256=sha(ROOT/'tmp/v2_upload_payload.json'),index_sha256=sha(ROOT/'tmp/v2_github_tree_elements.json'));stream.write(json.dumps(seal,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
 builder_path=HERE.parent/'postcommit_random_check_draft/build_fixture.py';spec=importlib.util.spec_from_file_location('postcommit_one_shot_builder',builder_path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
 built=builder.build(seal['seed'],COMMIT);cases=built['cases'];source.update(schema='postcommit_random_check_submitted_source_v1',submitted_commit=COMMIT,submitted_tree=TREE,submitted_index_sha256=seal['index_sha256'],submitted_payload_sha256=seal['payload_sha256'],submitted_runtime_and_model_files_verified=verified,tooling_note='Only CPU builder/runner metadata and event-absence handling added after submission; production runtime/model unchanged',postcommit_tools={str(p.relative_to(ROOT)):sha(p)for p in [builder_path,HERE/'run.py',Path(__file__)]})
 save(HERE/'source_freeze.json',source);source_sha=sha(HERE/'source_freeze.json');plans={}
 design=dict(submitted_commit=COMMIT,submitted_tree=TREE,draw_seed=seal['seed'],draw_seed_hex=seal['seed_hex'],seed_seal_sha256=sha(marker),fixed_cases=built['N'],preflight_valid=built['preflight_valid'],cases=cases,eligibility_census=built['eligibility_census'],not_population_accuracy=True,prediction_before_truth=True,no_redraw_or_retuning=True)
 save(HERE/'design.json',design);design_sha=sha(HERE/'design.json')
 for layout,plan in built['plans'].items():
  plan.update(session_id=f'v2s_postcommit_random_{layout}_{seed_hex[:8]}',execution_role='postcommit_random_spotcheck_not_population_accuracy',candidate_population_note='one seed after submitted commit; healthy-catalog eligibility, no old localization outcomes',formal_execution_gate='retain every drawn case; no replacement or retuning',regression_case_mix_per_layout='random tiers/layouts/rank/subtype/strength/onset; four tiers covered globally',design_sha256=design_sha,freeze_manifest_path=str((HERE/'source_freeze.json').relative_to(ROOT)),freeze_manifest_sha256=source_sha,submitted_commit=COMMIT)
  from flash_v1.startup_gc import configure_plan,validate_plan
  assert configure_plan(plan)==plan and validate_plan(plan)is True
  # The production scoped injector loads these exact rank catalogs, independently
  # of the private case metadata and unchanged controller.
  for case in [r for r in cases if r['layout']==layout]:
   rank=case['rank'];catalog_path=ROOT/'results/data16_sessions'/plan['scoped_reference_session']/f'scope_catalog_rank{rank}.json';catalog=read(catalog_path)
   kind=case.get('fault_target',{}).get('kind')
   if kind in('component','operator'):
    population=catalog['components'if kind=='component'else'operators'];assert any(row['slot_id']==case['fault_target']['slot_id']for row in population)
   if case['preflight_valid']:assert case['dose']['slowdown_extra_ratio']>0
  path=HERE/f'formal_{layout}_plan.json';save(path,plan);plans[layout]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),steps=plan['steps'],session_id=plan['session_id'])
 contract=dict(protocol='postcommit_compute_v2_random_4to6_v1',submitted_commit=COMMIT,submitted_tree=TREE,draw_seed=seal['seed'],fixed_cases=built['N'],preflight_valid=built['preflight_valid'],cases=cases,plans=plans,production_binding_sha256=sha(HERE.parent/'sparse3_gc_production_draft/source_binding.json'),design_sha256=design_sha,seed_seal_sha256=sha(marker),source_freeze_sha256=source_sha,manual_profile_requests_forbidden=True,single_fault_gpu_at_a_time=True,no_automatic_retry=True,health_scope='>=30 real monitored warm/recovery; count every healthy candidate; distinguish detection time from PROFILE/fault overlap',cost='cite final same-source complete ABBA mean and PROFILE budgets; never call injected-fault wall PROFILE overhead',predictions='seal both layouts public event snapshots before reading any injection truth')
 save(HERE/'contract.json',contract)
 ready=dict(status='postcommit_random_4to6_frozen_for_execution'if built['preflight_valid']else'drawn_invalid_no_execution_no_redraw',contract_sha256=sha(HERE/'contract.json'),source_freeze_sha256=source_sha,runner_sha256=sha(HERE/'run.py'),design_sha256=design_sha,seed_seal_sha256=sha(marker),submitted_commit=COMMIT,N=built['N'],layout_counts={l:sum(c['layout']==l for c in cases)for l in plans},CPU_only=True,GPU_executed=False)
 save(HERE/'ready.json',ready);print(json.dumps(ready,sort_keys=True))
if __name__=='__main__':main()
