"""Prepare frozen final-production-context MONITOR replay. Never launches."""
import ast,copy,hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BASE=HERE.parent/'monitor_sameprocess_draft';PROD=HERE.parent/'sparse3_gc_production_draft';LIFE=HERE.parent/'sparse3_gc_draft';PKG='results.layer_campaign.stage_forward_v1.monitor_gc_268_sparse3_draft'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def one(s,a,b):assert s.count(a)==1,a;return s.replace(a,b)
def main():
    assert not (HERE/'execution_start_legacy.json').exists()
    text=(BASE/'runtime.py').read_text()
    text=one(text,'from .contract import position','from .contract import position\nfrom flash_v1.startup_gc import install,restore_owned')
    text=one(text,'        optimizer.step=measured_optimizer_step','        optimizer.step=measured_optimizer_step\n        self.heap_policy=install(torch,rank,plan,output)\n        self.setup.append(dict(kind="startup_heap_collect_freeze",seconds=self.heap_policy.startup_seconds))')
    a=text.index('        # Manual request published after logical39');b=text.index('        self.args.online=enabled',a);text=text[:a]+text[b:]
    text=text.replace('<=396','<=268').replace('in (33,41)','==33').replace('in (48,64)','==64').replace("expected_epoch=(1 if item['block']==3 else 2) if item['profile'] else 0","expected_epoch=0").replace("len(self.blocks)==6","len(self.blocks)==4").replace("len(self.blocks)!=6","len(self.blocks)!=4").replace('incomplete six replay blocks','incomplete four replay blocks')
    text=one(text,"        self._boundary('done',397);self.finished=True;self._save()","        self._boundary('done',269)\n        self.setup.append(dict(kind='startup_heap_restore',seconds=restore_owned(self.heap_policy,self.output,'complete')))\n        self.finished=True;self._save()")
    text=one(text,'        self._save()\n        self.optimizer.step=self.original_optimizer_step',"        restore_owned(self.heap_policy,self.output,'aborted')\n        self._save()\n        self.optimizer.step=self.original_optimizer_step")
    (HERE/'runtime.py').write_text(text)
    text=(LIFE/'train_pipeline_cost.py').read_text().replace('results.layer_campaign.stage_forward_v1.sparse3_gc_draft.runtime',PKG+'.runtime').replace('from results.layer_campaign.stage_forward_v1.sparse3_gc_draft.collector import StageForwardEvidence','from flash_v1.stage_forward_sparse_collector import StageForwardEvidence')
    (HERE/'train_pipeline_cost.py').write_text(text)
    text=(PROD/'production/flash_v1/scripts/node_collection.py').read_text()
    text=one(text,"                last_iteration = max(row['end'] for row in plan['intervals'])","                last_iteration = 204 if plan.get('monitor_sameprocess_cost') else max(row['end'] for row in plan['intervals'])")
    (HERE/'node_collection_cost.py').write_text(text)
    text=(BASE/'node_cost.py').read_text().replace('results.layer_campaign.stage_forward_v1.monitor_sameprocess_draft',PKG).replace("spec.get('steps')!=396","spec.get('steps')!=268")
    (HERE/'node_cost.py').write_text(text)
    text=(BASE/'controller_cost.py').read_text().replace('requests={179:1,243:2}','requests={}').replace('manual_limit=2','manual_limit=0')
    (HERE/'controller_cost.py').write_text(text)
    text=(LIFE/'run.py').read_text().replace("plan['steps']==140","plan['steps']==268").replace("plan['controlled_manual_profile_limit']==2","plan['controlled_manual_profile_limit']==0").replace("plan['controlled_profile_requests']==[59,91]","plan['controlled_profile_requests']==[]")
    # CPU preparation may check staged expected production; execution always requires actual files.
    text=text.replace('def check():','def check(staged=False):')
    text=one(text,"        if sha(ROOT/name)!=expected:raise ValueError('source mismatch: '+name)","        location=ROOT/name\n        if staged and name in read(D/'source_binding.json')['production_files']:\n            location=ROOT/'results/layer_campaign/stage_forward_v1/sparse3_gc_production_draft/production'/name\n        if sha(location)!=expected:raise ValueError('source mismatch: '+name)")
    text=text.replace('contract,freeze=check()','contract,freeze=check(staged=not args.execute)')
    (HERE/'run.py').write_text(text)
    names=['contract.py','runtime.py','node_cost.py','node_collection_cost.py','controller_cost.py','train_pipeline_cost.py']
    for name in names:ast.parse((HERE/name).read_text())
    files={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names};binding=read(PROD/'source_binding.json');save(HERE/'source_binding.json',binding)
    base=read(LIFE/'freeze.json');prod={n:h for n,h in base['files'].items() if not n.startswith('results/')}
    prod.update({n:v['after_sha256'] for n,v in binding['production_files'].items()})
    freeze=dict(schema='startup_gc_monitor_ABBA268_v1',files={**prod,**files},cost_harness_files=files,collector_sha256=base['collector_sha256'],test_and_runner_files={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in ['build.py','run.py','test_cpu.py','source_binding.json']},production_gate_required=True,production_patch_sha256=sha(PROD/'production.patch'))
    save(HERE/'freeze.json',freeze)
    old=read(ROOT/'results/layer_campaign/monitor_sameprocess_v1/contract.json')
    contract=dict(status='CPU_prepared_after_production_gate_only',protocol='startup_gc_monitor_only_ABBA268_v1',gpu_execution_authorized=True,max_measurement_sessions=2,seed=1348533921,steps_per_layout=268,warmup=[1,12],blocks=[dict(block=i+1,mode=m,start=13+64*i,end=76+64*i,settle=[1,32],measure=[33,64]) for i,m in enumerate('ABBA')],monitor_pairs=[dict(A=1,B=2,logical=[33,64]),dict(A=4,B=3,logical=[33,64])],physical_ipc=dict(first=77,last=204,contiguous=True,no_A_publications=True,no_synthetic_samples=True),controlled_requests=[],profile_windows=[],monitor_only=True,profile_budget_claim=False,paired_cost_method=old['paired_cost_method'],bare=old['bare'],replay=old['replay'],B=dict(max_profiles=3,manual_profiles=0,quota_unexhausted=True,full_BEAST_recent3=True,natural_candidates_recorded_not_requested=True),startup_gc_policy=read(LIFE/'contract.json')['startup_gc_policy'],source_freeze_path=str((HERE/'freeze.json').relative_to(ROOT)),source_freeze_sha256=sha(HERE/'freeze.json'),plans={})
    contract['paired_cost_method']['primary'].pop('profile',None);contract['paired_cost_method'].pop('separate_metrics',None)
    contract['paired_cost_method']['uncertainty']['block_lengths']={'monitor_32_positions':4}
    contract['paired_cost_method']['decision']='MONITOR1% only: report AB and BA plus pooled ratio-of-summed-wall and stratified circular-block-bootstrap95%CI; CI crossing1% is inconclusive; retain negative values without pure negative-overhead claim'
    contract['paired_cost_method'].pop('metadata_savings',None)
    contract['replay']['restore_before_blocks']=[1,2,3,4]
    contract['replay']['timing']='fullwall original input through IPC/log/print; window33 pre-fence and64 end-fence/read; block digest/restore/setup and GC startup separately reported'
    contract['replay']['checks']=['every logical position identical input/loss','four equal final parameter/buffer/SGD/RNG digests','64 actual SGD calls each positive LR.001 and real parameter update']
    for layout in ('legacy','node_local_pp'):
        plan=read(LIFE/f'{layout}_plan.json');plan.update(session_id=f'v2s_sparse3_gc_monitor268_{layout}_01',steps=268,protocol=contract['protocol'],controlled_profile_requests=[],controlled_manual_profile_limit=0,integration_node_module=PKG+'.node_cost',cost_harness_files=files,source_freeze_path=contract['source_freeze_path'],freeze_manifest_sha256=contract['source_freeze_sha256'],cost_contract_path=str((HERE/'contract.json').relative_to(ROOT)),gc_callbacks_diagnostic=False,monitor_cost_timing=False)
        plan['intervals']=[dict(plan['intervals'][0],end=268)];save(HERE/f'{layout}_plan.json',plan)
        contract['plans'][layout]=dict(path=str((HERE/f'{layout}_plan.json').relative_to(ROOT)),sha256=sha(HERE/f'{layout}_plan.json'),session_id=plan['session_id'])
    save(HERE/'contract.json',contract)
    with tarfile.open(HERE/'isolated_backend.tar','w') as tar:
        for n in files:tar.add(ROOT/n,arcname=n)
    save(HERE/'ready.json',dict(status=contract['status'],contract_sha256=sha(HERE/'contract.json'),freeze_sha256=sha(HERE/'freeze.json'),backend_tar_sha256=sha(HERE/'isolated_backend.tar'),runner_sha256=sha(HERE/'run.py'),no_GPU_started=True))
if __name__=='__main__':main()
