"""Pin three original cases for the final no-diagnostic runtime smoke. No launch."""
import copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PROD=HERE.parent/'sparse3_gc_production_draft'
LEGACY=ROOT/'results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_reverted_baseline/formal_legacy_plan.json'
PP_BASE=LEGACY.with_name('formal_node_local_pp_plan.json')
PP_CASE=HERE.parent/'peer_v1/upper6/l2/formal_node_local_pp_plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
    source=read(ROOT/'results/layer_campaign/scope_guard_v1/peer_priority_v1/source_freeze.json')
    binding=read(PROD/'source_binding.json');files=dict(source['files']);files.update({k:v['after_sha256']for k,v in binding['production_files'].items()})
    freeze=dict(schema='production_sparse3_startup_gc_no_diag_smoke3_source_v1',files=files,binary_files=source.get('binary_files',{}),production_patch_sha256=sha(PROD/'production.patch'),diagnostic_overlay=False)
    save(HERE/'source_freeze.json',freeze);entries={};case_rows=[]
    for layout,parent,definition,stop in [('legacy',LEGACY,LEGACY,151),('node_local_pp',PP_BASE,PP_CASE,89)]:
        base=read(parent);defined=read(definition);plan=copy.deepcopy(base)
        if layout=='legacy':
            faults=[copy.deepcopy(r)for r in defined['intervals']if r.get('target_ranks')and r['end']<=136]
            intervals=[copy.deepcopy(r)for r in defined['intervals']if r['start']<=136]
            tail=copy.deepcopy(next(r for r in defined['intervals']if r['start']==137));tail['end']=stop;intervals.append(tail)
        else:
            faults=[copy.deepcopy(r)for r in defined['intervals']if r.get('target_ranks')and r['fault_scope']==dict(kind='stage',phase='fwd')]
            intervals=[copy.deepcopy(r)for r in defined['intervals']if r['start']<=74]
            tail=copy.deepcopy(next(r for r in defined['intervals']if r['start']==75));tail['end']=stop;intervals.append(tail)
        assert len(faults)==(2 if layout=='legacy'else 1)
        assert [r for r in intervals if r.get('target_ranks')]==faults
        assert intervals[0]['start']==1 and intervals[-1]['end']==stop
        assert all(a['end']+1==b['start']for a,b in zip(intervals,intervals[1:]))
        for row in faults:
            assert row['end']-row['start']+1==32
            pure_before=[r for r in intervals if not r.get('target_ranks')and r['end']<row['start']and r['start']>=row['start']-30]
            assert sum(r['end']-r['start']+1 for r in pure_before)>=30
            case_rows.append(dict(layout=layout,case_id=row['case_id'],seed=defined['seed'],fault_interval=row,parent_definition=str(definition.relative_to(ROOT)),parent_sha256=sha(definition)))
        plan.update(seed=defined['seed'],steps=stop,intervals=intervals,session_id=f'v2s_production_nodiag_smoke3_{layout}_01',protocol='production_no_diag_runtime_difference_smoke3_v1',fixed_denominator=len(faults),execution_role='three_case_diag_to_no_diag_runtime_smoke_not_new_accuracy_campaign',freeze_manifest_path=str((HERE/'source_freeze.json').relative_to(ROOT)),freeze_manifest_sha256=sha(HERE/'source_freeze.json'),gc_callbacks_diagnostic=False,monitor_cost_timing=False,startup_gc_fixed_model=True)
        # The launcher writes the exact sparse3 CONFIG and lifetime policy.
        import importlib.util
        spec=importlib.util.spec_from_file_location('prepare_staged_entry',PROD/'staged_entry.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with entry.launcher('/workspace/FLASH1022/tmp/no_diag_smoke3_prepare'):
            from flash_v1.startup_gc import configure_plan
            plan=configure_plan(plan)
        save(HERE/f'formal_{layout}_plan.json',plan);entries[layout]=dict(path=str((HERE/f'formal_{layout}_plan.json').relative_to(ROOT)),sha256=sha(HERE/f'formal_{layout}_plan.json'),session_id=plan['session_id'],steps=stop)
    contract=dict(status='CPU_ready_only_after_NEWreg8_gate_and_production_apply',protocol='production_no_diag_runtime_difference_smoke3_v1',fixed_cases=3,plans=entries,cases=case_rows,source_freeze_sha256=sha(HERE/'source_freeze.json'),source_freeze_path=str((HERE/'source_freeze.json').relative_to(ROOT)),production_binding_sha256=sha(PROD/'source_binding.json'),scope='Only diagnostic-to-no-diagnostic runtime difference; no new full-campaign accuracy estimate',gate='All three original cases target-correct with fault-contained complete8 core512/128markers + aux48 strict[1,3,7]/min3; pure-health localization0; any regression blocks and expands checks; no truth-based retiming/dose/threshold changes',health='12 warmup+30 healthy13..42 before first fault; legacy recovery75..104 healthy30; original32step faults,15step final tail; automatic BEAST and core-only continuation unchanged',production_apply_required=True,no_GPU_started=True)
    save(HERE/'contract.json',contract);save(HERE/'ready.json',dict(status=contract['status'],contract_sha256=sha(HERE/'contract.json'),source_freeze_sha256=sha(HERE/'source_freeze.json'),runner_sha256=sha(HERE/'run.py'),no_GPU_started=True))
    print('CPU pinned legacy2+PP1 original cases, seeds/doses preserved; no launch')
if __name__=='__main__':main()
