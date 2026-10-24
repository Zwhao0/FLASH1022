"""Check staged production vs tested isolated compute, plus old dense contracts."""
import ast,hashlib,importlib.util,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=HERE/'production';CANDIDATE=HERE.parent/'sparse3_gc_draft';sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def tree(path):return ast.parse(path.read_text())
def function(path,name):return next(n for n in tree(path).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
def same(left,right,name):
    a,b=function(left,name),function(right,name);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
    return hashlib.sha256(ast.dump(a,include_attributes=False).encode()).hexdigest()
def main():
    binding=json.loads((HERE/'source_binding.json').read_text())
    for relative,item in binding['production_files'].items():
        assert sha(OUT/relative)==item['after_sha256']
        assert (sha(ROOT/relative) if (ROOT/relative).exists() else None)==item['before_sha256']
        assert 'results.layer_campaign' not in (OUT/relative).read_text()
    mapping={}
    for filename,name,candidate in [('stage_forward_sampling.py','Sampler','sampling.py'),('stage_forward_sparse_collector.py','StageForwardEvidence','collector.py'),('stage_forward_sparse_peer.py','infer_bound','sparse_peer.py')]:
        mapping[name]=same(OUT/'flash_v1'/filename,CANDIDATE/candidate,name)
    # The frozen scoring body is exactly the candidate's build-time transformed
    # _score; the inference body remains the original binding/threshold math.
    candidate_math=load('candidate_sparse_math_for_mapping',HERE.parent/'sparse3_offline/adapter.py')
    expected=ast.parse(candidate_math.source).body[0]
    actual=function(OUT/'flash_v1/stage_forward_sparse_math.py','_score')
    assert ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)
    mapping['static_score']=hashlib.sha256(ast.dump(actual,include_attributes=False).encode()).hexdigest()
    bound=function(OUT/'flash_v1/stage_forward_sparse_math.py','_infer_bound')
    original=function(ROOT/'flash_v1/stage_forward_peer.py','infer_bound')
    assert [ast.dump(n,include_attributes=False) for n in bound.body]==[ast.dump(n,include_attributes=False) for n in original.body]
    mapping['static_infer_body']=hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in bound.body).encode()).hexdigest()
    math_tree=tree(OUT/'flash_v1/stage_forward_sparse_math.py')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','eval','compile') for n in ast.walk(math_tree))
    assert 'import inspect' not in (OUT/'flash_v1/stage_forward_sparse_math.py').read_text()
    assert 'Offline-only' not in (OUT/'flash_v1/stage_forward_sparse_math.py').read_text()
    header=next(n for n in math_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_PEER_SOURCE_SHA256' for t in n.targets))
    assert ast.literal_eval(header.value)==sha(ROOT/'flash_v1/stage_forward_peer.py')
    for name in ('diagnose','diagnose_timeout','_combine_stage_forward','controller'):
        mapping[name]=same(OUT/'flash_v1/online_v2.py',ROOT/'flash_v1/online_v2.py',name)
    modules={}
    for name in ('stage_forward_sampling','stage_forward_sparse_math','stage_forward_sparse_collector','stage_forward_sparse_peer'):
        modules[name]=load('flash_v1.'+name,OUT/'flash_v1'/f'{name}.py')
    production=load('staged_production_online',OUT/'flash_v1/online_v2.py')
    tests=load('isolated_sparse_tests',CANDIDATE/'test_cpu.py');tests.online=production
    # Same consumer protocol tests now execute staged production instead of the isolated consumer.
    suite=unittest.TestSuite(tests.SparseProtocolTests(name) for name in (
        'test_sparse_consumer_missing_duplicate_late_context_and_core_fallback',
        'test_parent_child_offsets_and_unobserved','test_frozen_model_raw8_to3_with_explicit_runtime_manifest'))
    # Existing complete8/default128 priority, deadline, hint tests exercise the unchanged legacy path.
    oldtests=load('old_priority_tests',ROOT/'flash_v1/tests/test_peer_priority.py');oldtests.online_v2=production
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(oldtests.PeerPriorityTests))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(suite)
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    # Source-level epoch0 branch cannot invoke the new sampler in either plan mode.
    staged_tree=tree(OUT/'flash_v1/train_pipeline.py')
    gate=next(n for n in ast.walk(staged_tree) if isinstance(n,ast.If) and '.selected(' in ast.unparse(n.test))
    from types import SimpleNamespace
    for enabled in (True,False):
        observer=SimpleNamespace(selected=lambda *a:(_ for _ in ()).throw(AssertionError('idle sampler')))
        scope=dict(evidence_active=False,sparse_aux_enabled=enabled,active_stage_aux=observer)
        module=ast.Module(body=[gate],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,'epoch0','exec'),scope)
        assert scope['active_stage_aux'] is observer
    from flash_v1.stage_forward_evidence import StageForwardEvidence as dense
    sparse=modules['stage_forward_sparse_collector'].StageForwardEvidence
    assert sparse.iteration is dense.iteration and sparse.drain is dense.drain and sparse._event is dense._event
    # Four complete saved windows: two contexts x two layouts. Only attach the
    # explicit sampling manifest; work/timing/model inputs are unchanged.
    from results.layer_campaign.stage_forward_v1.sparse3_gc_draft import sparse_peer as candidate_peer
    from results.layer_campaign.stage_forward_v1.sparse3_gc_draft.sampling import contract
    raw_equal=[]
    for context,label,pattern in [('plain','plain','v2s_sameprocess_baseline_{}_01'),('scoped_dispatch','scoped','v2s_scope_guard_reval_l3_{}_01')]:
        for layout in ('legacy','node_local_pp'):
            directory=ROOT/'results/data16_sessions'/pattern.format(layout);rows=[];hashes={}
            for path in sorted(directory.glob('host*/stage_forward_aux_rank*.jsonl')):
                hashes[str(path.relative_to(ROOT))]=sha(path)
                with path.open() as stream:
                    for line in stream:
                        row=json.loads(line)
                        if row['epoch']>1:break
                        if row['epoch']==1:rows.append(row)
            start=min(r['iteration'] for r in rows)
            selected=[dict(r,sampling_contract=contract(1,start)) for r in rows if start<=r['iteration']<start+8 and r['iteration']-start in (0,2,6)]
            model=json.loads((HERE.parent/'peer_v1'/f'model_{label}.json').read_text());binding_model=json.loads((HERE.parent/'peer_v1'/f'model_{label}_context_binding.json').read_text())
            before=candidate_peer.infer_bound(selected,model,list(range(16)),binding_model)
            after=modules['stage_forward_sparse_peer'].infer_bound(selected,model,list(range(16)),binding_model)
            assert before==after,(context,layout)
            offline=candidate_math.infer_sparse(selected,model,list(range(16)),binding_model,dict(schema='offline_sparse_aux_v1',window_size=8,mask='fixed3',start=start,epoch=1))
            assert {k:v for k,v in offline.items()if k!='sampling_contract'}=={k:v for k,v in after.items()if k!='sampling_contract'},('offline3',context,layout)
            raw_equal.append(dict(context=context,layout=layout,rows=len(selected),outputs_equal=True,status=after['status'],rank=after['rank'],source_hashes=hashes))
    report=dict(status='passed_CPU_draft_not_applied',tests_passed=result.testsRun,compute_AST_mapping=mapping,
        live_collector_sha256=sha(ROOT/'flash_v1/stage_forward_evidence.py'),patch_sha256=sha(HERE/'production.patch'),
        no_results_code_imports=True,old_default_full8_contract_preserved=True,epoch0_noop_chain_inherited=True,
        models_thresholds_unchanged=True,no_GPU_started=True,static_runtime_math=True,raw_output_equivalence=raw_equal)
    (HERE/'cpu_validation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
