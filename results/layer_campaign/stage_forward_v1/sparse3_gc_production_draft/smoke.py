"""CPU-only staged/live production entry smoke; stops before any SSH/GPU work."""
import argparse,hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class NetworkBoundaryReached(Exception):pass
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args();target=ROOT if args.live else HERE/'production'
    binding=json.loads((HERE/'source_binding.json').read_text())
    for path,item in binding['production_files'].items():assert sha(target/path)==item['after_sha256'],path
    for name in ('stage_forward_sampling','startup_gc','stage_forward_sparse_math','stage_forward_sparse_collector','stage_forward_sparse_peer'):
        load('flash_v1.'+name,target/'flash_v1'/f'{name}.py')
    launcher=load('production_entry_cpu',target/'flash_v1/scripts/data16.py')
    source_plan=dict(session_id='cpu_only_explicit_optin',online=True,evidence_enabled=True,stage_forward_aux_enabled=True,stage_forward_aux_model='model.json',stage_forward_aux_context_binding='binding.json')
    commands=[]
    def stop_before_network(command,**kwargs):
        commands.append(command);raise NetworkBoundaryReached()
    with tempfile.TemporaryDirectory() as temporary,patch.object(launcher,'ROOT',Path(temporary)),patch.object(launcher.sp,'run',side_effect=stop_before_network):
        try:launcher.run(source_plan,startup_gc_fixed_model=True)
        except NetworkBoundaryReached:pass
        else:raise AssertionError('CPU boundary was not reached')
        saved=json.loads((Path(temporary)/'results/data16_sessions/cpu_only_explicit_optin/plan.json').read_text())
        assert saved['startup_gc_fixed_model'] is True and saved['startup_gc_policy']['fixed_model_process_lifetime'] is True
        assert saved['stage_forward_aux_sampling']['sample_offsets']==[1,3,7]
        assert len(commands)==1 and commands[0][0]=='ssh'
        assert 'startup_gc_fixed_model' not in source_plan
    production=load('actual_production_consumer_cpu',target/'flash_v1/online_v2.py')
    tests=load('sparse_entry_tests',HERE.parent/'sparse3_gc_draft/test_cpu.py');tests.online=production
    suite=unittest.TestSuite(tests.SparseProtocolTests(name) for name in ('test_sparse_consumer_missing_duplicate_late_context_and_core_fallback','test_parent_child_offsets_and_unobserved','test_frozen_model_raw8_to3_with_explicit_runtime_manifest'))
    old=load('core_priority_entry_tests',ROOT/'flash_v1/tests/test_peer_priority.py');old.online_v2=production
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(old.PeerPriorityTests))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(suite)
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    report=dict(status='passed_CPU_no_network_GPU',source='live' if args.live else 'staged',real_entry_called='flash_v1.scripts.data16.run(plan,startup_gc_fixed_model=True)',plan_explicit_config_and_policy=True,network_commands_executed=0,consumer_tests_passed=result.testsRun,default_optimized=False,patch_sha256=sha(HERE/'production.patch'))
    (HERE/('live_smoke.json' if args.live else 'staged_smoke.json')).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
