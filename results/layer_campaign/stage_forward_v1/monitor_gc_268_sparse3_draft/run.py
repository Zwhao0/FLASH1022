"""Owned GPU entry; default --check is strictly local CPU-only. No automatic retry."""
import base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
D=HERE
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def check(staged=False):
    ready=read(D/'ready.json');contract=read(D/'contract.json');freeze=read(D/'freeze.json')
    assert ready['contract_sha256']==sha(D/'contract.json') and ready['freeze_sha256']==sha(D/'freeze.json')
    assert ready['runner_sha256']==sha(__file__) and ready['backend_tar_sha256']==sha(D/'isolated_backend.tar')
    assert contract['source_freeze_sha256']==sha(D/'freeze.json') and contract['max_measurement_sessions']==2
    for name,expected in {**freeze['files'],**freeze['test_and_runner_files']}.items():
        location=ROOT/name
        if staged and name in read(D/'source_binding.json')['production_files']:
            location=ROOT/'results/layer_campaign/stage_forward_v1/sparse3_gc_production_draft/production'/name
        if sha(location)!=expected:raise ValueError('source mismatch: '+name)
    for layout,entry in contract['plans'].items():
        p=ROOT/entry['path'];assert sha(p)==entry['sha256'];plan=read(p)
        assert plan['parallel_layout']==layout and plan['steps']==268 and plan['seed']==1348533921
        assert plan['online'] and plan['evidence_enabled'] and plan['stage_forward_aux_enabled']
        assert plan['monitor_profile_transition'] and plan['max_profiles']==3 and plan['controlled_manual_profile_limit']==0
        assert plan['controlled_profile_requests']==[] and plan['microbatches']==2 and not plan['monitor_benchmark']
        assert plan['cost_harness_files']==freeze['cost_harness_files']
        assert not any(plan.get(key) for key in ('scoped_compute_enabled','scoped_catalog_only','native_module_fma_cases','whole_compute_reference_session'))
        assert all(not i.get('target_ranks') and not i.get('delay_ms') and not i.get('slowdown_extra_ratio') for i in plan['intervals'])
    return contract,freeze

def execute(contract,freeze,layout):
    contract=dict(contract,plans={layout:contract['plans'][layout]})
    sys.path.insert(0,str(ROOT))
    from results.layer_campaign.continuation_v2.gates import owned_runner as r
    if any((D/name).exists() for name in (f'execution_start_{layout}.json',f'execution_complete_{layout}.json',f'execution_failure_{layout}.json')):
        raise FileExistsError('single baseline run already started; no retry')
    r.D=D;r.freeze=freeze
    # Deploy only the frozen isolated package; baseline production source is never overwritten.
    encoded=base64.b64encode((D/'isolated_backend.tar').read_bytes()).decode()
    for host,container,sudo in r.NODES:
        code="import base64,io,tarfile,sys;tarfile.open(fileobj=io.BytesIO(base64.b64decode(sys.stdin.read()))).extractall('.')"
        r.rpc(host,r.dock(host,container,sudo,['python','-c',code]),input=encoded)
    r.check_sources()
    for host,container,sudo in r.NODES:
        code='from pathlib import Path;names='+repr([v['session_id'] for v in contract['plans'].values()])+";assert not any(Path('results/data16_sessions',n).exists() for n in names)"
        r.rpc(host,r.dock(host,container,sudo,['python','-c',code]))
    save(D/f'execution_start_{layout}.json',dict(contract_sha256=sha(D/'contract.json'),freeze_sha256=sha(D/'freeze.json'),GPU_owner='extended64_protocol',no_automatic_retry=True))
    sessions={}
    try:
        for index,(layout,entry) in enumerate(contract['plans'].items()):
            check();plan=read(ROOT/entry['path']);r.rendezvous_port=30520+index
            session=r.detached_run(plan)
            sessions[layout]=dict(status='complete',path=str(session.relative_to(ROOT)),session_id=plan['session_id'],plan_sha256=sha(session/'plan.json'))
            save(D/f'completed_sessions_{layout}.json',sessions)
        save(D/f'execution_complete_{layout}.json',dict(status='complete',sessions=sessions,contract_sha256=sha(D/'contract.json'),source_freeze_sha256=sha(D/'freeze.json'),measurement_only=True))
    except BaseException as error:
        save(D/f'execution_failure_{layout}.json',dict(error=repr(error),sessions=sessions,active_session=r.active,no_automatic_retry=True));raise
    finally:
        if r.active:
            for host,container,sudo in r.NODES:
                try:r.rpc(host,r.dock(host,container,sudo,['touch',f'results/data16_sessions/{r.active}/STOP']))
                except Exception:pass

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--layout',choices=['legacy','node_local_pp'],default='legacy');args=parser.parse_args()
    contract,freeze=check(staged=not args.execute)
    if args.execute:execute(contract,freeze,args.layout)
    else:print(json.dumps(dict(status='passed_local_CPU_smoke_no_GPU_or_network',plans=contract['plans'],contract_sha256=sha(D/'contract.json'),freeze_sha256=sha(D/'freeze.json')),sort_keys=True))
if __name__=='__main__':main()
