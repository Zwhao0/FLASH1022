"""Post-commit 4-to-6-case spot check; --check is local CPU-only, never applies."""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT));PROD=HERE.parent/'sparse3_gc_production_draft'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def check(live=False):
    ready=read(HERE/'ready.json');assert ready.get('status')=='postcommit_random_4to6_frozen_for_execution';c=read(HERE/'contract.json');f=read(HERE/'source_freeze.json');binding=read(PROD/'source_binding.json')
    assert ready['contract_sha256']==sha(HERE/'contract.json') and ready['source_freeze_sha256']==sha(HERE/'source_freeze.json')and ready['runner_sha256']==sha(__file__)
    assert c['production_binding_sha256']==sha(PROD/'source_binding.json')and 4<=c['fixed_cases']<=6
    for path,item in binding['production_files'].items():
        location=ROOT/path if live else PROD/'production'/path
        assert sha(location)==item['after_sha256'],('production_no_diag_not_applied_or_source_mismatch',path)
    assert len(c['cases'])==c['fixed_cases'] and c.get('submitted_commit') and c.get('draw_seed') is not None
    assert set(row['tier'] for row in c['cases'])=={'L1','L2','L3','L4'}
    assert set(c['plans'])=={'legacy','node_local_pp'}
    assert all(sum(row['layout']==layout for row in c['cases'])>=2 for layout in c['plans'])
    assert all(type(row['rank']) is int and 0<=row['rank']<16 for row in c['cases'])
    for layout,e in c['plans'].items():
        assert sha(ROOT/e['path'])==e['sha256'];p=read(ROOT/e['path']);assert p['steps']==e['steps'] and p['stage_forward_aux_sampling']['expected_aux_rows']==48 and p['startup_gc_fixed_model']and not p['gc_callbacks_diagnostic']and not p['monitor_cost_timing']
        assert p.get('controlled_profile_requests',[])==[] and not p.get('monitor_benchmark')
        faults=sorted([r for r in p['intervals'] if r.get('target_ranks')],key=lambda r:r['start'])
        assert len(faults)==sum(r['layout']==layout for r in c['cases'])
        assert 43<=faults[0]['start']<=58
        for index,fault in enumerate(faults):
            assert len(fault['target_ranks'])==1 and fault['end']-fault['start']+1==32
            if index:assert 31<=fault['start']-faults[index-1]['end']<=46
        assert p['steps']>=faults[-1]['end']+30
        for row in [r for r in c['cases']if r['layout']==layout]:assert row['fault_interval']in p['intervals']and row['seed']==p['seed'] and row['fault_interval']['target_ranks']==[row['rank']]
    return c,f

def execute(c,f):
    if any((HERE/n).exists()for n in ('execution_start.json','execution_complete.json','execution_failure.json')):raise FileExistsError('one authorized spot check only; recover postprocessing on CPU without GPU retry')
    from flash_v1.scripts.data16 import run
    from results.layer_campaign.continuation_v2.gates import owned_runner as owner
    owner.D=HERE;owner.freeze=f;owner.check_sources()
    save(HERE/'execution_start.json',dict(status='starting',contract_sha256=sha(HERE/'contract.json'),source_freeze_sha256=sha(HERE/'source_freeze.json'),GPU_owner='extended64_protocol',production_APPLY_not_performed_by_runner=True))
    sessions={}
    try:
        for layout,e in c['plans'].items():
            check(live=True);plan=read(ROOT/e['path']);session=run(plan,startup_gc_fixed_model=True,timeout=7200)
            # host2 owns the only merged training log; host1 loss is not required.
            for host,node in [('host1',1),('host2',0)]:
                manifest=read(session/host/f'node{node}_manifest.json');assert manifest['status']=='completed' and read(session/host/'plan.json')==plan
            train=[json.loads(l)for l in(session/'host2/train.jsonl').read_text().splitlines()];assert len(train)==plan['steps']and train[-1]['iteration']==plan['steps']
            events_path=session/'host2/events.jsonl';events=[json.loads(l)for l in events_path.read_text().splitlines()];public=[r for r in events if r.get('event')in ('detection','profile_schedule','localization','quota_exhausted')]
            pred=HERE/f'{layout}_predictions_without_truth.json';assert not pred.exists()
            save(pred,dict(layout=layout,session_id=plan['session_id'],prediction_before_truth=True,events_sha256=sha(events_path),plan_sha256=sha(ROOT/e['path']),source_freeze_sha256=sha(HERE/'source_freeze.json'),events=public))
            sessions[layout]=dict(session_id=plan['session_id'],path=str(session.relative_to(ROOT)),prediction_sha256=sha(pred),plan_sha256=sha(ROOT/e['path']))
            save(HERE/'public_predictions_before_truth.json',dict(fixed_cases=c['fixed_cases'],prediction_before_truth=True,sessions=sessions))
        save(HERE/'execution_complete.json',dict(status='complete',fixed_cases=c['fixed_cases'],sessions=sessions,contract_sha256=sha(HERE/'contract.json'),source_freeze_sha256=sha(HERE/'source_freeze.json'),prediction_before_truth=True))
    except BaseException as error:
        save(HERE/'execution_failure.json',dict(error=repr(error),sessions=sessions,no_automatic_GPU_retry=True));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not (HERE/'contract.json').exists():
        if a.execute:p.error('waiting for submitted commit and one-shot frozen draw; execution unavailable')
        protocol=read(HERE/'protocol.json');assert protocol['seed'] is None and protocol['submitted_commit'] is None
        print(json.dumps(dict(status='CPU_draft_only_no_seed_no_GPU',case_count='Uniform{4,5,6} pending',protocol_sha256=sha(HERE/'protocol.json'))));return
    c,f=check(live=a.execute)
    if a.execute:execute(c,f)
    else:print(json.dumps(dict(status='CPU_staged_source_plan_check_passed_no_GPU',plans=c['plans'],cases=c['fixed_cases'],source_freeze_sha256=sha(HERE/'source_freeze.json'))))
if __name__=='__main__':main()
