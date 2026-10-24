"""CPU aggregation of every predeclared PP run/order; does not launch or tune."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BASE=HERE.parent/'sparse3_gc_draft';sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
score=load('fixed_sparse3_cost_math',BASE/'score_sparse3.py')
def bootstrap_ratio_of_sums(strata):return score._pooled(strata)
def check():
    c=read(HERE/'contract.json');ready=read(HERE/'ready.json')
    assert ready['contract_sha256']==sha(HERE/'contract.json') and ready['aggregate_sha256']==sha(__file__)
    assert sha(BASE/'freeze.json')==c['original_freeze_sha256'] and sha(BASE/'score_sparse3.py')==c['scorer_sha256']
    original=read(BASE/'node_local_pp_plan.json')
    for e in c['runs']:
        p=ROOT/e['plan_path'];assert sha(p)==e['plan_sha256'];plan=read(p)
        assert {k for k in set(plan)|set(original)if plan.get(k)!=original.get(k)} <= {'session_id'}
        assert plan['stage_forward_aux_sampling']['expected_aux_rows']==48
    assert len(c['strata'])==6 and c['max_additional_sessions']==2
    return c

def aggregate():
    c=check();all_strata=[];per_run={};source_files={}
    gc_helper=load('reuse_startup_gc_audit',HERE.parent/'sparse4_gc_lifetime_v2/score_gc_lifetime_legacy.py')
    # Seal ALL public predictions before opening any replay/evidence/GC rows.
    snapshots={}
    for e in c['runs']:
        session=ROOT/'results/data16_sessions'/e['session_id'];events_path=session/'host2/events.jsonl'
        events=[r for r in score.jlread(events_path)if r.get('event')in ('detection','manual_profile','profile_schedule','localization','quota_exhausted')]
        snapshot=dict(layout='node_local_pp',session_id=e['session_id'],source_sha256=sha(events_path),events=events,snapshot_before_truth_audit=True)
        out=HERE/f'run{e["run"]}_public.json'
        if out.exists():assert read(out)==snapshot
        else:score.put(out,snapshot)
        snapshots[e['run']]=snapshot
    reference_blocks=None
    for e in c['runs']:
        session=ROOT/'results/data16_sessions'/e['session_id'];plan=read(ROOT/e['plan_path'])
        assert read(session/'plan.json')==plan
        frozen=read(BASE/'freeze.json')
        for host,node in [('host1',1),('host2',0)]:
            m=read(session/host/f'node{node}_manifest.json');assert m['status']=='completed' and m['plan']==plan
            for name,digest in frozen['files'].items():
                if name.startswith('flash_v1/') and name.endswith(('.py','.cpp','.cu')):
                    assert m['source_sha256'].get(name)==digest,('source',e['run'],host,name)
        ranks,replay=score.verify_replay(session)
        # Cross-process same seed/initialization/input/state is checked, not inferred.
        blocks={}
        for p in session.glob('host*/cost_replay_rank*.json'):
            row=read(p);rank=int(p.stem.split('rank')[-1]);blocks[rank]=row['blocks']
        if reference_blocks is None:reference_blocks=blocks
        else:
            for rank in range(16):
                assert blocks[rank][0]['initial_parameter_sha256']==reference_blocks[rank][0]['initial_parameter_sha256']
                for key in score.EXACT_BLOCK_KEYS:
                    assert blocks[rank][0][key]==reference_blocks[rank][0][key],('process_replay',e['run'],rank,key)
                if rank==0:assert max(abs(a-b)for a,b in zip(blocks[rank][0]['losses'],reference_blocks[rank][0]['losses']))<1e-4
        health=score.audit_windows(session,'node_local_pp',plan,snapshots[e['run']]);assert health['passed'] and health['healthy_false_localizations']==0
        gc=gc_helper.audit_startup_gc(session);assert gc['passed']
        walls={i:max(ranks[r][i]['full_step_wall_ms']for r in range(16))for i in range(1,141)}
        strata=[];orders={}
        for order,b,cc in [('cold',1,2),('warm',4,3)]:
            phys=lambda block,l:13+32*(block-1)+l-1
            item=dict(left=[walls[phys(b,l)]for l in range(17,25)],right=[walls[phys(cc,l)]for l in range(17,25)],run=e['run'],order=order)
            all_strata.append(item);strata.append(item);orders[order]=score._paired(item['left'],item['right'],f'run{e["run"]}_{order}')
        per_run[str(e['run'])]=dict(session=e['session_id'],orders=orders,pooled=bootstrap_ratio_of_sums(strata),health=health,startup_gc_passed=True,replay_ranks=len(replay),plan_sha256=sha(ROOT/e['plan_path']))
    pooled=bootstrap_ratio_of_sums(all_strata);orders={o:bootstrap_ratio_of_sums([s for s in all_strata if s['order']==o])for o in ('cold','warm')}
    report=dict(schema=c['protocol'],complete_runs=3,complete_C_windows=6,all_fixed_positions_included=True,per_run=per_run,by_order=orders,pooled=pooled,pooled_ci_upper_below5=pooled['ci95_percent'][1]<5,all_run_order_points_below5=all(v['delta_percent']<5 for r in per_run.values()for v in r['orders'].values()),source_scope=c['source_and_plan_rule'],interpretation=c['interpretation'],contract_sha256=sha(HERE/'contract.json'),process_repeats_not_independent_blind_accuracy=True)
    score.put(HERE/'aggregate.json',report);print(json.dumps({k:v for k,v in report.items()if k not in ('per_run',)},sort_keys=True));return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        check();result=bootstrap_ratio_of_sums([dict(left=[100.]*8,right=[110.]*8)for _ in range(6)]);assert abs(result['delta_percent']-10)<1e-8 and all(abs(x-10)<1e-8 for x in result['ci95_percent']);print('CPU contract/only-session differences/all6strata/bootstrap equivalence passed;noGPU')
    else:aggregate()
if __name__=='__main__':main()
