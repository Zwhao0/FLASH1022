"""Read-only fixed two-run MONITOR aggregation; never launches or alters runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from results.layer_campaign.stage_forward_v1.sparse3_gc_draft.score_sparse3 import _pooled

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def load(entry,freeze,freeze_sha):
    session=ROOT/entry['session'];completion=ROOT/entry['completion'];snapshot=ROOT/entry['public_snapshot']
    completed=read(completion)
    require(completed['status']=='complete','runner completion missing')
    require(snapshot.is_file(),'sealed public snapshot missing')
    public=read(snapshot)
    public_events=public.get('public_events',public.get('events'))
    require(public.get('prediction_before_truth') is True and public_events==[] and public.get('session_id')==session.name,'public snapshot integrity/healthy events require review')
    require(completed.get('prediction_sha256')==digest(snapshot),'completion does not bind sealed public snapshot')
    require(public.get('source_freeze_sha256')==freeze_sha and completed.get('source_freeze_sha256')==freeze_sha,'snapshot/completion source freeze differs')
    require(public.get('plan_sha256')==digest(session/'plan.json') and completed.get('plan_sha256')==digest(session/'plan.json'),'snapshot/completion plan differs')
    train_path=session/'host2/train.jsonl';counts_path=session/'host2/cost_controller_counts.json';log_path=session/'host2/online.log';events_path=session/'host2/events.jsonl'
    if public.get('train_sha256') is not None or public.get('controller_counts_sha256') is not None:
        require(public.get('train_sha256')==digest(train_path) and public.get('controller_counts_sha256')==digest(counts_path),'sealed snapshot source differs')
        snapshot_source_mode='snapshot_hashes_verified'
    else:
        # The original PP recovery snapshot predates the hash-augmented schema.
        # Preserve it unchanged and verify its explicit absence cross-checks below.
        event_source=public.get('event_source',{})
        cross=event_source.get('absence_crosschecks',{})
        require(public.get('protocol')=='monitor_gc_268_infrastructure_recovery_public_v1' and event_source.get('exists') is False and event_source.get('path')==str(events_path.relative_to(ROOT)) and not events_path.exists(),'unhashed snapshot is not the known pre-hash recovery schema')
        require(log_path.is_file() and log_path.stat().st_size==0 and event_source.get('online_log_bytes')==0 and event_source.get('online_log_sha256')==digest(log_path),'unhashed recovery event absence does not match online log')
        snapshot_source_mode='prehash_recovery_snapshot_crosschecked'
    plan=read(session/'plan.json')
    require(plan.get('startup_gc_fixed_model') is True and plan['steps']==268,'incorrect actual plan')
    ranks={};sources=[];reference={}
    for host,node in [('host2',0),('host1',1)]:
        manifest=read(session/host/f'node{node}_manifest.json')
        require(manifest['status']=='completed' and manifest['plan']==plan,'node completion/plan mismatch')
        for name,sha in freeze['files'].items():
            if name.startswith('flash_v1/') and name.endswith(('.py','.cpp','.cu')):
                require(manifest['source_sha256'][name]==sha,'production source mismatch: '+name)
        sources.append(manifest['source_sha256'])
        ack=[json.loads(x)for x in(session/host/f'cost_actor_ack_history_node{node}.jsonl').read_text().splitlines()if x.strip()]
        require([(x['physical'],x['mode'])for x in ack]==[(1,'bare'),(77,'full'),(141,'full'),(205,'bare'),(269,'done')] and all(x['status']=='ready'for x in ack),'actor boundaries invalid')
        for path in (session/host).glob('cost_replay_rank*.json'):
            rank=int(path.stem.rsplit('rank',1)[1]);require(rank not in ranks,'duplicate rank')
            record=read(path);rows=record['rows'];blocks=record['blocks']
            require(record['status']=='complete' and len(rows)==268 and len(blocks)==4,'incomplete replay')
            by={x['physical']:x for x in rows};require(set(by)==set(range(1,269)),'physical coverage')
            require(all(x['actual_profile_epoch']==0 for x in rows),'unexpected PROFILE')
            for row in rows:
                physical=row['physical'];block=0 if physical<=12 else(physical-13)//64+1
                logical=physical if not block else(physical-13)%64+1
                mode='A'if block in(0,1,4)else'B'
                require(row['block']==block and row['logical']==logical and row['mode']==mode,'ABBA position mismatch')
                require(row['actual_online']==(mode=='B') and row['collection_context_bypassed']==(mode=='A') and row['collector_references_present']==(mode=='B'),'actual bare/full path mismatch')
            for end in(76,140,204,268):
                window=by[end]['window_end']
                require(window['gpu_fence_before_end_timestamp'] and window['main_thread_drain_serialize_write_in_window'],'window boundary timing contract')
                require(all(window[k]['role_rows']==0 and window[k]['aux_pending'] is False for k in('before_pending','before_normal_main_drain','after_pending')),'deferred collector work')
            keys=('parameter_sha256','buffer_sha256','optimizer_sha256','input_summary','private_rng_sha256','cpu_rng_sha256','cuda_rng_sha256','losses')
            for block in blocks:
                require(block['optimizer_step_calls']==64 and block['observed_lr_min']==block['observed_lr_max']==.001,'real SGD contract')
                require(block['initial_parameter_sha256']!=block['parameter_sha256'],'no SGD update')
                require(all(block[k]==blocks[0][k]for k in keys),'within-run replay differs')
            reference[rank]={k:blocks[0][k]for k in keys};ranks[rank]=by
    require(set(ranks)==set(range(16)),'16-rank population required')
    counts=read(session/'host2/cost_controller_counts.json')
    require(counts['monitor_sample_iterations']==list(range(77,205)) and counts['manual_requests_sent']==[],'real detector population/manual request mismatch')
    require(counts['natural_candidates']==0 and counts['ignored_child_candidates']==0,'healthy candidate requires explicit independent review, cannot silently aggregate')
    wall={i:max(ranks[r][i]['full_step_wall_ms']for r in ranks)for i in range(1,269)}
    strata=[]
    for order,a,b in [('AB',1,2),('BA',4,3)]:
        strata.append(dict(label=order,left=[wall[12+64*(a-1)+j]for j in range(33,65)],right=[wall[12+64*(b-1)+j]for j in range(33,65)]))
    if snapshot_source_mode=='prehash_recovery_snapshot_crosschecked':
        cross=public['event_source']['absence_crosschecks']
        require(cross.get('natural_candidates')==counts['natural_candidates'] and cross.get('ignored_child_candidates')==counts['ignored_child_candidates'] and cross.get('manual_requests_sent')==counts['manual_requests_sent'] and cross.get('profile_epoch_nonzero_steps')==sum(int(x['profile_epoch']!=0) for x in read_rows(train_path)),'recovery event absence cross-check differs from run data')
    return dict(session=entry['session'],completion_sha256=digest(completion),public_snapshot_sha256=digest(snapshot),snapshot_source_mode=snapshot_source_mode,strata=strata),plan,reference,sources

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    spec=read(args.inputs);require(len(spec['runs'])==2,'exactly original+one fixed repeat required')
    require(spec['runs'][0]['session']!=spec['runs'][1]['session'],'cannot duplicate one process as two runs')
    freeze_path=ROOT/spec['source_freeze'];freeze=read(freeze_path);freeze_sha=digest(freeze_path);loaded=[load(e,freeze,freeze_sha)for e in spec['runs']]
    a,b=loaded
    require(a[2]==b[2] and a[3]==b[3],'cross-run state/input/source differ')
    require({k:v for k,v in a[1].items()if k!='session_id'}=={k:v for k,v in b[1].items()if k!='session_id'},'repeat changed configuration')
    per_run=[dict(session=x[0]['session'],snapshot_source_mode=x[0]['snapshot_source_mode'],orders={s['label']:_pooled([s])for s in x[0]['strata']})for x in loaded]
    by_order={order:_pooled([x[0]['strata'][i]for x in loaded])for i,order in enumerate(['AB','BA'])}
    pooled=_pooled([s for x in loaded for s in x[0]['strata']])
    strict_budget=all(x['delta_percent']<1 and x['ci95_percent'][1]<1 for x in [*by_order.values(),pooled])
    average_budget=pooled['delta_percent']<1 and pooled['ci95_percent'][1]<1
    original=ROOT/spec['original_result']
    result=dict(layout=spec['layout'],fixed_runs=2,original_result=read(original),original_result_sha256=digest(original),per_run=per_run,combined_by_order=by_order,combined_all=pooled,average_budget_gate='within_budget'if average_budget else'inconclusive_or_above_budget',original_strict_per_order_gate='within_budget'if strict_budget else'inconclusive_or_above_budget',original_gate_unchanged=True,contract_sha256=digest(Path(__file__).with_name('contract.json')),aggregation_script_sha256=digest(Path(__file__)),no_further_adaptive_repeats=True)
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
