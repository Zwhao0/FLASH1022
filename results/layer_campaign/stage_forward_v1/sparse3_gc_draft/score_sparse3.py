#!/usr/bin/env python3
"""Independent CPU audit for the frozen sparse3 BCCB profile-only run."""
from __future__ import annotations
import hashlib, json, math, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from results.layer_campaign.stage_forward_v1.sparse3_gc_draft.contract import position
from results.layer_campaign.stage_forward_v1.sparse3_gc_draft.sampling import contract as sampling_contract
from flash_v1.online_v2 import evidence_integrity_reason
from flash_v1.localize_v2_linear import infer as infer_core, verify_model as verify_core
from flash_v1.stage_forward_localize import validate_row
from flash_v1.stage_forward_peer import verify_topology, verify_model as verify_peer
from results.layer_campaign.stage_forward_v1.sparse3_gc_draft.sparse_peer import infer_bound as infer_sparse_peer

RANKS = set(range(16))
WINDOWS = {1: (61, 68), 2: (93, 100)}
BLOCKS = [(1, 'B'), (2, 'C'), (3, 'C'), (4, 'B')]
SAMPLE_OFFSETS = (1, 3, 7)
EXACT_BLOCK_KEYS = ('parameter_sha256', 'buffer_sha256', 'optimizer_sha256',
                    'input_summary', 'private_rng_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jread(path): return json.loads(Path(path).read_text())
def jlread(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def finite_positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
def put(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')

def _percentile(xs, q):
    xs = sorted(xs); at = (len(xs)-1)*q; lo = math.floor(at); hi = math.ceil(at)
    return xs[lo] if lo == hi else xs[lo]*(hi-at) + xs[hi]*(at-lo)

def _block_indices(n, length, rng):
    out=[]
    while len(out)<n:
        start=rng.randrange(n); out.extend((start+i)%n for i in range(length))
    return out[:n]

def _paired(left, right, label):
    delta=100*(math.fsum(right)/math.fsum(left)-1)
    rng=random.Random(20261024); boot=[]
    for _ in range(10000):
        ix=_block_indices(len(left), 2, rng)
        boot.append(100*(math.fsum(right[i] for i in ix)/math.fsum(left[i] for i in ix)-1))
    return dict(label=label, baseline_total_ms=math.fsum(left), profiled_total_ms=math.fsum(right),
        delta_percent=delta, paired_position_delta_percent=[100*(b/a-1) for a,b in zip(left,right)],
        ci95_percent=[_percentile(boot,.025),_percentile(boot,.975)],
        bootstrap=dict(method='circular paired logical-position block bootstrap',seed=20261024,
                       replicates=10000,block_length=2))

def _pooled(strata):
    left_total=math.fsum(math.fsum(x['left']) for x in strata)
    right_total=math.fsum(math.fsum(x['right']) for x in strata)
    rng=random.Random(20261024); boot=[]
    for _ in range(10000):
        den=num=0.0
        for item in strata:
            ix=_block_indices(len(item['left']),2,rng)
            den += math.fsum(item['left'][i] for i in ix)
            num += math.fsum(item['right'][i] for i in ix)
        boot.append(100*(num/den-1))
    return dict(baseline_total_ms=left_total,profiled_total_ms=right_total,
        delta_percent=100*(right_total/left_total-1),ci95_percent=[_percentile(boot,.025),_percentile(boot,.975)],
        method='ratio of summed max-16-rank walls; bootstrap resamples paired positions within each cold/warm stratum')

def validate_binding(layout, session, contract, freeze, completion, plan_path):
    errors=[]; plan=jread(plan_path); entry=contract['plans'][layout]
    if sha(plan_path)!=entry['sha256']: errors.append('contract plan sha mismatch')
    actual_plan=session/'plan.json'
    if not actual_plan.exists() or jread(actual_plan)!=plan: errors.append('session plan differs from frozen plan')
    elif sha(actual_plan)!=completion['sessions'][layout]['plan_sha256']: errors.append('completion plan sha mismatch')
    if plan.get('session_id')!=session.name or completion['sessions'][layout].get('session_id')!=session.name: errors.append('session id mismatch')
    if completion['sessions'][layout].get('status')!='complete': errors.append('session completion status mismatch')
    if completion['sessions'][layout].get('path')!=str(session.relative_to(ROOT)): errors.append('session path mismatch')
    if plan.get('steps')!=140 or plan.get('seed')!=1348533921 or plan.get('parallel_layout')!=layout: errors.append('plan dimensions mismatch')
    if plan.get('freeze_manifest_sha256')!=sha(HERE/'freeze.json'): errors.append('plan freeze binding mismatch')
    if plan.get('source_freeze_path')!=contract.get('source_freeze_path'): errors.append('plan source path mismatch')
    for host, node in [('host1',1),('host2',0)]:
        mpath=session/host/f'node{node}_manifest.json'
        if not mpath.exists(): errors.append(f'missing {host} runtime manifest'); continue
        manifest=jread(mpath)
        if manifest.get('status')!='completed' or manifest.get('session_id')!=session.name: errors.append(f'{host} manifest incomplete')
        if manifest.get('plan')!=plan: errors.append(f'{host} manifest plan mismatch')
        if manifest.get('stage_forward_aux_model_sha256') != sha(ROOT/plan['stage_forward_aux_model']): errors.append(f'{host} aux model file mismatch')
        if manifest.get('stage_forward_aux_context_binding_file_sha256') != sha(ROOT/plan['stage_forward_aux_context_binding']): errors.append(f'{host} aux binding file mismatch')
    if sha(HERE/'freeze.json') != contract['source_freeze_sha256'] or completion.get('source_freeze_sha256') != sha(HERE/'freeze.json'):
        errors.append('freeze sha binding mismatch')
    if completion.get('contract_sha256') != sha(HERE/'contract.json'): errors.append('completion contract sha mismatch')
    if freeze.get('collector_sha256') != '57c7cd2d49e231448354521d51eaeb12d078b7e242ddc22f34c088fd5766a204': errors.append('base collector is not original 57c')
    if freeze.get('base_sparse4_gc_freeze_sha256')!=sha(HERE.parent/'sparse4_gc_lifetime_v2/freeze.json'): errors.append('base sparse4 GC source lineage mismatch')
    for rel, expected in freeze['files'].items():
        try:
            if sha(ROOT/rel)!=expected: errors.append(f'local freeze file mismatch: {rel}')
        except OSError: errors.append(f'missing frozen file: {rel}')
    check=jread(HERE/'remote_source_check.json')
    for host in ('host1','host2'):
        if check.get(host,{}).get('passed') is not True or check[host].get('files')!=freeze['files']:
            errors.append(f'{host} remote source mismatch')
    model_path=ROOT/plan['evidence_model']; model=jread(model_path)
    if sha(model_path)!=plan.get('evidence_model_file_sha256') or model.get('sha256')!=plan.get('frozen_model_sha256'):
        errors.append('core file/semantic model binding mismatch')
    aux_path=ROOT/plan['stage_forward_aux_model']; binding_path=ROOT/plan['stage_forward_aux_context_binding']
    aux_model=jread(aux_path); binding=jread(binding_path)
    if sha(aux_path)!=plan.get('stage_forward_aux_model_file_sha256',sha(aux_path)): errors.append('auxiliary file sha mismatch')
    if aux_model.get('model_type')!='flash_stage_forward_tp_peer_v1' or binding.get('model_sha256')!=aux_model.get('sha256'):
        errors.append('auxiliary peer model binding mismatch')
    return dict(passed=not errors,errors=errors,plan_sha256=sha(plan_path),freeze_sha256=sha(HERE/'freeze.json'),
                core_model_file_sha256=sha(model_path),core_model_sha256=model.get('sha256'),
                aux_model_file_sha256=sha(aux_path),aux_model_sha256=aux_model.get('sha256'),
                aux_binding_sha256=sha(binding_path))

def freeze_public_snapshot(session, layout, destination):
    actual=session/'host2/events.jsonl'; snapshot=jread(destination)
    if not actual.exists(): raise ValueError(f'missing event stream {actual}')
    event_rows=jlread(actual)
    public=[r for r in event_rows if r.get('event') in ('detection','manual_profile','profile_schedule','localization','quota_exhausted')]
    if snapshot.get('session_id')!=session.name or snapshot.get('layout')!=layout or snapshot.get('source_sha256')!=sha(actual):
        raise ValueError('pretruth public snapshot source/session hash mismatch')
    if public!=snapshot.get('events'): raise ValueError('pretruth public snapshot events differ from actual event stream')
    return snapshot

def verify_replay(session):
    files={}
    for p in session.rglob('cost_replay_rank*.json'):
        rank=int(p.stem.rsplit('rank',1)[1])
        if rank in files: raise ValueError(f'duplicate replay rank {rank}')
        files[rank]=p
    if set(files)!=RANKS: raise ValueError(f'replay rank population mismatch {sorted(files)}')
    by_rank={}; checks={}
    for rank,path in sorted(files.items()):
        rec=jread(path)
        if rec.get('status')!='complete' or rec.get('no_input_generation_cache') is not True or rec.get('actual_lr')!=.001 or rec.get('parameters_replayed_exact') is not True:
            raise ValueError(f'rank {rank} replay record invalid')
        rows=rec.get('rows')
        if not isinstance(rows,list) or len(rows)!=140: raise ValueError(f'rank {rank} expected 140 actual timed rows')
        physical={}
        for row in rows:
            p=row.get('physical')
            if type(p) is not int or p in physical: raise ValueError(f'rank {rank} duplicate/invalid physical row')
            expected=position(p)
            for k in ('block','mode','logical','measured','profile'):
                if row.get(k)!=expected[k]: raise ValueError(f'rank {rank} p{p}: {k} disagrees with frozen position')
            if not finite_positive(row.get('full_step_wall_ms')): raise ValueError(f'rank {rank} p{p}: invalid full wall')
            epoch=1 if 61<=p<=68 else 2 if 93<=p<=100 else 0
            if row.get('actual_profile_epoch')!=epoch: raise ValueError(f'rank {rank} p{p}: actual profile epoch mismatch')
            if row.get('actual_online') is not True or row.get('collection_context_bypassed') is not False or row.get('collector_references_present') is not True:
                raise ValueError(f'rank {rank} p{p}: expected active same-process monitor')
            physical[p]=row
        if set(physical)!=set(range(1,141)): raise ValueError(f'rank {rank}: missing physical positions')
        blocks=rec.get('blocks')
        if not isinstance(blocks,list) or len(blocks)!=4: raise ValueError(f'rank {rank}: block summaries incomplete')
        for expected_id,(expected_block,expected_mode) in enumerate(BLOCKS,1):
            b=blocks[expected_id-1]
            if (b.get('block'),b.get('mode'))!=(expected_block,expected_mode): raise ValueError(f'rank {rank}: block order mismatch')
            if (b.get('optimizer_step_calls')!=32 or b.get('observed_lr_min')!=.001 or b.get('observed_lr_max')!=.001): raise ValueError(f'rank {rank}: true SGD/LR contract failed')
            if not b.get('initial_parameter_sha256') or b.get('initial_parameter_sha256')==b.get('parameter_sha256'): raise ValueError(f'rank {rank}: no parameter update')
            summary=b.get('input_summary')
            if not isinstance(summary,dict) or set(summary)!={str(i) for i in range(1,33)} or any(not x.get('content_sha256') or len(x.get('microbatches',[]))!=2 for x in summary.values()):
                raise ValueError(f'rank {rank}: actual inputs/microbatches not fully fingerprinted')
            losses=b.get('losses')
            if not isinstance(losses,list) or len(losses)!=32: raise ValueError(f'rank {rank}: loss trajectory incomplete')
            if rank==0 and any(not finite_positive(v) for v in losses): raise ValueError('rank 0 loss values invalid')
            if rank!=0 and any(v is not None for v in losses): raise ValueError(f'rank {rank}: unexpected non-owner loss series')
            if not b.get('collector_hooks_empty_and_APIs_original'): raise ValueError(f'rank {rank}: bare source behavior not original')
            ep=12+expected_id*32; fence=physical[ep].get('window_end')
            drained=(isinstance(fence,dict) and fence.get('gpu_fence_before_end_timestamp') is True and
              fence.get('main_thread_drain_serialize_write_in_window') is True and all(
                isinstance(fence.get(k),dict) and fence[k].get('role_rows')==0 and fence[k].get('aux_pending') is False
                for k in ('before_pending','before_normal_main_drain','after_pending')))
            if not drained: raise ValueError(f'rank {rank}: block-end GPU fence/drain failure')
        ref=blocks[0]
        exact=all(all(b.get(k)==ref.get(k) for k in EXACT_BLOCK_KEYS) for b in blocks[1:])
        losses_equal=(all(all(abs(a-b)<1e-4 for a,b in zip(ref['losses'],b['losses'])) for b in blocks[1:]) if rank==0 else True)
        if not exact or not losses_equal: raise ValueError(f'rank {rank}: block replay hashes/losses differ')
        by_rank[rank]=physical
        checks[rank]=dict(blocks=4,sgd_calls_per_block=32,lr=.001,exact_state_input_rng_replay=exact,
                          loss_trajectory_equal=losses_equal,positive_parameter_updates=True,block_end_fences=4)
    return by_rank,checks

def audit_windows(session,layout,plan,snapshot):
    # Read and revalidate core semantic records and auxiliary sparse rows.
    roles=[]; markers=[]; aux=[]; per_epoch={}; errors=[]
    for p in session.rglob('evidence_rank*.jsonl'):
        for row in jlread(p):
            if row.get('record_type')=='iteration_complete': markers.append(row)
            else: roles.append(row)
    for p in session.rglob('stage_forward_aux_rank*.jsonl'): aux.extend(jlread(p))
    for epoch,(start,end) in WINDOWS.items():
        sched=[r for r in snapshot['events'] if r.get('event')=='profile_schedule' and r.get('epoch')==epoch]
        if len(sched)!=1 or (sched[0].get('start'),sched[0].get('end'))!=(start,end): errors.append(f'epoch{epoch}:schedule mismatch')
        schedule=sched[0] if len(sched)==1 else dict(epoch=epoch,start=start,end=end)
        rr=[r for r in roles if r.get('epoch')==epoch]; mm=[r for r in markers if r.get('epoch')==epoch]
        reason=evidence_integrity_reason(rr,schedule,RANKS)
        expected={(step,rank) for step in range(start,end+1) for rank in RANKS}
        marker_ids={(r.get('iteration'),r.get('rank')) for r in mm}
        counts={}
        for r in rr: counts[(r.get('iteration'),r.get('rank'))]=counts.get((r.get('iteration'),r.get('rank')),0)+1
        core_ok=(reason is None and len(rr)==512 and len(mm)==128 and marker_ids==expected and
                 all(r.get('rows_count')==4 for r in mm) and all(counts.get(pair)==4 for pair in expected))
        if not core_ok: errors.append(f'epoch{epoch}:core rows/markers/integrity fail {reason}')
        ar=[r for r in aux if r.get('epoch')==epoch]
        expected_steps={start+i-1 for i in SAMPLE_OFFSETS}; expected_aux={(i,r) for i in expected_steps for r in RANKS}
        aux_ids={(r.get('iteration'),r.get('rank')) for r in ar}
        for r in ar: validate_row(r)
        sampling_ok=(len(ar)==48 and aux_ids==expected_aux and all(r.get('sampling_contract')==sampling_contract(epoch,start) for r in ar))
        if not sampling_ok: errors.append(f'epoch{epoch}:sparse auxiliary row coverage/contract mismatch')
        original_path=session/'host2'/f'online_profile_{epoch}'/'original_dc89.json'
        evidence_path=session/'host2'/f'online_profile_{epoch}'/'evidence.json'
        aux_path=session/'host2'/f'online_profile_{epoch}'/'stage_forward_aux.json'
        core_model_path=ROOT/plan['evidence_model']; core_model=jread(core_model_path); verify_core(core_model)
        window_evidence=jread(evidence_path)
        if len(window_evidence)!=512: raise ValueError(f'epoch{epoch}: frozen online core evidence missing')
        core_calc=infer_core(window_evidence,core_model,RANKS)
        original=jread(original_path)
        if any(original.get(k)!=core_calc.get(k) for k in ('status','rank','reason')):
            errors.append(f'epoch{epoch}: stored core diagnosis differs from independent CPU replay')
        aux_model_path=ROOT/plan['stage_forward_aux_model']; binding_path=ROOT/plan['stage_forward_aux_context_binding']
        aux_model=jread(aux_model_path); binding=jread(binding_path); verify_peer(aux_model)
        topology=jread(session/'host2'/'topology.json'); verify_topology(aux_model,topology,layout)
        peer_calc=infer_sparse_peer(ar,aux_model,sorted(RANKS),binding)
        stored_aux_rows=jread(aux_path)
        if not isinstance(stored_aux_rows,list) or len(stored_aux_rows)!=48:
            errors.append(f'epoch{epoch}: saved online sparse rows incomplete')
        else:
            saved_ids={(r.get('iteration'),r.get('rank')) for r in stored_aux_rows}
            if saved_ids!=expected_aux: errors.append(f'epoch{epoch}: saved online sparse rows differ in rank/step coverage')
            import hashlib
            saved_serialized=json.dumps(stored_aux_rows,sort_keys=True,separators=(',',':'))
            if hashlib.sha256(saved_serialized.encode()).hexdigest() != ((snapshot_result:=next((r.get('result') for r in snapshot['events'] if r.get('event')=='localization' and r.get('epoch')==epoch),{})).get('stage_forward_aux') or {}).get('evidence_sha256'):
                errors.append(f'epoch{epoch}: saved online sparse evidence hash mismatch')
        final_event=[r for r in snapshot['events'] if r.get('event')=='localization' and r.get('epoch')==epoch]
        if len(final_event)!=1: errors.append(f'epoch{epoch}:missing/duplicate final localization event')
        else:
            final=final_event[0]['result']; pub_aux=final.get('stage_forward_aux') or {}
            for k in ('status','rank','reason','candidate_pair','runner_up_pair'):
                if pub_aux.get(k)!=peer_calc.get(k):
                    errors.append(f'epoch{epoch}: peer verdict mismatch for {k}')
            if final.get('frozen_model_hash')!=core_model.get('sha256'): errors.append(f'epoch{epoch}: final core hash mismatch')
            if final.get('coverage')!={'completion_markers':128,'expected_markers':128,'evidence_rows':512}: errors.append(f'epoch{epoch}: public core coverage mismatch')
        peer_support=peer_calc.get('pair_support',{})
        known_pairs=sum(1 for d in peer_support.values() if d.get('matched_iterations',0)>=3)
        if known_pairs<2 or peer_calc.get('coverage',{}).get('comparable_pair_count',0)<2: errors.append(f'epoch{epoch}: fewer than two support3 comparable TP pairs')
        peer_loc=peer_calc.get('status')=='localized'
        core_loc=core_calc.get('status')=='localized'
        if core_loc or peer_loc: errors.append(f'epoch{epoch}: healthy localization false alarm')
        per_epoch[str(epoch)]=dict(start=start,end=end,core=dict(passed=core_ok,rows=len(rr),markers=len(mm),status=core_calc.get('status'),rank=core_calc.get('rank'),reason=core_calc.get('reason')),
            sparse_aux=dict(passed=sampling_ok,rows=len(ar),expected_rows=48,unique_identities=len(aux_ids),peer_status=peer_calc.get('status'),peer_rank=peer_calc.get('rank'),peer_reason=peer_calc.get('reason'),known_tp_pairs=known_pairs,comparable_tp_pairs=peer_calc.get('coverage',{}).get('comparable_pair_count'),tp_pair_support=peer_support),
            final_public=dict(predicted=final_event[0]['result'].get('predicted') if final_event else None,source=(final_event[0]['result'].get('source') or 'dc89_core_or_inconclusive') if final_event else None) )
    final_events=[r for r in snapshot['events'] if r.get('event') in ('detection','profile_schedule','localization')]
    event_ok=(len([r for r in final_events if r['event']=='detection'])==2 and
      [(r.get('iteration'),r.get('epoch')) for r in final_events if r['event']=='detection']==[(59,1),(91,2)] and
      {(r.get('epoch'),r.get('start'),r.get('end')) for r in final_events if r['event']=='profile_schedule'}=={(1,61,68),(2,93,100)} and
      len([r for r in final_events if r['event']=='localization'])==2)
    if not event_ok: errors.append('public event count/order/window mismatch')
    return dict(passed=not errors,errors=errors,windows=per_epoch,healthy_false_localizations=sum(
        1 for x in per_epoch.values() if x['core']['status']=='localized' or x['sparse_aux']['peer_status']=='localized'),
        public_events_sha256=sha(session/'host2/events.jsonl'))

def run_layout(layout):
    D=HERE; contract_path=D/'contract.json'; freeze_path=D/'freeze.json'; completion_path=D/f'execution_complete_{layout}.json'
    contract=jread(contract_path); freeze=jread(freeze_path); completion=jread(completion_path)
    if completion.get('status')!='complete': raise ValueError('formal measurement incomplete')
    if completion.get('contract_sha256')!=sha(contract_path) or completion.get('source_freeze_sha256')!=sha(freeze_path): raise ValueError('execution completion binding mismatch')
    session=ROOT/completion['sessions'][layout]['path']
    public_path=D/f'{layout}_public.json'
    if not public_path.exists():
        events_path=session/'host2/events.jsonl'
        events=[r for r in jlread(events_path)if r.get('event')in ('detection','manual_profile','profile_schedule','localization','quota_exhausted')]
        put(public_path,dict(session_id=session.name,layout=layout,source_sha256=sha(events_path),events=events,snapshot_before_truth_audit=True))
    snapshot=freeze_public_snapshot(session,layout,public_path)
    # Public predictions were snapshotted before this scorer opens cost/truth.
    entry=contract['plans'][layout]; plan_path=ROOT/entry['path']
    binding=validate_binding(layout,session,contract,freeze,completion,plan_path)
    if not binding['passed']: raise ValueError('source/plan binding failed: '+'; '.join(binding['errors']))
    rank_data,replay=verify_replay(session)
    evidence=audit_windows(session,layout,jread(plan_path),snapshot)
    walls={p:max(rank_data[r][p]['full_step_wall_ms'] for r in RANKS) for p in range(1,141)}
    phys=lambda block,logical: 13+32*(block-1)+logical-1
    logicals=range(17,25)
    cold_left=[walls[phys(1,l)] for l in logicals]; cold_right=[walls[phys(2,l)] for l in logicals]
    warm_left=[walls[phys(4,l)] for l in logicals]; warm_right=[walls[phys(3,l)] for l in logicals]
    cold=_paired(cold_left,cold_right,'B1->C2_cold')
    warm=_paired(warm_left,warm_right,'B4->C3_warm')
    pooled=_pooled([dict(left=cold_left,right=cold_right),dict(left=warm_left,right=warm_right)])
    report=dict(schema='sparse3_profile_only_independent_score_v1',layout=layout,
      session_id=session.name,execution_status='complete',execution_binding=binding,
      replay=dict(ranks=16,rows_per_rank=140,block_order='BCCB',all_rank_checks_passed=len(replay)==16,replay_checks={str(k):v for k,v in replay.items()}),
      healthy_evidence=evidence,
      profile_cost=dict(step_metric='max full_step_wall_ms across 16 global ranks; includes complete timed step',
        pairs=dict(cold=cold,warm=warm),pooled=pooled,profile_budget_percent=5.0,
        monitor_budget_claim=False,limits='Two healthy sparse-profile windows only; no full-BEAST MONITOR budget claim.'),
      source_scope='Original collector 57c and fusion 3eb4; isolated sparse3 sampling, no token/metadata-counter patch.')
    import importlib.util
    spec=importlib.util.spec_from_file_location('reuse_gc_lifetime_audit',HERE.parent/'sparse4_gc_lifetime_v2/score_gc_lifetime_legacy.py');gc_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(gc_audit)
    report['startup_gc_lifetime']=gc_audit.audit_startup_gc(session)
    output=D/f'{layout}_score.json'; put(output,report)
    return report

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--layout',choices=['legacy','node_local_pp']);args=parser.parse_args()
    layouts=[args.layout]if args.layout else ['legacy','node_local_pp']
    results={layout:run_layout(layout)for layout in layouts}
    summary=dict(schema='sparse3_GC140_independent_score_v1',status='complete'if len(results)==2 else 'single_layout_complete',layouts={k:dict(healthy_evidence_passed=v['healthy_evidence']['passed'],healthy_false_localizations=v['healthy_evidence']['healthy_false_localizations'],startup_gc_passed=v['startup_gc_lifetime']['passed'],replay=v['replay']['all_rank_checks_passed'],profile_cost=v['profile_cost'])for k,v in results.items()},fixed_masks=[1,3,7],core8_unchanged=True,monitor_budget_claim=False)
    put(HERE/('summary.json'if len(results)==2 else args.layout+'_summary.json'),summary);print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
