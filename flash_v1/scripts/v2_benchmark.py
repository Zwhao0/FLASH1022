"""Predeclared matched-workload ABBA plans and CPU wall-time scoring."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from flash_v1.scripts.v2_collect import plan, interval, online_validation_plan

ORDER = ('A', 'B', 'B', 'A')


def benchmark_plan(session, layout, seed, mode, family, model=None, backend='native_semantic', *,
                   monitor_confirm_recent_three=True):
    if mode not in ('A', 'B') or family not in ('profile', 'profile_healthy', 'monitor', 'full_online'):
        raise ValueError('invalid benchmark mode/family')
    p = plan(session, layout, 'development', seed, 'healthy',
             monitor_confirm_recent_three=monitor_confirm_recent_three)
    p.update(protocol=f'v2_{family}_abba', benchmark_mode=mode,
             benchmark_work_lock=family.startswith('profile'),
             numerical_audit_steps=[], online=family == 'monitor' and mode == 'B',
             probe_enabled=mode == 'B', profile_enabled=family.startswith('profile') and mode == 'B',
             evidence_enabled=mode == 'B', max_profiles=3)
    if backend not in ('native_semantic', 'semantic_only'):
        raise ValueError('invalid PROFILE backend')
    if backend == 'semantic_only':
        p.update(profile_backend=backend, probe_enabled=False)
    if model:
        p['evidence_model'] = model
    if family == 'full_online':
        validation = online_validation_plan(session, layout, seed, model,
                                            monitor_confirm_recent_three=monitor_confirm_recent_three)
        p.update(intervals=validation['intervals'], online=mode=='B',
                 profile_enabled=False, benchmark_work_lock=True)
    if family == 'profile':
        smoke = plan(session, layout, 'development', seed, 'smoke',
                     monitor_confirm_recent_three=monitor_confirm_recent_three)
        faults = [r for r in smoke['intervals'] if r.get('target_ranks')]
        rows = [interval('warmup', 1, 12, record_start=13, warmup=True),
                interval('healthy_measurement', 13, 16)]
        for index, fault in enumerate(faults):
            copied = dict(fault, id=f'matched_fault_{index}', start=29 + index*16,
                          end=44 + index*16, record_start=29 + index*16)
            rows.append(copied)
        p['intervals'] = rows
        p['whole_compute_reference_session'] = smoke['whole_compute_reference_session']
    return p


def lock_workload(output, rank, delay, whole, reference_session=None):
    """Test-only cache: same injection units across ABBA, same old hook locations."""
    output = Path(output)
    reference_hash = None
    if reference_session:
        if Path(reference_session).name != reference_session:
            raise ValueError('benchmark reference must be a session basename')
        source = output.parent / reference_session / f'benchmark_work_rank{rank}.json'
        reference_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        reference = json.loads(source.read_text())['work']
        if set(reference['engines']) != set(delay.engines):
            raise ValueError('benchmark fault engines differ from reference')
        for name, engine in delay.engines.items():
            spec = reference['engine_specs'][name]
            if spec['engine'] != engine.engine or spec['blocks'] != getattr(engine,'blocks',None):
                raise ValueError('benchmark engine hardware geometry differs')
            prepared = {float(k): v for k,v in reference['engines'][name].items()}
            if set(prepared) != set(engine.prepared):
                raise ValueError('benchmark doses differ from reference')
            engine.prepared = prepared
            engine.units_per_ms = spec['units_per_ms']
        if whole is not None:
            saved = reference['whole']
            if saved['catalog_sha256'] != whole.catalog_sha256:
                raise ValueError('benchmark whole compute shape catalog differs')
            if saved['engine_spec']['blocks'] != whole.engine.blocks:
                raise ValueError('benchmark whole FMA hardware geometry differs')
            whole.quantum, whole.fma_model = saved['quantum'], saved['fma_model']
            whole.engine.units_per_ms = saved['engine_spec']['units_per_ms']
            whole.units_per_ms = whole.fma_model['units_per_ms']
            whole.intercept_ms = whole.fma_model['intercept_ms']
            (output/f'whole_compute_calibration_rank{rank}.json').write_text(
                json.dumps(whole.calibration_report(),indent=2)+'\n')
    work = dict(engines={name:{str(k):v for k,v in engine.prepared.items()}
                         for name,engine in delay.engines.items()},
                engine_specs={name:dict(engine=engine.engine,units_per_ms=engine.units_per_ms,
                                         blocks=getattr(engine,'blocks',None)) for name,engine in delay.engines.items()},
                whole=None if whole is None else dict(catalog_sha256=whole.catalog_sha256,
                     quantum=whole.quantum, fma_model=whole.fma_model,
                     engine_spec=dict(blocks=whole.engine.blocks,units_per_ms=whole.engine.units_per_ms)))
    digest = hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()
    (output/f'benchmark_work_rank{rank}.json').write_text(json.dumps(dict(rank=rank,
        reference_session=reference_session,reference_file_sha256=reference_hash,
        work_sha256=digest,work=work),indent=2)+'\n')
    return digest


def score_group(paths, output):
    """Report all strata and adjacent AB/BA pairs; never subtract injected work."""
    if len(paths) != 4:
        raise ValueError('ABBA requires four runs')
    records = []
    train_by_run, events_by_run = [], []
    numerical_losses = []
    source_hashes = []
    weight_hashes, comparable_plans = [], []
    physical_traces, cache_digests = [], []
    for path, mode in zip(map(Path, paths), ORDER):
        p = json.loads((path/'plan.json').read_text())
        if p.get('benchmark_mode') != mode:
            raise ValueError('order mismatch')
        train = [json.loads(x) for x in (path/'host2/train.jsonl').read_text().splitlines()]
        train_by_run.append({r['iteration']:r for r in train})
        if [r['iteration'] for r in train] != list(range(1,p['intervals'][-1]['end']+1)):
            raise ValueError('incomplete training sequence')
        manifests=[json.loads((path/h/f'node{i}_manifest.json').read_text())
                   for i,h in enumerate(('host2','host1'))]
        if any(m.get('status')!='completed' for m in manifests):
            raise ValueError('incomplete node')
        for key in ('configuration','source_sha256','model_sha256','probe_sha256','snapshot_sha256','evidence_model_sha256'):
            if manifests[0].get(key)!=manifests[1].get(key):
                raise ValueError(f'cross-node mismatch {key}')
        source_hashes.append(manifests[0]['source_sha256'])
        weight_hashes.append(manifests[0]['model_sha256'])
        comparable_plans.append({k:v for k,v in p.items() if k not in
            {'session_id','benchmark_mode','online','probe_enabled','profile_enabled','evidence_enabled',
             'benchmark_work_reference_session'}})
        numerical_losses.append([r['loss'] for r in train])
        sums=defaultdict(float);counts=defaultdict(int)
        for row in train:
            if row['iteration']<=12:continue
            segment=next(s for s in p['intervals'] if s['start']<=row['iteration']<=s['end'])
            stratum=segment.get('fault_mode') or ('healthy_with_idle_whole_injection_framework'
                if any(s.get('fault_mode')=='whole_compute' for s in p['intervals']) else 'healthy')
            sums[stratum]+=row['wall_iteration_ms'];counts[stratum]+=1
        events=path/'host2/events.jsonl'
        events=[json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
        events_by_run.append(events)
        injected=defaultdict(list)
        physical = {}
        caches = {}
        for host in ('host2','host1'):
            for f in (path/host).glob('benchmark_work_rank*.json'):
                saved=json.loads(f.read_text());caches[saved['rank']]=saved
        for host in ('host2','host1'):
            for pattern in ('injection_rank*.jsonl','whole_compute_truth_rank*.jsonl'):
                for f in (path/host).glob(pattern):
                    for text in f.read_text().splitlines():
                        truth=json.loads(text)
                        if truth.get('iteration',0)<=12 or 'calibrated_injected_ms' not in truth:
                            continue
                        segment=next(s for s in p['intervals'] if s['start']<=truth['iteration']<=s['end'])
                        injected[segment['fault_mode']].append(truth['calibrated_injected_ms'])
                        if 'buckets' in truth:
                            value={k:{field:v[field] for field in ('launch_rounds','injected_launches')}
                                   for k,v in truth['buckets'].items()}
                        else:
                            value=dict(calls=truth['hook_calls'],sleep_cycles=truth['injected_cycles'])
                            if truth.get('engine')=='cuda_fp32_fma' and truth['rank'] in caches:
                                prepared=caches[truth['rank']]['work']['engines']['cuda_fp32_fma']
                                if len(prepared)!=1:raise ValueError('ambiguous benchmark FMA dose')
                                value['fma_rounds']=next(iter(prepared.values()))['units']*truth['hook_calls']
                        physical[f"{truth['rank']}:{truth['iteration']}"]=value
        physical_traces.append(hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest())
        cache_digests.append({rank:value['work_sha256'] for rank,value in caches.items()})
        records.append(dict(session=p['session_id'],mode=mode,sums_ms=dict(sums),counts=dict(counts),
                            full_training_wall_ms=train[-1]['elapsed_walltime']*1000,
                            simulated_calibrated_gpu_work_ms={k:dict(mean=statistics.mean(v),sum=sum(v),steps=len(v)) for k,v in injected.items()},
                            detection_or_profile_events=sum(r.get('event') in ('detection','profile_schedule','localization') for r in events),
                            profile_steps=sum(r.get('profile_epoch',0)>0 for r in train)))
    strata=sorted(records[0]['sums_ms']);results={}
    for stratum in strata:
        a=sum(r['sums_ms'][stratum] for r in records if r['mode']=='A')
        b=sum(r['sums_ms'][stratum] for r in records if r['mode']=='B')
        pairs=[]
        for i in (0,2):
            first,second=records[i:i+2];av=first if first['mode']=='A' else second;bv=first if first['mode']=='B' else second
            pairs.append(100*(bv['sums_ms'][stratum]/av['sums_ms'][stratum]-1))
        results[stratum]=dict(baseline_sum_ms=a,enabled_sum_ms=b,overhead_pct=100*(b/a-1),adjacent_pair_overhead_pct=pairs)
        count_a=sum(r['counts'][stratum] for r in records if r['mode']=='A')
        count_b=sum(r['counts'][stratum] for r in records if r['mode']=='B')
        results[stratum].update(baseline_steps=count_a,enabled_steps=count_b,
                               incremental_ms_per_step=b/count_b-a/count_a)
    report=dict(protocol='same-seed model/SGD/iteration/token sequence ABBA; exclude only declared first12 warmup',
                records=records,strata=results,source_identity_across_runs=all(s==source_hashes[0] for s in source_hashes),
                weights_identical_across_runs=all(s==weight_hashes[0] for s in weight_hashes),
                matched_plans=all(p==comparable_plans[0] for p in comparable_plans),
                fixed_injection_work_caches=all(s==cache_digests[0] for s in cache_digests),
                actual_injection_trace_identical=all(s==physical_traces[0] for s in physical_traces),
                actual_injection_trace_sha256=physical_traces,
                max_loss_difference=max(abs(a-b) for losses in numerical_losses[1:] for a,b in zip(numerical_losses[0],losses)),
                simulation_cost_policy='matched injected work remains in both denominator and numerator; no subtraction or credit against monitoring cost',
                limitations=['Shared GPUs retain temporal drift; negative point estimates are not speedup claims.',
                             'PROFILE comparison includes native profiling plus v2 evidence versus no FLASH; MONITOR total is measured separately.',
                             'Calibrated injected GPU work is reported separately; complete injection host/framework overhead is not isolated or subtracted.',
                             'Forced offline PROFILE excludes online transport/inference CPU work; matched full-online windows must be checked separately.'])
    report['integrity_passed']=all(report[k] for k in ('source_identity_across_runs','weights_identical_across_runs',
        'matched_plans','fixed_injection_work_caches','actual_injection_trace_identical')) and report['max_loss_difference']<1e-4
    if 'monitor' in comparable_plans[0]['protocol']:
        report['integrity_passed'] &= not any(r['detection_or_profile_events'] or r['profile_steps'] for r in records)
    report['target_pct']=1 if 'monitor' in comparable_plans[0]['protocol'] else 5
    report['target_point_estimate_met']=report['integrity_passed'] and all(x['overhead_pct']<=report['target_pct'] for x in results.values())
    if 'full_online' in comparable_plans[0]['protocol']:
        from flash_v1.scripts.v2_report import score_online
        online_checks=[score_online(Path(paths[index])) for index in (1,2)]
        report['online_validation_passed']=all(r['integrity_passed'] and
            r['denominator']==2 and r['correct']==2 and r['healthy_detection_count']==0 for r in online_checks)
        full_a=sum(r['full_training_wall_ms'] for r in records if r['mode']=='A')
        full_b=sum(r['full_training_wall_ms'] for r in records if r['mode']=='B')
        report['full_training_wall_overhead_pct']=100*(full_b/full_a-1)
        windows=[]
        for bi,ai in ((1,0),(2,3)):
            schedules=[r for r in events_by_run[bi] if r['event']=='profile_schedule']
            for s in schedules:
                steps=list(range(s['start'],s['end']+1))
                a=sum(train_by_run[ai][step]['wall_iteration_ms'] for step in steps)
                b=sum(train_by_run[bi][step]['wall_iteration_ms'] for step in steps)
                localization=next((r for r in events_by_run[bi] if r['event']=='localization'
                                   and r.get('epoch')==s['epoch']),None)
                windows.append(dict(enabled_session=records[bi]['session'],baseline_session=records[ai]['session'],
                    start=s['start'],end=s['end'],steps=len(steps),baseline_sum_ms=a,enabled_sum_ms=b,
                    post_window_result_latency_ms=None if localization is None else
                        1000*(localization['wall_time']-train_by_run[bi][s['end']]['walltime']),
                    overhead_pct=100*(b/a-1)))
        a=sum(w['baseline_sum_ms'] for w in windows);b=sum(w['enabled_sum_ms'] for w in windows)
        report['matched_full_online_profile_windows']=windows
        report['matched_full_online_profile_overhead_pct']=100*(b/a-1) if a else None
        report['integrity_passed'] &= (report['online_validation_passed'] and len(windows)==4
                                       and all(w['steps']==8 for w in windows))
        report['target_point_estimate_met'] &= (report['integrity_passed'] and a>0 and 100*(b/a-1)<=5
                                                and report['full_training_wall_overhead_pct']<=5)
        report['limitations']=[x for x in report['limitations'] if not x.startswith('Forced offline')]
        report['limitations'].append('Matched eight-step timing excludes any post-window inference tail; full-training wall includes that concurrent CPU path. Tail latency is reported separately and is not claimed as compute cost.')
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report
