"""CPU-only v2 blind protocol. No GPU launch, localization read, or plan reroll.

Actual design generation requires a validated, hash-checked freeze manifest.
Keep every prevalidation attempt and all 32 formal cases, including failures.
"""
from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import statistics

LAYOUTS = ('legacy', 'node_local_pp')
MODES = ('attention_fwd_sleep', 'experts_bwd_sleep', 'attention_bwd_sleep',
         'attention_bwd_compute', 'whole_compute')
MIN_DEGRADATION_PCT = 30.0
MAX_PREVALIDATION_ROUNDS = 3
MAX_MULTIPLIER = 2.0
REQUIRED_SOURCES = ('flash_v1/localize_v2.py', 'flash_v1/localize_v2_linear.py', 'flash_v1/online_v2.py',
    'flash_v1/evidence.py', 'flash_v1/online.py', 'flash_v1/analysis.py',
    'flash_v1/whole_compute.py', 'flash_v1/collection.py', 'flash_v1/fault_workloads.py',
    'flash_v1/tensor_parallel.py', 'flash_v1/train_pipeline.py',
    'flash_v1/scripts/node_collection.py', 'flash_v1/scripts/v2_transport.py',
    'flash_v1/scripts/v2_blind_protocol.py')


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def sha256_json(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _positive(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('finite positive timing/dose required')
    return float(value)


@dataclass(frozen=True)
class VerifiedFreeze:
    manifest_path: Path
    root: Path
    manifest_sha256: str
    manifest: dict


def verify_freeze(manifest_path, root):
    path,root=Path(manifest_path).resolve(),Path(root).resolve()
    raw=path.read_bytes();manifest=json.loads(raw)
    if manifest.get('protocol')!='v2_blind32' or manifest.get('validated') is not True:
        raise ValueError('validated v2_blind32 freeze required')
    files=manifest.get('files',{})
    if not isinstance(files,dict) or not set(REQUIRED_SOURCES).issubset(files):
        raise ValueError('freeze is missing inference/observer/protocol sources')
    model_file=manifest.get('model_file')
    if not model_file or model_file not in files:raise ValueError('frozen model_file required')
    for name,digest in files.items():
        candidate=(root/name).resolve()
        if Path(name).is_absolute() or not candidate.is_relative_to(root):
            raise ValueError('freeze paths must remain under repository root')
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'freeze file hash mismatch: {name}')
    for layout in LAYOUTS:
        reference=manifest.get('whole_reference_sessions',{}).get(layout)
        catalogs=manifest.get('whole_catalog_files',{}).get(layout,[])
        if not isinstance(reference,str) or not reference or Path(reference).name!=reference:
            raise ValueError('both layouts require existing healthy whole-compute reference sessions')
        if not isinstance(catalogs,list) or not catalogs or any(name not in files for name in catalogs):
            raise ValueError('both layouts require hash-frozen whole catalog files or archives')
    from flash_v1.online_v2 import verify_model
    verify_model(json.loads((root/model_file).read_text()))
    return VerifiedFreeze(path,root,hashlib.sha256(raw).hexdigest(),manifest)


def _gate(gate):
    if not isinstance(gate,VerifiedFreeze):raise ValueError('verify freeze before building blind inputs')
    current=verify_freeze(gate.manifest_path,gate.root)
    if current.manifest_sha256!=gate.manifest_sha256:raise ValueError('freeze manifest changed')


def _coordinates(layout,rank):
    return (rank//8, rank%2, (rank//2)%2,
            rank//8 if layout=='legacy' else (rank//4)%2)


def _balanced(assignment,layout):
    for mode in MODES:
        ranks=[rank for rank,value in enumerate(assignment) if value==mode]
        for dim in range(4):
            sides=Counter(_coordinates(layout,rank)[dim] for rank in ranks)
            if abs(sides[0]-sides[1])>1:return False
    return True


def build_design(gate,seed=None):
    """Call only after final freeze; no outcomes are accepted as inputs."""
    _gate(gate)
    # Sampling happens strictly after the freeze gate, including automatic seed.
    if seed is None:seed=secrets.randbits(63)
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('seed must be a nonnegative 63-bit integer')
    rng=random.Random(seed);extras=list(MODES);rng.shuffle(extras)
    layouts={}
    for layout,extra in zip(LAYOUTS,extras):
        assignment=[mode for mode in MODES for _ in range(4 if mode==extra else 3)]
        for attempt in range(1,100001):
            rng.shuffle(assignment)
            if _balanced(assignment,layout):break
        else:raise RuntimeError(f'balanced assignment search exhausted for seed={seed} layout={layout}; retain failure, do not reroll seed')
        order=list(range(16));rng.shuffle(order)
        cases=[dict(case_id=f'{layout}_case{index:02d}',rank=rank,fault_mode=assignment[rank],
                    order=index,target_degradation_pct=round(rng.uniform(40,80),3))
               for index,rank in enumerate(order,1)]
        layouts[layout]=dict(cases=cases,mode_counts=dict(Counter(assignment)),assignment_attempts=attempt)
    result=dict(schema_version=1,protocol='v2_blind32',freeze_manifest_sha256=gate.manifest_sha256,
        seed=seed,total_cases=32,layouts=layouts,target_range_pct=[40,80],
        minimum_prevalidated_degradation_pct=MIN_DEGRADATION_PCT,
        prevalidation_policy=dict(max_rounds=3,minimum_degradation_pct=30,
            adjustment='only increase by min(2, max(1, 1.1*target_pct/measured_pct)); nonpositive measured pct stops automatic adjustment',
            measurement='elapsed wall timing only; healthy 5 then fault 8, last 5 fault steps',
            all_attempts_retained=True),
        formal_policy='all 32 original cases retained; primary detector->PROFILE->localizer online; no replacement/reordering/dose changes after formal start')
    validate_design(result,gate)
    return result


def validate_design(design,gate):
    _gate(gate)
    if design.get('protocol')!='v2_blind32' or design.get('freeze_manifest_sha256')!=gate.manifest_sha256 or design.get('total_cases')!=32:
        raise ValueError('design/freeze mismatch')
    if set(design.get('layouts',{}))!=set(LAYOUTS):raise ValueError('both layouts required')
    extras=[]
    for layout in LAYOUTS:
        cases=design['layouts'][layout]['cases']
        if len(cases)!=16 or sorted(case['rank'] for case in cases)!=list(range(16)):
            raise ValueError('each layout must target every GPU exactly once')
        if [case['order'] for case in cases]!=list(range(1,17)):raise ValueError('case order changed')
        counts=Counter(case['fault_mode'] for case in cases)
        if set(counts)!=set(MODES) or sorted(counts.values())!=[3,3,3,3,4]:
            raise ValueError('each layout requires 3/4 cases of every mode')
        assignment=[None]*16
        for case in cases:assignment[case['rank']]=case['fault_mode']
        if not _balanced(assignment,layout):raise ValueError('topology coordinate imbalance')
        if len({case['case_id'] for case in cases})!=16:raise ValueError('duplicate case identity')
        extras.append(next(mode for mode,count in counts.items() if count==4))
        if any(not 40<=_positive(case['target_degradation_pct'])<=80 for case in cases):raise ValueError('target outside frozen range')
    if len(set(extras))!=2:raise ValueError('global mode counts must be 6 or 7')


def initial_doses(design,whole_references,base_delay_ms=450.):
    """Whole refs are pre-existing elapsed calibration, not localization results.

    whole_references[layout][str(rank)] has slowdown_extra_ratio and
    actual_degradation_pct from the unchanged existing healthy-catalog setup.
    """
    doses={}
    for layout in LAYOUTS:
        doses[layout]={}
        for case in design['layouts'][layout]['cases']:
            target=case['target_degradation_pct']
            if case['fault_mode']=='whole_compute':
                reference=whole_references[layout][str(case['rank'])]
                value=_positive(reference['slowdown_extra_ratio'])*target/_positive(reference['actual_degradation_pct'])
            else:value=_positive(base_delay_ms)*target/60.
            doses[layout][case['case_id']]=round(value,6)
    return doses


def _interval(name,start,length,**fields):
    row=dict(id=name,start=start,end=start+length-1,record_start=start,
             target_ranks=[],delay_ms=0.,slowdown_extra_ratio=0.,fault_mode=None,segment='healthy')
    row.update(fields);return row


def _base(gate,design,layout,session,intervals,online):
    validate_design(design,gate)
    if layout not in LAYOUTS:raise ValueError('invalid layout')
    plan=dict(schema_version=1,protocol='v2_blind32',session_id=session,parallel_layout=layout,
        layout=layout,seed=design['seed'],steps=intervals[-1]['end'],seq_len=16,microbatches=2,
        warmup_steps=12,online=online,probe_enabled=False,profile_enabled=False,
        evidence_enabled=online,measurement_only=not online,max_profiles=32 if online else 0,
        role='blind_test' if online else 'strength_prevalidation',
        fault_kind='mixed_compute_simulation',design_sha256=sha256_json(design),
        freeze_manifest_sha256=gate.manifest_sha256,
        whole_compute_reference_session=gate.manifest['whole_reference_sessions'][layout],
        intervals=intervals)
    if online:plan['profile_backend']='semantic_only'
    return plan


def _fault(case,start,length,dose,baseline,record_skip=0):
    whole=case['fault_mode']=='whole_compute';dose=_positive(dose)
    return _interval(case['case_id'],start,length,record_start=start+record_skip,
        target_ranks=[case['rank']],rank=case['rank'],fault_mode=case['fault_mode'],segment='fault',
        case_id=case['case_id'],delay_ms=1. if whole else dose,
        slowdown_extra_ratio=dose if whole else 0.,dose=dose,
        target_degradation_pct=case['target_degradation_pct'],baseline_interval_id=baseline)


def build_prevalidation(gate,design,layout,session,doses,round_index):
    if type(round_index) is not int or not 1<=round_index<=3:raise ValueError('prevalidation is limited to three rounds')
    rows=[_interval('warmup',1,12,segment='warmup',warmup=True)]
    for case in design['layouts'][layout]['cases']:
        healthy=_interval('healthy_before_'+case['case_id'],rows[-1]['end']+1,5)
        rows.append(healthy)
        rows.append(_fault(case,healthy['end']+1,8,doses[case['case_id']],healthy['id'],record_skip=3))
    rows.append(_interval('healthy_final',rows[-1]['end']+1,5))
    plan=_base(gate,design,layout,session,rows,False)
    plan.update(round_index=round_index,measurement_source='host2/train.jsonl wall_iteration_ms only')
    return plan


def measure_prevalidation(plan,timing_rows):
    """No localizer, predictions, PROFILE, or injection log is read."""
    if any(plan.get(key) is not False for key in ('online','probe_enabled','profile_enabled','evidence_enabled')):
        raise ValueError('prevalidation must disable online/probe/profile/evidence')
    if plan.get('measurement_only') is not True or plan.get('role')!='strength_prevalidation':raise ValueError('wrong measurement plan')
    timing={}
    for row in timing_rows:
        if any(key in row for key in ('predicted','localization','rank_scores','rank_scores_us')):
            raise ValueError('localization results forbidden in strength prevalidation')
        if row.get('profile_epoch',0):raise ValueError('PROFILE activity forbidden during elapsed-only prevalidation')
        step=row.get('iteration',row.get('step'))
        if type(step) is not int or step in timing:raise ValueError('invalid/duplicate timing step')
        timing[step]=_positive(row['wall_iteration_ms'])
    if set(timing)!=set(range(1,plan['steps']+1)):raise ValueError('retain complete timing log for every planned iteration')
    intervals={row['id']:row for row in plan['intervals']};records={}
    for fault in plan['intervals']:
        if fault['segment']!='fault':continue
        healthy=intervals[fault['baseline_interval_id']]
        samples={}
        for name,segment in [('healthy',healthy),('fault',fault)]:
            steps=list(range(segment['record_start'],segment['end']+1))
            if len(steps)!=5:raise ValueError('exactly five elapsed samples required per phase')
            values=[timing[step] for step in steps]
            samples[name+'_iterations']=steps;samples[name+'_elapsed_ms']=values
            samples[name+'_mean_ms']=statistics.fmean(values)
        degradation=100*(samples['fault_mean_ms']/samples['healthy_mean_ms']-1)
        records[fault['case_id']]=dict(case_id=fault['case_id'],rank=fault['rank'],fault_mode=fault['fault_mode'],
            target_degradation_pct=fault['target_degradation_pct'],dose=fault['dose'],
            measured_degradation_pct=degradation,passed=degradation>=30,**samples)
    if len(records)!=16:raise ValueError('all 16 fixed cases must be measured')
    all_elapsed_rows=[dict(iteration=step,wall_iteration_ms=value) for step,value in sorted(timing.items())]
    return dict(protocol='v2_blind32_strength',layout=plan['layout'],round_index=plan['round_index'],
        all_elapsed_rows=all_elapsed_rows,timing_rows_sha256=sha256_json(all_elapsed_rows),
        design_sha256=plan['design_sha256'],plan_sha256=sha256_json(plan),
        all_passed=all(record['passed'] for record in records.values()),by_case=records)


def next_doses(measurement):
    if measurement['round_index']>=3 and not measurement['all_passed']:
        raise ValueError('three rounds exhausted; retain failures, do not replace cases')
    doses={}
    for key,row in measurement['by_case'].items():
        observed=row['measured_degradation_pct'];current=_positive(row['dose'])
        if row['passed']:doses[key]=current;continue
        if not math.isfinite(observed) or observed<=0:
            raise ValueError('nonpositive signal: stop automatic dosing and diagnose elapsed evidence')
        multiplier=min(2.,max(1.,1.1*row['target_degradation_pct']/observed))
        doses[key]=round(current*multiplier,6)
    return doses


def build_formal(gate,design,layout,session,measurement,attempt_history):
    validate_design(design,gate)
    if measurement.get('layout')!=layout or measurement.get('design_sha256')!=sha256_json(design) or not measurement.get('all_passed'):
        raise ValueError('formal requires passed strength confirmation of this exact design/layout')
    if not attempt_history or attempt_history[-1]!=measurement:raise ValueError('retain every prevalidation record including final round')
    cases=design['layouts'][layout]['cases'];validate_attempt_history(cases,attempt_history)
    rows=[_interval('healthy_baseline',1,45,baseline=True)]
    for case in cases:
        verified=measurement['by_case'][case['case_id']]
        if verified['measured_degradation_pct']<30:raise ValueError('case below preregistered strength floor')
        fault=_fault(case,rows[-1]['end']+1,25,verified['dose'],rows[-1]['id'])
        fault['verified_degradation_pct']=verified['measured_degradation_pct'];rows.append(fault)
        rows.append(_interval('healthy_after_'+case['case_id'],fault['end']+1,35))
    rows.append(_interval('healthy_final',rows[-1]['end']+1,15))
    plan=_base(gate,design,layout,session,rows,True)
    if plan['steps']!=1020:raise AssertionError('formal schedule must be exactly 1020 steps')
    plan.update(evidence_model=gate.manifest['model_file'],detector_and_localizer_frozen=True,
        retain_all_cases=True,retain_all_false_alarms=True,
        prevalidation_history_sha256=[sha256_json(attempt) for attempt in attempt_history])
    return plan


def validate_attempt_history(cases,history):
    expected={case['case_id']:case for case in cases}
    if not 1<=len(history)<=3 or [attempt['round_index'] for attempt in history]!=list(range(1,len(history)+1)):
        raise ValueError('complete sequential attempt history required')
    for index,attempt in enumerate(history):
        if set(attempt['by_case'])!=set(expected):raise ValueError('cases dropped or replaced')
        for key,row in attempt['by_case'].items():
            case=expected[key]
            if row['rank']!=case['rank'] or row['fault_mode']!=case['fault_mode'] or row['target_degradation_pct']!=case['target_degradation_pct']:
                raise ValueError('case identity/mode/strength changed')
        if index:
            if history[index-1]['all_passed']:raise ValueError('stop prevalidation once all cases pass; no optional repeats')
            if attempt['design_sha256']!=history[0]['design_sha256'] or attempt['layout']!=history[0]['layout']:
                raise ValueError('attempt history changed layout/design')
            required=next_doses(history[index-1])
            if any(attempt['by_case'][key]['dose']!=required[key] for key in expected):
                raise ValueError('dose change outside preregistered elapsed-only rule')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    # Refuse an overwrite before sampling even a seed; do not silently reroll.
    generation_record=args.output.with_suffix(args.output.suffix+'.generation.json')
    if args.output.exists() or generation_record.exists():raise FileExistsError('blind design or generation attempt already exists; no replacement/reroll')
    gate=verify_freeze(args.freeze,args.root)
    seed=secrets.randbits(63) if args.seed is None else args.seed
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('seed must be a nonnegative 63-bit integer')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with generation_record.open('xb') as stream:
        stream.write(encoded(dict(seed=seed,freeze_manifest_sha256=gate.manifest_sha256,status='generation_started; retain on any subsequent failure')))
    design=build_design(gate,seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(encoded(design))
    print(json.dumps(dict(output=str(args.output),design_sha256=sha256_json(design),freeze_sha256=gate.manifest_sha256)))

if __name__=='__main__':main()
