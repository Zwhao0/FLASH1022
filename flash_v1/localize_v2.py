"""Research-only local-duration attribution; no online imports or v1 changes.

Input is normalized evidence, never a fault plan. Duration evidence may be an
isolated compute operation or a module interval minus the union of its complete
same-stream communication intervals. The latter is *local stream residual*,
not kernel-only compute time. Healthy baselines pool semantic keys across ranks.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from statistics import median

SCHEMA_VERSION = 1
FORBIDDEN = {'target_rank','target_ranks','fault_rank','fault_mode','delay_ms','dose','case_id','plan_id','truth'}
MIN_SAMPLES = 6
MIN_ITERATIONS = 3
NOISE_LOG_FLOOR = math.log(1.15)


def _validate(row):
    if FORBIDDEN.intersection(row):
        raise ValueError('fault truth/plan fields must stay outside inference input')
    if type(row.get('rank')) is not int or type(row.get('iteration')) is not int:
        raise ValueError('integer rank/iteration required')


def semantic_key(row):
    """Pool equivalent layer/operator roles; layer number is diagnostic only.

    No physical rank, host, communicator ID, or injection identity. Shape must
    include parameter/activation dimensions so unlike operators cannot alias.
    """
    return json.dumps([row['measurement'], row.get('layer_type','decoder'), row['phase'], row['slot'], row.get('shape',[]), row.get('dtype')],sort_keys=True,separators=(',',':'))


def local_residual(module_start_us, module_end_us, communication_intervals):
    """Subtract complete same-stream interval union, clipping to module scope.

    These are local CUDA elapsed offsets; never cross-host timestamp values.
    Missing EP/other communication must mark coverage_complete=False upstream.
    """
    start,end=float(module_start_us),float(module_end_us)
    if not math.isfinite(start+end) or end<=start:raise ValueError('invalid module interval')
    intervals=[]
    for left,right in communication_intervals:
        left,right=float(left),float(right)
        if not math.isfinite(left+right) or right<left:raise ValueError('invalid communication interval')
        if left<start or right>end:raise ValueError('communication outside module scope')
        intervals.append((left,right))
    union=0.;tail=start
    for left,right in sorted(intervals):
        union+=max(0.,right-max(left,tail));tail=max(tail,right)
    return max(0.,end-start-union)


def _usable(row):
    _validate(row)
    duration=row.get('duration_us')
    return (row.get('coverage_complete') is True and row.get('measurement') in {'local_residual','isolated_compute'}
            and isinstance(duration,(int,float)) and not isinstance(duration,bool)
            and math.isfinite(duration) and duration>0)


def _fit_baselines(rows):
    grouped=defaultdict(list)
    for row in rows:
        if _usable(row):grouped[semantic_key(row)].append(math.log(row['duration_us']))
    result={}
    for key,values in grouped.items():
        if len(values)<MIN_SAMPLES:continue
        center=median(values)
        result[key]=dict(log_median=center,log_scale=max(NOISE_LOG_FLOOR,1.4826*median(abs(v-center) for v in values)),count=len(values))
    return result


def _interpolated_reference(row,model):
    """Bounded healthy token-count interpolation for the same expert operator.

    Interpolate median elapsed time in linear time/work space, never extrapolate.
    Other dimensions/dtype/semantic role must match; only healthy anchors fit it.
    """
    shape=row.get('shape',[])
    if (not model.get('allow_token_interpolation') or row.get('slot')!='experts'
            or len(shape)!=2 or type(shape[0]) is not int):return None
    request=json.loads(semantic_key(row))
    anchors=[]
    for key,reference in model['baselines'].items():
        signature=json.loads(key)
        if signature[:4]!=request[:4] or signature[5:]!=request[5:]:continue
        candidate=signature[4]
        if len(candidate)==2 and candidate[1:]==shape[1:]:
            anchors.append((candidate[0],reference))
    lower=[item for item in anchors if item[0]<shape[0]]
    upper=[item for item in anchors if item[0]>shape[0]]
    if not lower or not upper:return None
    left,low=max(lower,key=lambda item:item[0]);right,high=min(upper,key=lambda item:item[0])
    weight=(shape[0]-left)/(right-left)
    center=(1-weight)*math.exp(low['log_median'])+weight*math.exp(high['log_median'])
    return dict(log_median=math.log(center),log_scale=max(low['log_scale'],high['log_scale']),
                count=min(low['count'],high['count']),interpolated=True,anchor_tokens=[left,right])


def score(rows,model):
    """Only repeated comparable measurements contribute to rank scores."""
    values=defaultdict(list);iterations=defaultdict(set);reference_keys=defaultdict(set);diagnostics=defaultdict(int)
    observed=set()
    for row in rows:
        _validate(row);observed.add(row['rank'])
        if not _usable(row):diagnostics['incomplete_or_invalid']+=1;continue
        key=semantic_key(row)
        reference=model['baselines'].get(key)
        if reference is None:
            reference=_interpolated_reference(row,model)
            if reference is not None:diagnostics['interpolated_semantic_shape']+=1
        if reference is None:diagnostics['unknown_semantic_shape']+=1;continue
        family=json.dumps([row['measurement'],row.get('layer_type','decoder'),row['phase'],row['slot'],row.get('dtype')],separators=(',',':'))
        identity=(row['rank'],family)
        reference_keys[identity].add(key)
        values[identity].append((math.log(row['duration_us'])-reference['log_median'])/reference['log_scale'])
        iterations[identity].add(row['iteration'])
    ranks={rank:0. for rank in observed};supports=defaultdict(list)
    for (rank,key),z in values.items():
        if len(iterations[rank,key])<MIN_ITERATIONS:
            diagnostics['insufficient_repeat']+=1;continue
        value=median(z)
        ranks[rank]=max(ranks[rank],value)
        supports[rank].append(dict(key=key,reference_keys=sorted(reference_keys[rank,key]),z=value,iterations=len(iterations[rank,key])))
    return dict(rank_scores=ranks,support=dict(supports),diagnostics=dict(diagnostics))


def fit(healthy_fit_rows,healthy_calibration_windows,*,fit_sessions,calibration_sessions):
    """Fit and calibrate on distinct healthy sessions, no fault-trained choices."""
    healthy_fit_rows=list(healthy_fit_rows)
    expected_ranks={row['rank'] for row in healthy_fit_rows}
    fit_sessions=set(fit_sessions);calibration_sessions=set(calibration_sessions)
    if not fit_sessions or not calibration_sessions or fit_sessions & calibration_sessions:
        raise ValueError('nonempty disjoint healthy fit/calibration sessions required')
    model=dict(model_type='flash_local_v2',schema_version=SCHEMA_VERSION,allow_token_interpolation=True,world_size=len(expected_ranks),fit_sessions=sorted(fit_sessions),calibration_sessions=sorted(calibration_sessions),baselines=_fit_baselines(healthy_fit_rows),min_score=6.,margin=2.,
               metric='shape-standardized log-duration z; median per semantic phase/slot across >=3 iterations; maximum over roles',session_contract='explicit disjoint session identities; no within-session random split')
    if not model['baselines']:raise ValueError('no supported healthy semantic shapes')
    calibration=[score(window,model) for window in healthy_calibration_windows]
    if not calibration:raise ValueError('independent healthy calibration windows required')
    if any(set(result['rank_scores'])!=expected_ranks or set(result['support'])!=expected_ranks for result in calibration):
        raise ValueError('calibration lacks complete comparable rank evidence')
    max_healthy=max(max(result['rank_scores'].values(),default=0.) for result in calibration)
    model['min_score']=max(6.,max_healthy+2.)
    model['calibration_windows']=len(calibration)
    model['max_healthy_score']=max_healthy
    model['sha256']=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return model


def infer(rows,model,expected_ranks):
    result=score(rows,model)
    result.update(status='rejected',rank=None,reason='insufficient_or_ambiguous_evidence')
    if len(set(expected_ranks))!=model.get('world_size',len(set(expected_ranks))):
        result['reason']='model_topology_world_size_mismatch';return result
    if set(result['rank_scores'])!=set(expected_ranks):
        result['reason']='incomplete_rank_coverage';return result
    if set(result['support'])!=set(expected_ranks):
        result['reason']='incomplete_comparable_evidence';return result
    ordered=sorted(result['rank_scores'],key=result['rank_scores'].get,reverse=True)
    if len(ordered)<2:return result
    best,second=ordered[:2]
    if (result['rank_scores'][best]>model['min_score'] and
        result['rank_scores'][best]-result['rank_scores'][second]>model['margin']):
        result.update(status='localized',rank=best,reason='unique_local_duration_anomaly')
    return result


def verify_model(model):
    if model.get('model_type') != 'flash_local_v2' or model.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported v2 model')
    payload={key:value for key,value in model.items() if key!='sha256'}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if model.get('sha256')!=digest:raise ValueError('v2 model integrity mismatch')
    if not model.get('baselines') or model['min_score']<6 or model['margin']<2:
        raise ValueError('invalid v2 evidence thresholds')


def read_session_evidence(path):
    """Read measurements only; never opens a manifest, plan, or injection file."""
    from pathlib import Path
    rows=[]
    for file in sorted(Path(path).glob('host*/evidence_rank*.jsonl')):
        for line in file.read_text().splitlines():
            row=json.loads(line)
            if row.get('record_type','measurement')=='measurement':rows.append(row)
    if not rows:raise ValueError('session has no evidence measurements')
    return rows


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('fit')
    create.add_argument('--fit-session',type=Path,required=True)
    create.add_argument('--cal-session',type=Path,required=True)
    create.add_argument('--start',type=int,default=13);create.add_argument('--end',type=int,default=60)
    create.add_argument('--output',type=Path,required=True)
    predict=sub.add_parser('infer')
    predict.add_argument('--session',type=Path,required=True);predict.add_argument('--model',type=Path,required=True)
    predict.add_argument('--start',type=int,required=True);predict.add_argument('--end',type=int,required=True)
    predict.add_argument('--world-size',type=int,default=16);predict.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='fit':
        train=[row for row in read_session_evidence(args.fit_session) if args.start<=row['iteration']<=args.end]
        cal=read_session_evidence(args.cal_session)
        windows=[[row for row in cal if start<=row['iteration']<start+8] for start in range(args.start,args.end+1,8)]
        result=fit(train,windows,fit_sessions=[str(args.fit_session.resolve())],calibration_sessions=[str(args.cal_session.resolve())])
    else:
        model=json.loads(args.model.read_text());verify_model(model)
        rows=[row for row in read_session_evidence(args.session) if args.start<=row['iteration']<=args.end]
        result=infer(rows,model,range(args.world_size))
        result['window']=[args.start,args.end];result['model_sha256']=model['sha256']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),status=result.get('status','fitted'),rank=result.get('rank'),sha256=result.get('sha256'))))

if __name__=='__main__':main()
