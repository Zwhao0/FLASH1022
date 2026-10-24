"""Additive local-duration candidate; preserves the previous log model/module.

Only healthy sessions fit semantic workload baselines and window tail limits.
A duration is standardized in microseconds, so a six-scale limit represents an
additive increment. No fault mode, rank template, or dose is a model feature.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from flash_v1 import localize_v2 as common

MODEL_TYPE='flash_local_v2_linear'
RELATIVE_NOISE_FLOOR=.15
ABSOLUTE_NOISE_FLOOR_US=1.


def _baselines(rows):
    grouped=defaultdict(list)
    for row in rows:
        if common._usable(row):grouped[common.semantic_key(row)].append(row['duration_us'])
    result={}
    for key,values in grouped.items():
        if len(values)<common.MIN_SAMPLES:continue
        center=median(values)
        result[key]=dict(median_us=center,scale_us=max(ABSOLUTE_NOISE_FLOOR_US,RELATIVE_NOISE_FLOOR*center,
                         1.4826*median(abs(value-center) for value in values)),count=len(values))
    return result


def _interpolated(row,model):
    shape=row.get('shape',[])
    if (not model.get('allow_token_interpolation') or row.get('slot')!='experts'
            or len(shape)!=2 or type(shape[0]) is not int):return None
    request=json.loads(common.semantic_key(row));anchors=[]
    for key,reference in model['baselines'].items():
        signature=json.loads(key)
        if signature[:4]!=request[:4] or signature[5:]!=request[5:]:continue
        candidate=signature[4]
        if len(candidate)==2 and candidate[1:]==shape[1:]:anchors.append((candidate[0],reference))
    lower=[item for item in anchors if item[0]<shape[0]];upper=[item for item in anchors if item[0]>shape[0]]
    if not lower or not upper:return None
    left,low=max(lower,key=lambda item:item[0]);right,high=min(upper,key=lambda item:item[0])
    weight=(shape[0]-left)/(right-left)
    return dict(median_us=(1-weight)*low['median_us']+weight*high['median_us'],
                scale_us=max(low['scale_us'],high['scale_us']),count=min(low['count'],high['count']),
                interpolated=True,anchor_tokens=[left,right])


def score(rows,model):
    values=defaultdict(list);iterations=defaultdict(set);reference_keys=defaultdict(set)
    diagnostics=defaultdict(int);observed=set()
    for row in rows:
        common._validate(row);observed.add(row['rank'])
        if not common._usable(row):diagnostics['incomplete_or_invalid']+=1;continue
        key=common.semantic_key(row);reference=model['baselines'].get(key)
        if reference is None:
            reference=_interpolated(row,model)
            if reference is not None:diagnostics['interpolated_semantic_shape']+=1
        if reference is None:diagnostics['unknown_semantic_shape']+=1;continue
        family=json.dumps([row['measurement'],row.get('layer_type','decoder'),row['phase'],row['slot'],row.get('dtype')],separators=(',',':'))
        identity=(row['rank'],family);reference_keys[identity].add(key)
        values[identity].append((row['duration_us']-reference['median_us'])/reference['scale_us'])
        iterations[identity].add(row['iteration'])
    ranks={rank:0. for rank in observed};supports=defaultdict(list)
    for (rank,key),z in values.items():
        if len(iterations[rank,key])<common.MIN_ITERATIONS:
            diagnostics['insufficient_repeat']+=1;continue
        value=median(z);ranks[rank]=max(ranks[rank],value)
        supports[rank].append(dict(key=key,reference_keys=sorted(reference_keys[rank,key]),z=value,iterations=len(iterations[rank,key])))
    return dict(rank_scores=ranks,support=dict(supports),diagnostics=dict(diagnostics))


def fit(healthy_fit_rows,healthy_calibration_windows,*,fit_sessions,calibration_sessions):
    rows=list(healthy_fit_rows);expected={row['rank'] for row in rows}
    fit_sessions=set(fit_sessions);calibration_sessions=set(calibration_sessions)
    if not fit_sessions or not calibration_sessions or fit_sessions & calibration_sessions:
        raise ValueError('nonempty disjoint healthy fit/calibration sessions required')
    model=dict(model_type=MODEL_TYPE,schema_version=1,world_size=len(expected),allow_token_interpolation=True,
        fit_sessions=sorted(fit_sessions),calibration_sessions=sorted(calibration_sessions),
        baselines=_baselines(rows),min_score=6.,margin=2.,
        noise_rule=dict(relative_floor=.15,absolute_floor_us=1.,mad_multiplier=1.4826),
        metric='shape-standardized additive duration z; median per phase/slot across >=3 iterations; maximum over roles')
    if not model['baselines']:raise ValueError('no supported healthy semantic shapes')
    calibration=[score(window,model) for window in healthy_calibration_windows]
    if not calibration or any(set(result['rank_scores'])!=expected or set(result['support'])!=expected for result in calibration):
        raise ValueError('independent calibration needs complete comparable rank evidence')
    maximum=max(max(result['rank_scores'].values()) for result in calibration)
    model.update(min_score=max(6.,maximum+2.),max_healthy_score=maximum,calibration_windows=len(calibration))
    model['sha256']=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return model


def verify_model(model):
    if model.get('model_type')!=MODEL_TYPE or model.get('schema_version')!=1:raise ValueError('unsupported additive model')
    payload={key:value for key,value in model.items() if key!='sha256'}
    if model.get('sha256')!=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest():
        raise ValueError('additive model integrity mismatch')
    if not model.get('baselines') or model['min_score']<6 or model['margin']<2:raise ValueError('invalid additive evidence thresholds')


def infer(rows,model,expected_ranks):
    expected=set(expected_ranks);result=score(rows,model)
    result.update(status='rejected',rank=None,reason='insufficient_or_ambiguous_evidence')
    if len(expected)!=model['world_size']:result['reason']='model_topology_world_size_mismatch';return result
    if set(result['rank_scores'])!=expected:result['reason']='incomplete_rank_coverage';return result
    if set(result['support'])!=expected:result['reason']='incomplete_comparable_evidence';return result
    ordered=sorted(result['rank_scores'],key=result['rank_scores'].get,reverse=True)
    if len(ordered)<2:return result
    best,second=ordered[:2]
    if result['rank_scores'][best]>model['min_score'] and result['rank_scores'][best]-result['rank_scores'][second]>model['margin']:
        result.update(status='localized',rank=best,reason='unique_local_duration_anomaly')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('fit');create.add_argument('--fit-session',type=Path,required=True);create.add_argument('--cal-session',type=Path,required=True)
    create.add_argument('--start',type=int,default=13);create.add_argument('--end',type=int,default=60);create.add_argument('--output',type=Path,required=True)
    predict=sub.add_parser('infer');predict.add_argument('--session',type=Path,required=True);predict.add_argument('--model',type=Path,required=True)
    predict.add_argument('--start',type=int,required=True);predict.add_argument('--end',type=int,required=True);predict.add_argument('--world-size',type=int,default=16);predict.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve existing candidate/prediction artifact')
    if args.command=='fit':
        train=[row for row in common.read_session_evidence(args.fit_session) if args.start<=row['iteration']<=args.end]
        cal=common.read_session_evidence(args.cal_session)
        windows=[[row for row in cal if start<=row['iteration']<start+8] for start in range(args.start,args.end+1,8)]
        result=fit(train,windows,fit_sessions=[str(args.fit_session.resolve())],calibration_sessions=[str(args.cal_session.resolve())])
    else:
        model=json.loads(args.model.read_text());verify_model(model)
        rows=[row for row in common.read_session_evidence(args.session) if args.start<=row['iteration']<=args.end]
        result=infer(rows,model,range(args.world_size));result.update(window=[args.start,args.end],model_sha256=model['sha256'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),status=result.get('status','fitted'),rank=result.get('rank'),sha256=result.get('sha256'))))

if __name__=='__main__':main()
