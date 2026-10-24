"""Explicit v2 entry point; reuses the unchanged v1 detector/controller.

Only this process substitutes its diagnostic callback. Existing ``online.py``
entry points and historical models remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import signal
from flash_v1 import online
from flash_v1 import localize_v2 as log_model


def _model_api(model):
    if model.get('model_type')=='flash_local_v2':return log_model
    if model.get('model_type')=='flash_local_v2_linear':
        from flash_v1 import localize_v2_linear
        return localize_v2_linear
    raise ValueError('unsupported evidence model type')


def verify_model(model):
    return _model_api(model).verify_model(model)


def infer(rows,model,expected_ranks):
    return _model_api(model).infer(rows,model,expected_ranks)


ROLES = {('attention','forward'),('attention','backward'),
         ('experts','forward'),('experts','backward')}
COMM_COUNTS = {('attention','forward'):1, ('attention','backward'):3,
               ('experts','forward'):1, ('experts','backward'):2}
PHASE_CODES = {('attention','forward'):1, ('attention','backward'):2,
               ('experts','forward'):3, ('experts','backward'):4}


def _finite(value,positive=False):
    return (isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and (value>0 if positive else value>=0))


def evidence_integrity_reason(rows,schedule,expected_ranks):
    """Fail closed before scoring: complete four-role same-stream evidence."""
    from flash_v1.evidence import interval_residual
    expected={(step,rank) for step in range(schedule['start'],schedule['end']+1) for rank in expected_ranks}
    roles={identity:[] for identity in expected}
    for row in rows:
        if not isinstance(row,dict) or type(row.get('iteration')) is not int or type(row.get('rank')) is not int:
            return 'invalid_evidence_rank_or_iteration'
        identity=(row.get('iteration'),row.get('rank'))
        role=(row.get('slot'),row.get('phase'))
        if identity not in roles or role not in ROLES:return 'unexpected_evidence_identity_or_role'
        roles[identity].append(role)
        if row.get('coverage_complete') is not True:return 'incomplete_or_invalid_scope'
        if type(row.get('stream_id')) is not int:return 'invalid_scope_stream'
        if type(row.get('layer')) is not int or row['layer']<0:return 'invalid_semantic_layer'
        duration,total=row.get('duration_us'),row.get('module_elapsed_us')
        if not _finite(duration) or not _finite(total,positive=True):return 'nonfinite_or_invalid_scope_duration'
        intervals=row.get('communication_intervals')
        if not isinstance(intervals,list) or len(intervals)!=COMM_COUNTS[role]:return 'incomplete_communication_coverage'
        paired=[]
        for interval in intervals:
            if not isinstance(interval,dict):return 'invalid_communication_interval_record'
            left,right=interval.get('start_us'),interval.get('end_us')
            if not _finite(left) or not _finite(right):return 'nonfinite_or_invalid_communication_interval'
            expected_phase=((row['layer']+1)<<8)|PHASE_CODES[role]
            if interval.get('phase_code')!=expected_phase:return 'communication_semantic_phase_mismatch'
            paired.append((left,right))
        try:recomputed=interval_residual(total,paired)
        except (ValueError,TypeError):return 'communication_interval_outside_scope'
        if not math.isclose(duration,recomputed,rel_tol=1e-6,abs_tol=.05):return 'local_residual_union_mismatch'
    if any(len(observed)!=4 or set(observed)!=ROLES for observed in roles.values()):return 'incomplete_or_duplicate_semantic_roles'
    return None


def _combine_stage_forward(original, auxiliary, model_type):
    """Fuse accepted same-window peer evidence without changing the core hint."""
    if original.get('status')=='localized' and model_type!='flash_stage_forward_tp_peer_v1':
        return original
    if auxiliary.get('status') != 'localized':
        return dict(original,stage_forward_aux=auxiliary)
    # Keep original scores/reasons in the saved original_dc89.json. Only the
    # public verdict changes; auxiliary scores/coverage remain explicitly nested.
    result = dict(original,status='localized',rank=auxiliary['rank'],
                  reason=('stage_forward_tp_peer_priority' if model_type=='flash_stage_forward_tp_peer_v1'
                          else 'stage_forward_aux_fallback'),
                  source='stage_forward_aux',stage_forward_aux=auxiliary)
    if original.get('profile_transition') is not None:
        result['profile_transition_core'] = {key:original.get(key) for key in ('status','rank','profile_transition')}
    return result

def _stage_forward_auxiliary(client, output, schedule, original, configuration, expected):
    """Prefer an accepted TP-peer verdict; otherwise preserve the original core."""
    if (not configuration.get('stage_forward_aux_enabled', False)
            or not configuration.get('stage_forward_aux_model')):
        return original
    if not configuration.get('stage_forward_aux_context_binding'):
        return dict(original, stage_forward_aux=dict(status='rejected', reason='missing_instrumentation_context_binding'))
    if schedule['end'] - schedule['start'] != 7:
        return dict(original, stage_forward_aux=dict(status='rejected', reason='requires_eight_step_window'))
    sampling = configuration.get('stage_forward_aux_sampling')
    if sampling is None:
        required={(step, rank) for step in range(schedule['start'], schedule['end']+1) for rank in expected}
    else:
        from flash_v1.stage_forward_sampling import CONFIG
        if sampling != CONFIG:
            return dict(original, stage_forward_aux=dict(status='rejected',reason='explicit_sparse_sampling_contract_required'))
        required={(schedule['start']+offset-1, rank) for offset in CONFIG['sample_offsets'] for rank in expected}
    try:
        auxiliary_model=json.loads(Path(configuration['stage_forward_aux_model']).read_text())
        if not isinstance(auxiliary_model,dict):raise ValueError('invalid_auxiliary_model_record')
        model_type=auxiliary_model.get('model_type')
        if sampling is not None and model_type != 'flash_stage_forward_tp_peer_v1':
            raise ValueError('sparse3 requires the bound TP-peer model')
        if model_type not in ('flash_stage_forward_tp_peer_v1','flash_stage_forward_healthy_v1'):
            raise ValueError('unknown_stage_auxiliary_model_type')
        # Historical absolute-duration auxiliaries retain their original policy.
        if original.get('status') == 'localized' and model_type == 'flash_stage_forward_healthy_v1':
            return original
        # No later epoch can start before this diagnosis completes. Two bounded
        # windows retain current records plus any duplicates, without rereading
        # every previous full-stage work vector from this session.
        limit=len(required)
        transported=[online.decode(value) for node in range(2)
            for value in client.lrange(f'online_stage_forward_aux_node{node}',-limit,-1)]
        if any(not isinstance(row,dict) or type(row.get('iteration')) is not int
               or type(row.get('rank')) is not int or type(row.get('epoch')) is not int
               or row.get('record_type') != 'stage_forward_aux' for row in transported):
            raise ValueError('invalid_auxiliary_transport_metadata')
        rows=[row for row in transported if row['epoch']==schedule['epoch']
              and schedule['start']<=row['iteration']<=schedule['end']]
        seen={(row['iteration'],row['rank']) for row in rows}
        if not required.issubset(seen):
            return None  # Retry only within the existing bounded drain deadline.
        if seen != required or len(rows) != len(required):
            raise ValueError('duplicate_or_invalid_auxiliary_identity')
        if sampling is not None:
            from flash_v1.stage_forward_sampling import contract as sampling_contract
            if any(row.get('sampling_contract') != sampling_contract(schedule['epoch'],schedule['start']) for row in rows):
                raise ValueError('incompatible_sparse_schedule_contract')
        destination=Path(output)/f"online_profile_{schedule['epoch']}"
        serialized=json.dumps(rows,sort_keys=True,separators=(',',':'))
        (destination/'stage_forward_aux.json').write_text(serialized+'\n')
        binding=json.loads(Path(configuration['stage_forward_aux_context_binding']).read_text())
        if not isinstance(binding,dict):raise ValueError('invalid_auxiliary_binding_record')
        if model_type=='flash_stage_forward_tp_peer_v1':
            from flash_v1.stage_forward_peer import verify_topology
            if sampling is None:
                from flash_v1.stage_forward_peer import infer_bound as infer_auxiliary
            else:
                from flash_v1.stage_forward_sparse_peer import infer_bound as infer_auxiliary
            current_topology=json.loads((Path(output)/'topology.json').read_text())
            verify_topology(auxiliary_model,current_topology,configuration.get('parallel_layout'))
        elif model_type=='flash_stage_forward_healthy_v1':
            from flash_v1.stage_forward_localize import infer_bound as infer_auxiliary
            # This new topology-only descriptor was absent from historical
            # healthy-v1 semantic keys. Preserve saved raw and old model hashes.
            rows=[dict(row,stage_metadata={key:value for key,value in row['stage_metadata'].items()
                         if key!='tp_group_ranks'}) if 'stage_metadata' in row else row for row in rows]
        else:
            raise ValueError('unknown_stage_auxiliary_model_type')
        auxiliary=infer_auxiliary(rows,auxiliary_model,expected,binding)
        if not isinstance(auxiliary,dict):raise ValueError('invalid_auxiliary_verdict_record')
        auxiliary=dict(auxiliary, model_type=model_type, evidence_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                       frozen_model_hash=auxiliary_model.get('sha256'))
    except (ValueError,TypeError,OSError,KeyError) as error:
        return dict(original,stage_forward_aux=dict(status='rejected',reason=str(error)))
    return _combine_stage_forward(original, auxiliary, model_type)


def diagnose(client,output,schedule,model):
    verify_model(model)
    topology=json.loads((Path(output)/'topology.json').read_text())
    expected=set(range(topology['world_size']))
    try:
        transported=[online.decode(value) for node in range(2)
                     for value in client.lrange(f'online_evidence_node{node}',0,-1)]
    except (ValueError,TypeError):
        return dict(status='rejected',rank=None,predicted='inconclusive',reason='malformed_evidence_transport',frozen_model_hash=model['sha256'])
    if any(not isinstance(row,dict) or type(row.get('iteration')) is not int or type(row.get('rank')) is not int or type(row.get('epoch')) is not int or row.get('record_type') not in ('measurement','iteration_complete') for row in transported):
        return dict(status='rejected',rank=None,predicted='inconclusive',reason='invalid_evidence_transport_metadata',frozen_model_hash=model['sha256'])
    records=[row for row in transported if schedule['start']<=int(row['iteration'])<=schedule['end']
             and int(row.get('epoch',schedule['epoch']))==schedule['epoch']]
    markers=[row for row in records if row.get('record_type')=='iteration_complete']
    seen={(int(row['iteration']),int(row['rank'])) for row in markers}
    required={(step,rank) for step in range(schedule['start'],schedule['end']+1) for rank in expected}
    if not required.issubset(seen):return None
    rows=[row for row in records if row.get('record_type')!='iteration_complete']
    destination=Path(output)/f"online_profile_{schedule['epoch']}"
    destination.mkdir(exist_ok=True)
    evidence=json.dumps(rows,sort_keys=True,separators=(',',':'))
    (destination/'evidence.json').write_text(evidence+'\n')
    (destination/'evidence_markers.json').write_text(json.dumps(markers)+'\n')
    identities=[(row['rank'],row['iteration'],row['slot'],row['phase'],row.get('occurrence',0)) for row in rows]
    row_counts=Counter((int(row['iteration']),int(row['rank'])) for row in rows)
    count_mismatch=any(row_counts[int(marker['iteration']),int(marker['rank'])]!=int(marker['rows_count']) for marker in markers)
    integrity_reason=evidence_integrity_reason(rows,schedule,expected)
    if count_mismatch:
        result=dict(status='rejected',rank=None,reason='completion_row_count_mismatch')
    elif len(set(identities))!=len(identities):
        result=dict(status='rejected',rank=None,reason='duplicate_evidence')
    elif len(markers)!=len(required):
        result=dict(status='rejected',rank=None,reason='duplicate_or_invalid_completion_markers')
    elif integrity_reason is not None:
        result=dict(status='rejected',rank=None,reason=integrity_reason)
    else:
        result=infer(rows,model,expected)
    configuration_path=Path(output)/'configuration.json'
    configuration=json.loads(configuration_path.read_text()) if configuration_path.exists() else {}
    if configuration.get('monitor_profile_transition',False):
        from flash_v1.profile_transition import candidate
        transition=candidate(rows,model,expected,schedule,result)
        if transition is not None:result['profile_transition']=transition
    if configuration.get('stage_forward_aux_enabled',False):
        original_snapshot=dict(result, predicted=f"rank:{result['rank']}" if result['status']=='localized' else 'inconclusive',
            run_id=destination.name, frozen_model_hash=model['sha256'],
            evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
            coverage=dict(completion_markers=len(seen),expected_markers=len(required),evidence_rows=len(rows)))
        (destination/'original_dc89.json').write_text(json.dumps(original_snapshot,sort_keys=True)+'\n')
        result=_stage_forward_auxiliary(client,output,schedule,result,configuration,expected)
        if result is None:return None
    result.update(predicted=f"rank:{result['rank']}" if result['status']=='localized' else 'inconclusive',
                  run_id=destination.name,frozen_model_hash=model['sha256'],
                  evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
                  coverage=dict(completion_markers=len(seen),expected_markers=len(required),evidence_rows=len(rows)))
    return result


def diagnose_timeout(client, output, schedule, model):
    """At the existing drain deadline, preserve a completed original diagnosis."""
    destination=Path(output)/f"online_profile_{schedule['epoch']}"
    try:
        original=json.loads((destination/'original_dc89.json').read_text())
        if (not isinstance(original,dict) or original.get('run_id')!=destination.name
                or original.get('frozen_model_hash')!=model.get('sha256')
                or original.get('status') not in ('localized','rejected')):
            return None
        rank=original.get('rank')
        if original['status']=='localized' and (type(rank) is not int or original.get('predicted')!=f'rank:{rank}'):
            return None
        return dict(original,stage_forward_aux=dict(status='unavailable',reason='bounded_auxiliary_drain_timeout'))
    except (OSError,ValueError,TypeError,KeyError):
        return None


# The unchanged controller only calls this hook after its original 10s deadline.
diagnose.timeout_result=diagnose_timeout


def controller(client,output,model,warmup=12,max_profiles=3,stopped=lambda:False):
    verify_model(model)
    original=online.diagnose
    try:
        online.diagnose=diagnose
        return online.controller(client,output,model,warmup,max_profiles,stopped)
    finally:
        online.diagnose=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--redis-port',type=int,default=17422)
    parser.add_argument('--warmup',type=int,default=12);parser.add_argument('--max-profiles',type=int,default=3)
    args=parser.parse_args()
    model=json.loads(Path(args.model).read_text());verify_model(model)
    stop=[False]
    def terminate(*_):stop[0]=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    import redis
    client=redis.Redis(port=args.redis_port,decode_responses=True)
    controller(client,args.output,model,args.warmup,args.max_profiles,lambda:stop[0])

if __name__=='__main__':main()
