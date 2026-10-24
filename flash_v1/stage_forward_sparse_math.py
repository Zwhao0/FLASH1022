"""Frozen sparse peer functions; generated at build time, no runtime source rewriting."""
from statistics import median
from flash_v1.stage_forward_peer import _index, _differences, _pair_id, verify_model, _hash, MODEL_TYPE, MIN_SUPPORT
BASE_PEER_SOURCE_SHA256 = 'd5a2b0c4b0bd8ed43573ec8d7419d106bf9e2f27aa926193e5fee77d1d4dfada'
MASKS={'fixed3':(1,3,7)}

def _score(rows, model, expected_ranks, expected_steps):
    index, steps, layout, epochs = _index(rows, expected_ranks, model['instrumentation_context'])
    if steps != expected_steps or len(epochs) != 1:
        raise ValueError('expected_complete_explicit_sparse_rank_step_window')
    if layout not in model['layouts']:
        raise ValueError('unknown_layout')
    result = dict(pair_scores={}, pair_support={}, unknown={},
        coverage=dict(structural_rows=len(rows), expected_ranks=sorted(expected_ranks), layout=layout))
    for pair in model['layouts'][layout]['pairs']:
        identity = _pair_id(pair)
        reference = model['layouts'][layout]['baselines'][identity]
        values, unknown = _differences(index, steps, pair, reference['configuration'])
        zs = [(item['difference_us']-reference['offset_us'])/reference['scale_us'] for item in values]
        detail = dict(pair=pair, matched_iterations=len(zs), unknown_iterations=len(expected_steps)-len(zs),
            comparable=len(zs)>=MIN_SUPPORT, iteration_signed_z=[dict(iteration=item['iteration'], z=z)
                for item, z in zip(values, zs)])
        if len(zs) >= MIN_SUPPORT:
            signed = median(zs)
            direction = 1 if signed > 0 else -1 if signed < 0 else 0
            detail.update(signed_median_z=signed,
                directional_rank=pair[0] if direction > 0 else pair[1] if direction < 0 else None,
                above_threshold_same_direction_iterations=sum(z*direction>model['min_score'] for z in zs))
            result['pair_scores'][identity] = abs(signed)
        if unknown:
            result['unknown'][identity] = unknown
        if len(zs) < MIN_SUPPORT:
            result['unknown'].setdefault(identity, []).append(dict(reason='fewer_than_three_matched_iterations'))
        result['pair_support'][identity] = detail
    result['coverage'].update(comparable_pairs=sorted(result['pair_scores']),
        comparable_pair_count=len(result['pair_scores']),
        matched_pair_iterations=sum(item['matched_iterations'] for item in result['pair_support'].values()),
        unknown_pairs=sorted(result['unknown']))
    return result

def _infer_bound(rows,model,expected_ranks,binding,_score):
    verify_model(model)
    if (binding.get('sha256')!=_hash({key:value for key,value in binding.items() if key!='sha256'}) or
            binding.get('model_sha256')!=model['sha256'] or binding.get('model_type')!=MODEL_TYPE or
            binding.get('instrumentation_context')!=model['instrumentation_context'] or
            binding.get('topology_sha256_by_layout')!={layout:item['topology_sha256'] for layout,item in model['layouts'].items()}):
        raise ValueError('peer context/topology binding mismatch')
    if sorted(expected_ranks)!=list(range(16)):
        raise ValueError('peer expected-rank population mismatch')
    rows=list(rows)
    try:
        result=_score(rows,model,expected_ranks)
    except (ValueError,TypeError,KeyError) as error:
        return dict(status='rejected',rank=None,reason=str(error),frozen_model_sha256=model['sha256'],binding_sha256=binding['sha256'])
    result.update(status='rejected',rank=None,reason='insufficient_or_ambiguous_peer_contrast',
        frozen_model_sha256=model['sha256'],binding_sha256=binding['sha256'],
        instrumentation_context=model['instrumentation_context'])
    ordered=sorted(result['pair_scores'],key=result['pair_scores'].get,reverse=True)
    if len(ordered)<2:
        result['reason']='fewer_than_two_comparable_pairs'
        return result
    best,runner=ordered[:2]
    detail=result['pair_support'][best]
    result.update(candidate_pair=best,runner_up_pair=runner,candidate=detail['directional_rank'],
        actual_pair_margin=result['pair_scores'][best]-result['pair_scores'][runner])
    if (result['pair_scores'][best]>model['min_score'] and result['actual_pair_margin']>model['margin'] and
            detail['above_threshold_same_direction_iterations']>=MIN_SUPPORT):
        result.update(status='localized',rank=detail['directional_rank'],reason='persistent_matched_tp2_peer_residual_contrast')
    elif detail['above_threshold_same_direction_iterations']<MIN_SUPPORT:
        result['reason']='insufficient_persistent_directional_threshold_support'
    return result

def infer_sparse(rows,model,expected_ranks,binding,contract):
    if (not isinstance(contract,dict) or contract.get('schema')!='offline_sparse_aux_v1'
            or contract.get('window_size')!=8 or contract.get('mask') not in MASKS
            or type(contract.get('start')) is not int or contract.get('epoch') is None):
        raise ValueError('explicit fixed8-window sparse contract required')
    offsets=MASKS[contract['mask']]
    steps=[contract['start']+offset-1 for offset in offsets]
    rows=list(rows)
    if any(row.get('epoch')!=contract['epoch'] for row in rows):
        return dict(status='rejected',rank=None,reason='sparse_epoch_mismatch')
    # The original infer implementation verifies the unchanged model and binding,
    # applies the same median/min3/directional score/margin/known-pair competition.
    result=_infer_bound(rows,model,expected_ranks,binding,
        lambda selected,m,e:_score(selected,m,e,steps))
    result['sampling_contract']=dict(contract,sample_offsets=list(offsets),
        unsampled_offsets=[i for i in range(1,9) if i not in offsets],
        unsampled_is_healthy=False,core_window_unchanged=True)
    return result
