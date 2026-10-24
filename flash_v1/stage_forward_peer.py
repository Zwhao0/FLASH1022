"""TP2 matched-current-work residual contrasts, with healthy physical scales.

Independent auxiliary model: never modifies dc89 or the earlier stage models.
Context is observer installation, not fault activity. Only current paired work
must match; there is no historical routed-token acceptance range.
"""
import hashlib
import json
import math
from statistics import median

from flash_v1.stage_forward_localize import audit_healthy, semantic_key, validate_row


MODEL_TYPE = 'flash_stage_forward_tp_peer_v1'
RELATIVE_FLOOR = .15
MIN_SCORE, MIN_MARGIN, MIN_SUPPORT = 6., 2., 3


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def topology_pairs(topology):
    if topology.get('world_size') != 16:
        raise ValueError('peer model requires the current sixteen-rank population')
    pairs = sorted({tuple(sorted(group['members'])) for group in topology['groups']
                    if group['name'] == 'tensor_parallel'})
    if (len(pairs) != 8 or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs) or
            sorted(rank for pair in pairs for rank in pair) != list(range(16))):
        raise ValueError('topology must declare eight disjoint actual TP2 groups')
    return [list(pair) for pair in pairs]


def verify_topology(model, topology, layout):
    verify_model(model)
    if layout not in model['layouts'] or topology_pairs(topology) != model['layouts'][layout]['pairs']:
        raise ValueError('runtime topology does not match the peer model TP2 partition')
    if topology.get('parallel_layout') != layout:
        raise ValueError('runtime topology layout mismatch')


def _configuration(row):
    copied = dict(row)
    copied['stage_metadata'] = {key: value for key, value in row['stage_metadata'].items()
                               if key not in ('tp_group_members','tp_group_ranks')}
    return semantic_key(copied)


def _matched(left, right):
    return (_configuration(left) == _configuration(right) and
        all(a['work'] == b['work'] and a['stage_inputs'] == b['stage_inputs']
            for a, b in zip(left['microbatches'], right['microbatches'])))


def _pair_id(pair):
    return ','.join(map(str, pair))


def _index(rows, expected_ranks, context):
    index, steps, layouts, epochs = {}, set(), set(), set()
    for row in rows:
        validate_row(row)
        metadata = row['stage_metadata']
        if metadata.get('instrumentation_context') != context:
            raise ValueError('missing_or_incompatible_instrumentation_context')
        if metadata.get('tp_size') != 2:
            raise ValueError('peer contrast is restricted to TP2')
        declared=metadata.get('tp_group_ranks')
        if (not isinstance(declared,list) or len(declared)!=2 or
                any(type(rank) is not int for rank in declared) or
                len(set(declared))!=2 or row['rank'] not in declared):
            raise ValueError('missing_or_invalid_observed_tp_group_ranks')
        identity = row['iteration'], row['rank']
        if row['rank'] not in expected_ranks or identity in index:
            raise ValueError('unexpected_rank_or_duplicate_sidecar')
        index[identity] = row
        steps.add(row['iteration'])
        epochs.add(row['epoch'])
        layouts.add(metadata['layout'])
    if len(layouts) != 1 or not steps or index.keys() != {
            (step, rank) for step in steps for rank in expected_ranks}:
        raise ValueError('incomplete_rank_step_structure_or_mixed_layout')
    return index, sorted(steps), next(iter(layouts)), epochs


def _differences(index, steps, pair, expected_configuration=None):
    values, unknown = [], []
    for step in steps:
        left, right = (index[step, rank] for rank in pair)
        for row in (left, right):
            declared = row['stage_metadata']['tp_group_ranks']
            if sorted(declared) != pair:
                raise ValueError('declared_current_tp_group_mismatch')
        if not _matched(left, right):
            unknown.append(dict(iteration=step, reason='current_two_mb_work_or_configuration_mismatch'))
            continue
        if expected_configuration is not None and _configuration(left) != expected_configuration:
            unknown.append(dict(iteration=step, reason='unknown_model_or_shape_configuration'))
            continue
        values.append(dict(iteration=step, difference_us=left['duration_us']-right['duration_us'],
            pair_mean_us=(left['duration_us']+right['duration_us'])/2))
    return values, unknown


def _score(rows, model, expected_ranks):
    index, steps, layout, epochs = _index(rows, expected_ranks, model['instrumentation_context'])
    if len(steps) != 8 or steps[-1]-steps[0] != 7 or len(epochs) != 1:
        raise ValueError('expected_complete_single_epoch_eight_step_window')
    if layout not in model['layouts']:
        raise ValueError('unknown_layout')
    result = dict(pair_scores={}, pair_support={}, unknown={},
        coverage=dict(structural_rows=len(rows), expected_ranks=sorted(expected_ranks), layout=layout))
    for pair in model['layouts'][layout]['pairs']:
        identity = _pair_id(pair)
        reference = model['layouts'][layout]['baselines'][identity]
        values, unknown = _differences(index, steps, pair, reference['configuration'])
        zs = [(item['difference_us']-reference['offset_us'])/reference['scale_us'] for item in values]
        detail = dict(pair=pair, matched_iterations=len(zs), unknown_iterations=8-len(zs),
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


def fit(rows, manifest, topologies, instrumentation_context):
    """Healthy-only fit/cal; accepts the already approved 32/16 or plain8/8."""
    if instrumentation_context not in ('plain', 'scoped_dispatch'):
        raise ValueError('unsupported static observer context')
    expected = manifest['expected_ranks']
    if sorted(expected) != list(range(16)):
        raise ValueError('peer model requires sixteen actual ranks')
    model = dict(model_type=MODEL_TYPE, schema_version=1, world_size=16, tp_size=2,
        instrumentation_context=instrumentation_context, min_score=MIN_SCORE, margin=MIN_MARGIN,
        min_support=MIN_SUPPORT, window_size=8, min_comparable_pairs=2, relative_floor=RELATIVE_FLOOR,
        scale_rule='max(1us,1.4826*MAD(d_fit),0.15*median((residualA+residualB)/2)_fit)',
        offset_rule='fit-only median(residual(sorted_pair[0])-residual(sorted_pair[1]))',
        direction_rule='median signed z determines side; >=3 distinct matched iteration z values must each cross the same directional calibrated threshold',
        candidate_rule='absolute median signed z competes across known pairs; true known runner-up margin>2; unknowns never scored zero',
        healthy_manifest=manifest, layouts={})
    audit = dict(status='healthy_calibrated_peer_candidate_not_independently_validated',
        context=instrumentation_context, by_layout={}, calibration=[])
    for layout, sources in sorted(manifest['by_layout'].items()):
        subset = {phase: [row for row in rows[phase] if row['stage_metadata']['layout']==layout]
                  for phase in ('fit','calibration')}
        audit['by_layout'][layout] = audit_healthy(subset, dict(expected_ranks=expected,
            window_size=8, protocol=manifest.get('protocol','fit32_cal16_independent_v1'), **sources))
        pairs = topology_pairs(topologies[layout])
        if topologies[layout].get('parallel_layout')!=layout:
            raise ValueError('healthy topology declaration layout mismatch')
        fi, steps, observed_layout, _ = _index(subset['fit'], expected, instrumentation_context)
        if observed_layout != layout:
            raise ValueError('healthy topology layout mismatch')
        item = dict(pairs=pairs, topology_sha256=_hash(topologies[layout]), baselines={})
        for pair in pairs:
            values, unknown = _differences(fi, steps, pair)
            configurations = {_configuration(fi[value['iteration'],pair[0]]) for value in values}
            if len(values)<MIN_SUPPORT or len(configurations)!=1:
                raise ValueError('inadequate healthy matched-pair fit or mixed configurations')
            differences = [value['difference_us'] for value in values]
            offset=median(differences)
            mad=median(abs(value-offset) for value in differences)
            physical=median(value['pair_mean_us'] for value in values)
            if not math.isfinite(physical) or physical <= 0:
                raise ValueError('nonpositive healthy physical work duration')
            item['baselines'][_pair_id(pair)] = dict(offset_us=offset, mad_difference_us=mad,
                fit_pair_mean_us=physical, scale_us=max(1.,1.4826*mad,RELATIVE_FLOOR*physical),
                fit_matched_iterations=len(values), fit_unknown=unknown,
                configuration=configurations.pop())
        model['layouts'][layout]=item
    maximum=0.
    for layout,sources in sorted(manifest['by_layout'].items()):
        source=sources['calibration']
        for start in range(source['start'],source['end']+1,8):
            window=[row for row in rows['calibration'] if row['stage_metadata']['layout']==layout and start<=row['iteration']<start+8]
            result=_score(window,model,expected)
            if len(result['pair_scores'])<2:
                raise ValueError('inadequate comparable healthy calibration pairs')
            maximum=max(maximum,max(result['pair_scores'].values()))
            audit['calibration'].append(dict(layout=layout,start=start,end=start+7,**result))
    model.update(max_healthy_calibration_absolute_z=maximum,
        min_score=max(MIN_SCORE,maximum+MIN_MARGIN), calibration_windows=len(audit['calibration']),
        fit_rows_sha256=_rows_hash(rows['fit']), calibration_rows_sha256=_rows_hash(rows['calibration']))
    model['sha256']=_hash(model)
    audit.update(model_sha256=model['sha256'],min_score=model['min_score'],margin=model['margin'],
        limits=['TP2 and single-rank slowdown only; pair common-mode or dual faults can cancel.',
                'Residual still includes CPU enqueue bias; physical scale floor is fixed from the pre-existing noise principle.',
                'Context-dependent offsets/scales are fit-only; same-session plain8/8 data and TP views are correlated.',
                'No fault label, rank, phase, active state or dose fits offsets/scales/thresholds.'])
    return model,audit


def _rows_hash(rows):
    digest=hashlib.sha256()
    for row in sorted(rows,key=lambda row:(row['stage_metadata']['layout'],row['iteration'],row['rank'])):
        digest.update(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def verify_model(model):
    if model.get('model_type')!=MODEL_TYPE or model.get('schema_version')!=1 or model.get('world_size')!=16:
        raise ValueError('unsupported peer model')
    if model.get('sha256')!=_hash({key:value for key,value in model.items() if key!='sha256'}):
        raise ValueError('peer model hash mismatch')
    if (model.get('tp_size')!=2 or model.get('relative_floor')!=RELATIVE_FLOOR or
            model.get('min_score',0)<MIN_SCORE or model.get('margin',0)<MIN_MARGIN or
            model.get('min_support')!=MIN_SUPPORT or model.get('window_size')!=8 or
            model.get('min_comparable_pairs')!=2):
        raise ValueError('invalid peer scope or evidence thresholds')
    if model.get('instrumentation_context') not in ('plain','scoped_dispatch'):
        raise ValueError('invalid peer observer context')
    for item in model['layouts'].values():
        if sorted(rank for pair in item['pairs'] for rank in pair)!=list(range(16)) or any(len(pair)!=2 for pair in item['pairs']):
            raise ValueError('invalid TP2 pair partition')
        for reference in item['baselines'].values():
            scale=max(1.,1.4826*reference['mad_difference_us'],RELATIVE_FLOOR*reference['fit_pair_mean_us'])
            if not math.isfinite(reference['offset_us']) or reference['scale_us']!=scale or scale<=0:
                raise ValueError('invalid peer physical noise scale')


def context_binding(model,provenance):
    verify_model(model)
    if not provenance:
        raise ValueError('context/topology provenance required')
    binding=dict(binding_type='flash_stage_forward_context_binding_v1',model_type=MODEL_TYPE,
        model_sha256=model['sha256'],instrumentation_context=model['instrumentation_context'],
        topology_sha256_by_layout={layout:item['topology_sha256'] for layout,item in model['layouts'].items()},
        provenance=provenance)
    binding['sha256']=_hash(binding)
    return binding


def infer_bound(rows,model,expected_ranks,binding):
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


def fallback(original,rows,model,expected_ranks,binding):
    if original.get('status')=='localized':
        return original
    auxiliary=infer_bound(rows,model,expected_ranks,binding)
    return auxiliary if auxiliary['status']=='localized' else dict(original,stage_forward_peer=auxiliary)
