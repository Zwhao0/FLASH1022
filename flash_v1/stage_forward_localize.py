"""CPU healthy-only stage residual models with explicit instrumentation binding.

Retain full observed work vectors; use only the approved low-dimensional token
domain and robust duration baseline. Calibration is not independent validation.
Historical infer remains unbound for reproducibility; runtime uses infer_bound.
"""
from collections import defaultdict
import hashlib
import json
import math
from statistics import median

from flash_v1.stage_forward_evidence import COLLECTIVES, SCHEMA, strict_residual


MODEL_TYPE = 'flash_stage_forward_healthy_v1'
MIN_SUPPORT = 3
MIN_SCORE = 6.
MIN_MARGIN = 2.


def _finite(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _shape(item, dimensions, allow_empty=False):
    if not isinstance(item, dict) or not isinstance(item.get('dtype'), str):
        raise ValueError('missing shape/dtype')
    shape = item.get('shape')
    allowed = (dimensions,) if isinstance(dimensions, int) else dimensions
    if not isinstance(shape, list) or len(shape) not in allowed or any(
            type(size) is not int or size < (0 if allow_empty and index == 0 else 1)
            for index, size in enumerate(shape)):
        raise ValueError('invalid tensor shape')
    return shape


def validate_row(row):
    """Recompute both MB residuals and aggregate, independently of coverage flag."""
    if row.get('schema') != SCHEMA or row.get('record_type') != 'stage_forward_aux':
        raise ValueError('unexpected auxiliary schema')
    if row.get('coverage_complete') is not True or row.get('invalid_reasons') != []:
        raise ValueError('incomplete auxiliary evidence')
    if any(type(row.get(key)) is not int or row[key] < 0 for key in ('rank', 'iteration', 'epoch')):
        raise ValueError('invalid auxiliary identity')
    expected = row.get('expected_collectives')
    if not isinstance(expected, dict) or set(expected) != set(COLLECTIVES) or any(
            type(value) is not int or value < 0 for value in expected.values()):
        raise ValueError('invalid collective manifest')
    metadata_count = row.get('expected_ep_metadata_spans')
    if type(metadata_count) is not int or metadata_count < 0:
        raise ValueError('invalid metadata manifest')
    microbatches = row.get('microbatches')
    if row.get('expected_microbatches') != 2 or not isinstance(microbatches, list) or len(microbatches) != 2:
        raise ValueError('expected two forward microbatches')
    totals, pairs, work_keys = [], 0, None
    for index, item in enumerate(microbatches):
        if item.get('microbatch') != index or type(item.get('stream_id')) is not int:
            raise ValueError('invalid microbatch identity')
        inputs = item.get('stage_inputs')
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise ValueError('expected one stage input per microbatch')
        _shape(inputs[0], 2)
        work = item.get('work')
        if not isinstance(work, dict) or not work or (work_keys is not None and set(work) != work_keys):
            raise ValueError('inconsistent complete work population')
        work_keys = set(work)
        for key, values in work.items():
            if not isinstance(key, str) or not isinstance(values, list) or len(values) != 1:
                raise ValueError('invalid work identity or call count')
            # HF dense MLP keeps [B,S,H]; routed/shared expert leaves use [T,H].
            name = key.split(':', 1)[-1]
            _shape(values[0], (2, 3) if name == '' else 2, allow_empty=True)
        intervals = item.get('communication_intervals')
        if not isinstance(intervals, list) or any(not isinstance(value, dict) for value in intervals):
            raise ValueError('invalid communication intervals')
        counts = {name: sum(value.get('kind') == name for value in intervals) for name in COLLECTIVES}
        if counts != expected or item.get('collective_counts') != expected:
            raise ValueError('collective manifest mismatch')
        if sum(value.get('kind') == 'ep_metadata' for value in intervals) != metadata_count:
            raise ValueError('incomplete EP metadata spans')
        if any(value.get('kind') not in (*COLLECTIVES, 'ep_metadata') for value in intervals):
            raise ValueError('unsupported communication kind')
        value = strict_residual(item.get('forward_elapsed_us'),
            [(value.get('start_us'), value.get('end_us')) for value in intervals])
        if not _finite(item.get('duration_us')) or not math.isclose(value, item['duration_us'], rel_tol=1e-6, abs_tol=.05):
            raise ValueError('microbatch residual mismatch')
        if item.get('event_pairs') != len(intervals) + 1:
            raise ValueError('microbatch event count mismatch')
        totals.append(value)
        pairs += item['event_pairs']
    if not _finite(row.get('duration_us')) or not math.isclose(math.fsum(totals), row['duration_us'], rel_tol=1e-6, abs_tol=.05):
        raise ValueError('aggregate residual mismatch')
    if row.get('event_pairs') != pairs:
        raise ValueError('aggregate event count mismatch')
    if not isinstance(row.get('stage_metadata'), dict) or not row['stage_metadata']:
        raise ValueError('missing stage metadata')
    return row


def semantic_key(row):
    """Group only invariant actual shapes/config; no rank or duration feature."""
    # These redundant census fields were added after the first healthy capture.
    # Complete module identities below already encode the same EP/dense layout.
    metadata = {key: value for key, value in row['stage_metadata'].items()
                if key not in ('ep_layer_indices', 'dense_layer_indices', 'instrumentation_context')}
    payload = dict(stage_metadata=metadata,
        expected_collectives=row['expected_collectives'],
        expected_ep_metadata_spans=row['expected_ep_metadata_spans'],
        microbatches=[dict(stage_input=item['stage_inputs'][0],
            modules={key: dict(width=values[0]['shape'][-1], dtype=values[0]['dtype'])
                     for key, values in sorted(item['work'].items())})
            for item in row['microbatches']])
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def workload_diagnostics(row):
    """No timing division: summarize real routed work plus imbalance for audit."""
    routed, shared, dense = [], [], []
    for item in row['microbatches']:
        for key, values in item['work'].items():
            name = key.split(':', 1)[-1]
            tokens = math.prod(values[0]['shape'][:-1])
            if name.startswith('experts.'):
                routed.append(tokens)
            elif name == 'shared_expert':
                shared.append(tokens)
            elif name == '':
                dense.append(tokens)
            else:
                raise ValueError('unknown MLP role; inspect actual data before choosing features')
    return dict(routed_tokens=sum(routed), shared_tokens=sum(shared), dense_tokens=sum(dense),
        empty_routed_calls=sum(value == 0 for value in routed),
        max_routed_tokens=max(routed, default=0),
        routed_tokens_squared=sum(value * value for value in routed))


def _describe(values):
    center = median(values)
    return dict(count=len(values), min=min(values), max=max(values), median=center,
                mad=median(abs(value - center) for value in values))


def audit_healthy(rows, manifest):
    """Validate exact healthy rank/step grids, then describe work/noise only.

    rows={'fit': [...], 'calibration': [...]}; each manifest phase supplies
    session_id, seed, healthy=True, layout, start, end. This provenance assertion
    must be backed by the lead's session plan; sidecars contain no fault truth.
    """
    phases = ('fit', 'calibration')
    expected = manifest.get('expected_ranks')
    if not isinstance(expected, list) or len(set(expected)) != len(expected) or not expected or any(
            type(rank) is not int or rank < 0 for rank in expected):
        raise ValueError('explicit unique expected ranks required')
    if manifest.get('window_size') != 8:
        raise ValueError('complete eight-step calibration windows required')
    grouped = {phase: defaultdict(list) for phase in phases}
    protocol = manifest.get('protocol', 'fit32_cal16_independent_v1')
    if protocol not in ('fit32_cal16_independent_v1', 'plain_cost_fit8_cal8_v1'):
        raise ValueError('unknown healthy cohort protocol')
    phases_and_steps = (('fit', 8), ('calibration', 8)) if protocol == 'plain_cost_fit8_cal8_v1' else (('fit', 32), ('calibration', 16))
    for phase, steps in phases_and_steps:
        source = manifest[phase]
        if source.get('healthy') is not True or not source.get('session_id') or type(source.get('seed')) is not int:
            raise ValueError('independent healthy provenance required')
        start, end = source.get('start'), source.get('end')
        if type(start) is not int or type(end) is not int or end - start + 1 != steps:
            raise ValueError('healthy steps do not match explicit cohort protocol')
        if protocol == 'plain_cost_fit8_cal8_v1' and (start, end) != (
                (85, 92) if phase == 'fit' else (104, 111)):
            raise ValueError('plain reuse is bound to existing cost windows85..92/104..111')
        required = {(rank, step) for rank in expected for step in range(start, end + 1)}
        seen = set()
        for row in rows[phase]:
            validate_row(row)
            identity = row['rank'], row['iteration']
            if identity in seen or identity not in required:
                raise ValueError('duplicate or out-of-schedule healthy row')
            seen.add(identity)
            if row['stage_metadata'].get('layout') != source.get('layout'):
                raise ValueError('healthy layout mismatch')
            if protocol == 'plain_cost_fit8_cal8_v1' and row['stage_metadata'].get('instrumentation_context') != 'plain':
                raise ValueError('plain reuse requires explicitly attested plain context')
            grouped[phase][semantic_key(row)].append(row)
        if seen != required:
            raise ValueError('incomplete healthy rank-step grid')
    if protocol == 'fit32_cal16_independent_v1':
        if (manifest['fit']['session_id'] == manifest['calibration']['session_id'] or
                manifest['fit']['seed'] == manifest['calibration']['seed']):
            raise ValueError('fit/calibration sessions and seeds must be disjoint')
    elif (manifest['fit']['session_id'] != manifest['calibration']['session_id'] or
            manifest['fit']['seed'] != manifest['calibration']['seed']):
        raise ValueError('explicit plain reuse must preserve the original same session/seed')
    summaries = []
    for key in sorted(set(grouped['fit']) | set(grouped['calibration'])):
        detail = dict(semantic_key=json.loads(key))
        for phase in phases:
            values = grouped[phase].get(key, [])
            features = [workload_diagnostics(row) for row in values]
            detail[phase] = dict(rows=len(values),
                duration_us=_describe([row['duration_us'] for row in values]) if values else None,
                workload={name: _describe([item[name] for item in features])
                          for name in features[0]} if features else {})
        summaries.append(detail)
    return dict(status='audit_only_no_model_selected', protocol=protocol, fit_rows=len(rows['fit']),
        calibration_rows=len(rows['calibration']), groups=summaries,
        metric='same-stream forward residual including CPU enqueue gaps and resource contention')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _row_hash(rows):
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda row: (row['stage_metadata']['layout'], row['rank'], row['iteration'])):
        digest.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def _supported(row, baseline):
    work = workload_diagnostics(row)
    if any(work[name] != value for name, value in baseline['fixed_work'].items()):
        return False, 'fixed_work_mismatch'
    left, right = baseline['routed_token_domain']
    if not left <= work['routed_tokens'] <= right:
        return False, 'routed_token_outside_observed_healthy_domain'
    return True, None


def _score_window(rows, model, expected_ranks):
    """Unknown work is absent from numerical competition, never assigned zero."""
    expected = set(expected_ranks)
    result = dict(rank_scores={}, support={}, unknown={}, reasons=[],
                  coverage=dict(expected_ranks=sorted(expected), structural_rows=len(rows)))
    observed, steps, epochs, layouts = set(), set(), set(), set()
    values, keys, unknown = defaultdict(list), defaultdict(set), defaultdict(lambda: defaultdict(int))
    for row in rows:
        try:
            validate_row(row)
        except (ValueError, TypeError, KeyError) as error:
            result['reasons'].append('invalid_sidecar:' + str(error))
            return result
        identity = row['rank'], row['iteration']
        if row['rank'] not in expected or identity in observed:
            result['reasons'].append('unexpected_rank_or_duplicate_sidecar')
            return result
        observed.add(identity)
        steps.add(row['iteration'])
        epochs.add(row['epoch'])
        layouts.add(row['stage_metadata']['layout'])
        key = semantic_key(row)
        reference = model['baselines'].get(key)
        if reference is None:
            unknown[row['rank']]['unknown_semantic_configuration'] += 1
            continue
        supported, reason = _supported(row, reference)
        if not supported:
            unknown[row['rank']][reason] += 1
            continue
        values[row['rank']].append((row['duration_us'] - reference['median_us']) / reference['scale_us'])
        keys[row['rank']].add(key)
    if (len(steps) != 8 or max(steps, default=0) - min(steps, default=0) != 7 or
            len(epochs) != 1 or len(layouts) != 1 or
            observed != {(rank, step) for rank in expected for step in steps}):
        result['reasons'].append('incomplete_rank_eight_step_structure_or_mixed_epoch_layout')
        return result
    for rank in sorted(expected):
        count = len(values[rank])
        comparable = count >= MIN_SUPPORT and len(keys[rank]) == 1
        result['support'][rank] = dict(comparable_iterations=count,
            unknown_iterations=8-count, comparable=comparable,
            semantic_group_sha256=_hash(next(iter(keys[rank]))) if len(keys[rank]) == 1 else None)
        if comparable:
            result['rank_scores'][rank] = median(values[rank])
        else:
            unknown[rank]['fewer_than_three_comparable_iterations_or_mixed_semantics'] += 1
        if unknown[rank]:
            result['unknown'][rank] = dict(unknown[rank])
    result['coverage'].update(comparable_ranks=sorted(result['rank_scores']),
        unknown_ranks=sorted(result['unknown']),
        comparable_iterations=sum(len(value) for value in values.values()))
    return result


def fit(rows, manifest):
    """Fit-only centers/scales; health calibration domains/tails, never faults.

    Accept a single layout manifest or {'by_layout': {...}, expected_ranks,
    window_size:8}. Numeric token domains are the observed fit+cal closed
    envelope; they are calibrated support, not independent coverage validation.
    """
    expected = manifest['expected_ranks']
    layouts = manifest['by_layout'] if 'by_layout' in manifest else {
        manifest['fit']['layout']: {'fit': manifest['fit'], 'calibration': manifest['calibration']}}
    audits = {}
    for layout, sources in layouts.items():
        subset = {phase: [row for row in rows[phase] if row['stage_metadata']['layout'] == layout]
                  for phase in ('fit', 'calibration')}
        audits[layout] = audit_healthy(subset, dict(expected_ranks=expected,
            window_size=manifest['window_size'],
            protocol=manifest.get('protocol', 'fit32_cal16_independent_v1'), **sources))
    if any(row['stage_metadata']['layout'] not in layouts for phase in rows for row in rows[phase]):
        raise ValueError('unexpected layout in healthy cohort')
    grouped = {phase: defaultdict(list) for phase in ('fit', 'calibration')}
    for phase in grouped:
        for row in rows[phase]:
            grouped[phase][semantic_key(row)].append(row)
    if set(grouped['fit']) != set(grouped['calibration']):
        raise ValueError('healthy fit/cal semantic populations differ')
    model = dict(model_type=MODEL_TYPE, schema_version=1, world_size=len(expected),
        min_score=MIN_SCORE, margin=MIN_MARGIN, min_support=MIN_SUPPORT, window_size=8,
        baseline_rule='fit-only median; scale=max(1us,1.4826*fit MAD)',
        domain_rule='closed routed-token envelope observed in healthy fit+cal; shared/dense work fixed; no extrapolation',
        unknown_rule='exclude unknown scores; target needs support3 and at least one comparable same-semantic peer; at least two comparable ranks',
        metric='stage forward stream residual, including CPU enqueue gaps and resource contention',
        fit_rows_sha256=_row_hash(rows['fit']), calibration_rows_sha256=_row_hash(rows['calibration']),
        healthy_manifest=manifest, baselines={})
    for key, fitted in grouped['fit'].items():
        durations = [row['duration_us'] for row in fitted]
        center = median(durations)
        mad = median(abs(value-center) for value in durations)
        both = fitted + grouped['calibration'][key]
        work = [workload_diagnostics(row) for row in both]
        fixed = {}
        for name in ('shared_tokens', 'dense_tokens'):
            values = {item[name] for item in work}
            if len(values) != 1:
                raise ValueError('nonconstant fixed work; inspect model/input configuration')
            fixed[name] = values.pop()
        model['baselines'][key] = dict(median_us=center, mad_us=mad, scale_us=max(1., 1.4826*mad),
            count=len(fitted), calibration_count=len(grouped['calibration'][key]),
            fixed_work=fixed, routed_token_domain=[min(item['routed_tokens'] for item in work),
                                                  max(item['routed_tokens'] for item in work)])
    calibration = []
    maximum = -math.inf
    for layout, sources in sorted(layouts.items()):
        source = sources['calibration']
        for start in range(source['start'], source['end'] + 1, 8):
            window = [row for row in rows['calibration'] if row['stage_metadata']['layout'] == layout
                      and start <= row['iteration'] < start + 8]
            score = _score_window(window, model, expected)
            if score['reasons'] or set(score['rank_scores']) != set(expected):
                raise ValueError('healthy calibration has incomplete comparable population')
            for value in score['rank_scores'].values():
                maximum = max(maximum, value)
            calibration.append(dict(layout=layout, start=start, end=start+7, **score))
    model['max_healthy_calibration_score'] = maximum
    model['min_score'] = max(MIN_SCORE, maximum + MIN_MARGIN)
    model['calibration_windows'] = len(calibration)
    model['sha256'] = _hash(model)
    audit = dict(status='healthy_calibrated_not_independently_validated',
        protocol=manifest.get('protocol', 'fit32_cal16_independent_v1'),
        by_layout=audits, calibration=calibration,
        model_sha256=model['sha256'], min_score=model['min_score'], margin=model['margin'],
        limitations=['calibration also selects supported work envelope; it is not an independent validation set',
            'fixed8-window rows and TP/DP views are correlated; not independent sample counts',
            'only planned held-out healthy segments can assess future coverage/false positives',
            'whole-stage aggregation may dilute component/operator delays below residual noise'])
    return model, audit


def verify_model(model):
    if model.get('model_type') != MODEL_TYPE or model.get('schema_version') != 1:
        raise ValueError('unsupported stage auxiliary model')
    if model.get('sha256') != _hash({key: value for key, value in model.items() if key != 'sha256'}):
        raise ValueError('stage auxiliary model hash mismatch')
    if (model.get('min_score', 0) < MIN_SCORE or model.get('margin', 0) < MIN_MARGIN or
            model.get('min_support') != MIN_SUPPORT or model.get('window_size') != 8):
        raise ValueError('invalid auxiliary evidence thresholds')
    for reference in model['baselines'].values():
        if not _finite(reference['median_us']) or not _finite(reference['scale_us'], positive=True):
            raise ValueError('invalid auxiliary healthy baseline')


def infer(rows, model, expected_ranks):
    if model is None:
        return dict(status='rejected', rank=None, reason='stage_aux_model_not_frozen')
    verify_model(model)
    if len(set(expected_ranks)) != model['world_size']:
        return dict(status='rejected', rank=None, reason='stage_aux_world_size_mismatch')
    result = _score_window(list(rows), model, expected_ranks)
    result.update(status='rejected', rank=None, reason='insufficient_or_ambiguous_stage_aux',
                  frozen_model_sha256=model['sha256'])
    if result['reasons']:
        result['reason'] = result['reasons'][0]
        return result
    ordered = sorted(result['rank_scores'], key=result['rank_scores'].get, reverse=True)
    if len(ordered) < 2:
        result['reason'] = 'fewer_than_two_comparable_ranks'
        return result
    best, second = ordered[:2]
    peers = [rank for rank in ordered if rank != best and
        result['support'][rank]['semantic_group_sha256'] == result['support'][best]['semantic_group_sha256']]
    result.update(candidate=best, runner_up=second, comparable_same_semantic_peers=peers)
    if not peers:
        result['reason'] = 'candidate_lacks_comparable_same_semantic_peer'
    elif (result['rank_scores'][best] > model['min_score'] and
          result['rank_scores'][best] - result['rank_scores'][second] > model['margin']):
        result.update(status='localized', rank=best, reason='unique_calibrated_stage_forward_residual')
    return result


def context_binding(model, instrumentation_context, provenance):
    """External immutable binding, leaving the scoped 764c model untouched."""
    verify_model(model)
    if instrumentation_context not in ('plain', 'scoped_dispatch') or not provenance:
        raise ValueError('explicit context and attested provenance required')
    required = ('plain' if model['healthy_manifest'].get('protocol') == 'plain_cost_fit8_cal8_v1'
                else 'scoped_dispatch')
    if instrumentation_context != required:
        raise ValueError('context is incompatible with the frozen model cohort protocol')
    binding = dict(binding_type='flash_stage_forward_context_binding_v1',
        model_sha256=model['sha256'], instrumentation_context=instrumentation_context,
        provenance=provenance)
    binding['sha256'] = _hash(binding)
    return binding


def infer_bound(rows, model, expected_ranks, binding):
    """Runtime entry: validate static all-rank observer context before scoring."""
    verify_model(model)
    if (binding.get('binding_type') != 'flash_stage_forward_context_binding_v1' or
            binding.get('sha256') != _hash({key: value for key, value in binding.items() if key != 'sha256'}) or
            binding.get('model_sha256') != model['sha256'] or
            binding.get('instrumentation_context') not in ('plain', 'scoped_dispatch')):
        raise ValueError('invalid auxiliary model context binding')
    required = ('plain' if model['healthy_manifest'].get('protocol') == 'plain_cost_fit8_cal8_v1'
                else 'scoped_dispatch')
    if binding['instrumentation_context'] != required:
        raise ValueError('bound context is incompatible with the frozen model cohort protocol')
    rows = list(rows)
    observed = [row.get('stage_metadata', {}).get('instrumentation_context', 'missing')
                if isinstance(row, dict) else 'malformed_row' for row in rows]
    expected = binding['instrumentation_context']
    if not observed or any(value != expected for value in observed):
        return dict(status='rejected', rank=None, reason='incompatible_or_missing_instrumentation_context',
            expected_instrumentation_context=expected, observed_instrumentation_contexts=sorted({str(value) for value in observed}),
            frozen_model_sha256=model['sha256'], binding_sha256=binding['sha256'])
    result = infer(rows, model, expected_ranks)
    result.update(instrumentation_context=expected, binding_sha256=binding['sha256'])
    return result


def fallback(original, rows, model, expected_ranks):
    """A successful frozen dc89 decision is returned untouched, by identity."""
    if original.get('status') == 'localized':
        return original
    auxiliary = infer(rows, model, expected_ranks)
    if auxiliary.get('status') == 'localized':
        return auxiliary
    return dict(original, stage_aux=auxiliary)
