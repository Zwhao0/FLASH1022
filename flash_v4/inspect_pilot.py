"""Inspect saved pilot observation quality without scoring or locating faults."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

MAX_RANK_BYTES = 1024 * 1024
LIBRARY_TIMING_KINDS = ('gemm', 'lt_matmul', 'allreduce')
KERNEL_KIND = 'kernel'
TIMING_KINDS = (*LIBRARY_TIMING_KINDS, KERNEL_KIND)
KNOWN_KINDS = frozenset((*TIMING_KINDS, 'communicator', 'summary'))
KERNEL_LAUNCH_APIS = frozenset((
    'runtime', 'runtime_ex', 'runtime_ptsz', 'runtime_ex_ptsz',
    'driver', 'driver_ex', 'driver_ptsz', 'driver_ex_ptsz',
))
KERNEL_SUMMARY_COUNTERS = (
    'candidates', 'sampled', 'emitted', 'dropped', 'skipped',
    'nccl', 'unknown', 'blas', 'nested',
)
KERNEL_ID_HASH_SCHEME = 'fnv1a64_device_name'
COMMON_TIMING_FIELDS = (
    'rank', 'iteration', 'sequence', 'stream', 'comm_id', 'op', 'numel',
    'dtype', 'dtype_b', 'dtype_c', 'dtype_d', 'compute', 'batch', 'shape',
    'elapsed_us', 'group', 'timing_scope', 'skip',
)
SUMMARY_COUNTERS = (
    'profile_seen', 'sampled', 'skipped', 'dropped', 'emitted', 'pending_peak',
    'data_bytes', 'stride', 'profile_iterations', 'max_iterations', 'max_seconds',
    'max_bytes',
)
COMPLETION_FIELDS = ('completed_profile_iterations', 'window_complete', 'stop_reason')


def _valid_completion(row):
    return (_is_int(row.get('completed_profile_iterations'))
            and row['completed_profile_iterations'] >= 0
            and isinstance(row.get('window_complete'), bool)
            and isinstance(row.get('stop_reason'), str) and bool(row['stop_reason']))
SETUP_FIELDS = (
    'mode', 'scheduling', 'first_iteration', 'last_iteration',
    'timing_scope', 'comparison',
)


def _is_int(value):
    return type(value) is int


def _is_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(value)


def _reject_constant(value):
    raise ValueError(f'non-standard JSON number: {value}')


def _load_json(line):
    return json.loads(line, parse_constant=_reject_constant)


def _kernel_summary_errors(value):
    if not isinstance(value, dict):
        return ['kernel']
    errors = []
    if not isinstance(value.get('enabled'), bool):
        errors.append('kernel.enabled')
    for field in KERNEL_SUMMARY_COUNTERS:
        if not _is_int(value.get(field)) or value[field] < 0:
            errors.append('kernel.' + field)
    if not _is_int(value.get('stride')) or value['stride'] != 256:
        errors.append('kernel.stride')
    if value.get('id_hash') != KERNEL_ID_HASH_SCHEME:
        errors.append('kernel.id_hash')
    counters_valid = all(_is_int(value.get(field)) and value[field] >= 0
                         for field in KERNEL_SUMMARY_COUNTERS)
    if counters_valid:
        if value['sampled'] > value['candidates']:
            errors.append('kernel.sampled_exceeds_candidates')
        if value['emitted'] + value['dropped'] != value['sampled']:
            errors.append('kernel.sampled_accounting')
        if value['skipped'] > value['emitted']:
            errors.append('kernel.skipped_exceeds_emitted')
    return errors


def _empty_timing():
    return {
        'rows': 0,
        'successful_rows': 0,
        'valid_elapsed': 0,
        'skip_rows': 0,
        'null_elapsed': 0,
        'invalid_elapsed': 0,
        'schema_invalid': 0,
    }


def _empty_rank():
    return {
        'rows_by_kind': Counter(),
        'schema_errors_by_kind': Counter(),
        'timing': {kind: _empty_timing() for kind in TIMING_KINDS},
        'kernel_sampling_strides': Counter(),
        'iterations': set(),
        'summary_count': 0,
        'summary_invalid_count': 0,
        'summary_values': defaultdict(list),
        'summary_completions': [],
        'kernel_summary_records': [],
        'summary_missing_fields': set(),
        'communicator_records': [],
        'observation_files': [],
        'mixed_rank_file': False,
    }


def _row_schema_errors(row, kind):
    errors = []
    if not _is_int(row.get('rank')) or row['rank'] < 0:
        errors.append('rank')
    if kind == KERNEL_KIND:
        for field in ('iteration', 'sequence'):
            if not _is_int(row.get(field)) or row[field] < 0:
                errors.append(field)
        if not isinstance(row.get('kernel_id'), str) or not re.fullmatch(
                r'[0-9a-fA-F]{16}', row['kernel_id']):
            errors.append('kernel_id')
        for field in ('grid', 'block'):
            value = row.get(field)
            if (not isinstance(value, list) or len(value) != 3
                    or any(not _is_int(part) or part <= 0 for part in value)):
                errors.append(field)
        if (not _is_int(row.get('dynamic_shared_bytes'))
                or row['dynamic_shared_bytes'] < 0):
            errors.append('dynamic_shared_bytes')
        if (not isinstance(row.get('launch_api'), str)
                or row['launch_api'] not in KERNEL_LAUNCH_APIS):
            errors.append('launch_api')
        if (not _is_int(row.get('sampling_stride'))
                or row['sampling_stride'] != 256):
            errors.append('sampling_stride')
        elif (_is_int(row.get('iteration')) and row['iteration'] >= 0
              and _is_int(row.get('sequence')) and row['sequence'] >= 0
              and row['sequence'] % row['sampling_stride']
              != row['iteration'] % row['sampling_stride']):
            errors.append('sampling_rule')
        if not isinstance(row.get('skip'), str):
            errors.append('skip')
        elapsed = row.get('elapsed_us', 'missing')
        skip = isinstance(row.get('skip'), str) and bool(row.get('skip'))
        if elapsed == 'missing':
            errors.append('elapsed_us')
        elif elapsed is None:
            if not skip:
                errors.append('null_elapsed_without_skip')
        elif not _is_number(elapsed) or elapsed < 0:
            errors.append('elapsed_us')
    elif kind in LIBRARY_TIMING_KINDS:
        for field in ('iteration', 'sequence', 'stream', 'numel', 'dtype', 'op', 'group'):
            if not _is_int(row.get(field)) or row[field] < 0:
                errors.append(field)
        extended_fields = ('dtype_b', 'dtype_c', 'dtype_d', 'compute', 'batch')
        if kind in ('gemm', 'lt_matmul'):
            for field in extended_fields:
                if not _is_int(row.get(field)) or row[field] < 0:
                    errors.append(field)
        else:
            # Native AllReduce rows carry -1 sentinels for BLAS-only metadata.
            for field in extended_fields:
                if not _is_int(row.get(field)):
                    errors.append(field)
        shape = row.get('shape')
        if (not isinstance(shape, list) or len(shape) != 3
                or any(not _is_int(part) or part < 0 for part in shape)):
            errors.append('shape')
        if not isinstance(row.get('comm_id'), str):
            errors.append('comm_id')
        if not isinstance(row.get('timing_scope'), str):
            errors.append('timing_scope')
        if not isinstance(row.get('skip'), str):
            errors.append('skip')
        elapsed = row.get('elapsed_us', 'missing')
        skip = isinstance(row.get('skip'), str) and bool(row.get('skip'))
        if elapsed == 'missing':
            errors.append('elapsed_us')
        elif elapsed is None:
            if not skip:
                errors.append('null_elapsed_without_skip')
        elif not _is_number(elapsed) or elapsed < 0:
            errors.append('elapsed_us')
    elif kind == 'communicator':
        if not isinstance(row.get('comm_id'), str) or not row['comm_id']:
            errors.append('comm_id')
        for field in ('comm_rank', 'members'):
            if not _is_int(row.get(field)) or row[field] < 0:
                errors.append(field)
        if _is_int(row.get('members')) and row['members'] <= 0:
            errors.append('members')
        if not isinstance(row.get('blocking'), bool):
            errors.append('blocking')
        if not isinstance(row.get('mapping'), str):
            errors.append('mapping')
    elif kind == 'summary':
        # Legacy evidence remains readable, but missing explicit schedule-end
        # certification can never establish a complete collection window.
        if any(field in row for field in COMPLETION_FIELDS) and not _valid_completion(row):
            errors.append('completion_fields')
        if 'kernel' in row:
            errors.extend(_kernel_summary_errors(row['kernel']))
        for field in SUMMARY_COUNTERS:
            value = row.get(field)
            if field == 'max_seconds':
                valid = _is_number(value) and value > 0
            else:
                valid = _is_int(value) and value >= 0
            if not valid:
                errors.append(field)
        for field in ('blas_sampling', 'comparison', 'coverage'):
            if not isinstance(row.get(field), str):
                errors.append(field)
    return errors


def _setup_record(path, index):
    record = {'index': index, 'raw_bytes': path.stat().st_size,
              'json_status': 'invalid', 'schema_errors': [], 'mode': None,
              'profile_window': None}
    try:
        value = _load_json(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        record['json_status'] = 'invalid'
        return record
    if not isinstance(value, dict):
        record['json_status'] = 'non_object'
        return record
    record['json_status'] = 'valid'
    record['mode'] = value.get('mode') if isinstance(value.get('mode'), str) else None
    record['schema_errors'] = [field for field in SETUP_FIELDS if field not in value]
    if record['mode'] not in ('profile', 'monitor'):
        record['schema_errors'].append('mode_value')
    first, last = value.get('first_iteration'), value.get('last_iteration')
    if (not _is_int(first) or not _is_int(last) or last < first
            or last - first + 1 != 8):
        record['schema_errors'].append('profile_window')
    else:
        record['profile_window'] = [first, last]
    for field in ('scheduling', 'timing_scope', 'comparison'):
        if not isinstance(value.get(field), str):
            record['schema_errors'].append(field + '_type')
    return record


def _observation_file(path, index, ranks, cohort_rows, communicators):
    size = path.stat().st_size
    result = {
        'index': index,
        'raw_bytes': size,
        'within_1_mib': size <= MAX_RANK_BYTES,
        'physical_lines': 0,
        'valid_json_objects': 0,
        'invalid_json_lines': 0,
        'blank_lines': 0,
        'non_object_json_lines': 0,
        'reported_ranks': [],
        'json_status': 'not_checked_over_limit' if size > MAX_RANK_BYTES else 'empty',
    }
    if size > MAX_RANK_BYTES:
        return result
    try:
        data = path.read_bytes()
        if len(data) > MAX_RANK_BYTES:
            result['raw_bytes'] = len(data)
            result['within_1_mib'] = False
            result['json_status'] = 'not_checked_over_limit'
            return result
        text = data.decode('utf-8')
    except (OSError, UnicodeError):
        result['json_status'] = 'invalid_utf8_or_unreadable'
        return result

    reported_ranks = set()
    for line in text.splitlines():
        result['physical_lines'] += 1
        if not line.strip():
            result['blank_lines'] += 1
            result['invalid_json_lines'] += 1
            continue
        try:
            row = _load_json(line)
        except (json.JSONDecodeError, ValueError):
            result['invalid_json_lines'] += 1
            continue
        if not isinstance(row, dict):
            result['non_object_json_lines'] += 1
            continue
        result['valid_json_objects'] += 1
        kind = row.get('kind')
        rank = row.get('rank')
        if _is_int(rank) and rank >= 0:
            reported_ranks.add(rank)
            rank_data = ranks[rank]
            if isinstance(kind, str):
                rank_data['rows_by_kind'][kind] += 1
            else:
                rank_data['schema_errors_by_kind']['missing_kind'] += 1

            if kind in TIMING_KINDS:
                timing = rank_data['timing'][kind]
                timing['rows'] += 1
                errors = _row_schema_errors(row, kind)
                if errors:
                    timing['schema_invalid'] += 1
                    rank_data['schema_errors_by_kind'][kind] += len(errors)
                elif kind == KERNEL_KIND:
                    rank_data['kernel_sampling_strides'][row['sampling_stride']] += 1
                skip = isinstance(row.get('skip'), str) and bool(row.get('skip'))
                elapsed = row.get('elapsed_us', 'missing')
                if skip:
                    timing['skip_rows'] += 1
                if elapsed is None:
                    timing['null_elapsed'] += 1
                elif elapsed == 'missing' or not _is_number(elapsed) or elapsed < 0:
                    timing['invalid_elapsed'] += 1
                if _is_number(elapsed) and elapsed >= 0 and not skip:
                    timing['successful_rows'] += 1
                if _is_int(row.get('iteration')) and row['iteration'] >= 0:
                    rank_data['iterations'].add(row['iteration'])
                valid_elapsed = (_is_number(elapsed) and elapsed >= 0 and not skip and not errors)
                if valid_elapsed:
                    timing['valid_elapsed'] += 1
                    if kind in LIBRARY_TIMING_KINDS:
                        signature = (kind, row['op'], row['dtype'], tuple(row['shape']))
                        cohort_rows[signature][rank] += 1
            elif kind == 'communicator':
                errors = _row_schema_errors(row, kind)
                if errors:
                    rank_data['schema_errors_by_kind'][kind] += len(errors)
                elif row['members'] > 0 and row['comm_rank'] < row['members']:
                    item = {
                        'comm_id': row['comm_id'],
                        'rank': rank,
                        'comm_rank': row['comm_rank'],
                        'members': row['members'],
                    }
                    rank_data['communicator_records'].append(item)
                    communicators[row['comm_id']].append(item)
                else:
                    rank_data['schema_errors_by_kind'][kind] += 1
            elif kind == 'summary':
                errors = _row_schema_errors(row, kind)
                rank_data['summary_count'] += 1
                rank_data['summary_missing_fields'].update(errors)
                if errors:
                    rank_data['summary_invalid_count'] += 1
                if 'kernel' in row:
                    rank_data['kernel_summary_records'].append({
                        'value': row['kernel'] if isinstance(row['kernel'], dict) else None,
                        'schema_errors': _kernel_summary_errors(row['kernel']),
                    })
                if _valid_completion(row):
                    rank_data['summary_completions'].append({field: row.get(field) for field in
                        (*COMPLETION_FIELDS, 'profile_iterations', 'max_iterations', 'dropped')})
                for field in SUMMARY_COUNTERS:
                    value = row.get(field)
                    if (field == 'max_seconds' and _is_number(value) and value > 0) or (
                            field != 'max_seconds' and _is_int(value) and value >= 0):
                        rank_data['summary_values'][field].append(value)
            elif kind not in KNOWN_KINDS:
                rank_data['schema_errors_by_kind']['unknown_kind'] += 1
        else:
            if isinstance(kind, str):
                # The rank is unavailable, so retain a file-level schema boundary only.
                if kind not in KNOWN_KINDS:
                    result.setdefault('unknown_kinds', Counter())[kind] += 1

    result['reported_ranks'] = sorted(reported_ranks)
    if result['physical_lines'] == 0:
        result['json_status'] = 'empty_no_samples'
    elif (result['invalid_json_lines'] or result['blank_lines']
          or result['non_object_json_lines']):
        result['json_status'] = 'invalid_or_mixed'
    else:
        result['json_status'] = 'valid_json_lines'
    if 'unknown_kinds' in result:
        result['unknown_kinds'] = dict(result['unknown_kinds'])
    for rank in reported_ranks:
        ranks[rank]['observation_files'].append(result)
        if len(reported_ranks) > 1:
            ranks[rank]['mixed_rank_file'] = True
    return result


def _communicator_checks(communicators):
    checks = []
    for comm_id in sorted(communicators):
        records = communicators[comm_id]
        member_values = {item['members'] for item in records}
        ranks = [item['rank'] for item in records]
        comm_ranks = [item['comm_rank'] for item in records]
        members = next(iter(member_values)) if len(member_values) == 1 else None
        unique_ranks = set(ranks)
        rank_count_matches = members is not None and len(unique_ranks) == members
        comm_rank_complete = (members is not None
                              and set(comm_ranks) == set(range(members)))
        duplicates = len(ranks) - len(unique_ranks)
        consistent = len(member_values) == 1
        complete = bool(consistent and rank_count_matches and comm_rank_complete and not duplicates)
        checks.append({
            'comm_id': comm_id,
            'declared_members': members,
            'members_values_consistent': consistent,
            'global_rank_record_count': len(unique_ranks),
            'comm_rank_count': len(set(comm_ranks)),
            'duplicate_global_rank_records': duplicates,
            'global_rank_count_matches_members': rank_count_matches,
            'comm_rank_range_complete': comm_rank_complete,
            'membership_status': 'members_match' if complete else 'incomplete_or_inconsistent',
        })
    return checks


def _rank_output(rank, data, expected_iterations):
    files = data['observation_files']
    total_bytes = sum(item['raw_bytes'] for item in files)
    if not files:
        byte_status = 'unattributed'
    elif total_bytes > MAX_RANK_BYTES or any(not item['within_1_mib'] for item in files):
        byte_status = 'over_limit'
    elif len(files) != 1 or data['mixed_rank_file']:
        byte_status = 'ambiguous_multiple_or_mixed_rank_files'
    else:
        byte_status = 'within_limit'
    observed_iterations = sorted(data['iterations'])
    if expected_iterations is None:
        missing_iterations = None
    else:
        missing_iterations = sorted(set(expected_iterations) - data['iterations'])
    timing = data['timing']
    completion = {'status': 'unknown', 'completed_profile_iterations': None,
                  'stop_reason': None, 'evidence': 'missing_or_ambiguous_explicit_schedule_end_summary'}
    if data['summary_count'] == 1 and len(data['summary_completions']) == 1:
        item = data['summary_completions'][0]
        expected_count = len(expected_iterations) if expected_iterations is not None else 8
        complete = (item['window_complete'] is True
                    and item['completed_profile_iterations'] == expected_count == 8
                    and item['profile_iterations'] == item['max_iterations'] == 8
                    and item['dropped'] == 0 and data['summary_invalid_count'] == 0
                    and item['stop_reason'] in ('window_closed', 'iteration_budget')
                    and byte_status == 'within_limit')
        completion = {'status': 'complete' if complete else 'incomplete',
                      'completed_profile_iterations': item['completed_profile_iterations'],
                      'stop_reason': item['stop_reason'],
                      'evidence': 'explicit_native_schedule_end_summary'}
    observed_timing_by_kind = {
        kind: data['rows_by_kind'].get(kind, 0) for kind in TIMING_KINDS
    }
    observed_timing_total = sum(observed_timing_by_kind.values())
    observed_successful_timing_total = sum(
        timing[kind]['successful_rows'] for kind in TIMING_KINDS)
    reported_emitted = data['summary_values'].get('emitted', [])
    emitted_status = ('match' if len(reported_emitted) == 1
                      and reported_emitted[0] == observed_successful_timing_total
                      else 'mismatch' if len(reported_emitted) == 1 else 'unknown')
    kernel_rows = observed_timing_by_kind[KERNEL_KIND]
    kernel_records = data['kernel_summary_records']
    if not kernel_rows and not kernel_records:
        kernel_summary_status = 'not_reported_no_kernel_rows'
        kernel_summary = None
    elif not kernel_records:
        kernel_summary_status = 'missing'
        kernel_summary = None
    elif len(kernel_records) != 1:
        kernel_summary_status = 'ambiguous_multiple_summaries'
        kernel_summary = None
    else:
        kernel_record = kernel_records[0]
        kernel_summary = kernel_record['value']
        kernel_summary_status = (
            'valid' if kernel_summary is not None and not kernel_record['schema_errors']
            else 'invalid_schema')
    if (kernel_summary_status in ('valid', 'invalid_schema')
            and kernel_summary is not None and _is_int(kernel_summary.get('emitted'))):
        kernel_emitted_matches = kernel_summary['emitted'] == kernel_rows
    else:
        kernel_emitted_matches = None
    kernel_enabled_matches_rows = (
        (kernel_summary['enabled'] or not bool(kernel_rows)) if kernel_summary is not None
        and isinstance(kernel_summary.get('enabled'), bool) else None)
    kernel_sampling = {
        'summary_status': kernel_summary_status,
        'summary_counters': kernel_summary,
        'enabled_status_consistent_with_rows': kernel_enabled_matches_rows,
        'emitted_count_matches_observed_kernel_rows': kernel_emitted_matches,
        'sampling_stride_counts_in_rows': dict(sorted(data['kernel_sampling_strides'].items())),
        'sampling_rule': (
            'named non-NCCL, non-BLAS outer launch candidates; sampled when '
            'sequence % stride == public iteration % stride'),
        'kernel_id_hash_scheme': KERNEL_ID_HASH_SCHEME,
        'budget_reported': {
            'max_profile_iterations': data['summary_values'].get('max_iterations', []),
            'max_profile_seconds': data['summary_values'].get('max_seconds', []),
            'max_observation_bytes': data['summary_values'].get('max_bytes', []),
            'hard_rank_file_bytes': MAX_RANK_BYTES,
        },
    }
    return {
        'rank': rank,
        'observation_file_count': len(files),
        'raw_bytes_total': total_bytes,
        'raw_bytes_limit_status': byte_status,
        'rows_by_kind': dict(sorted(data['rows_by_kind'].items())),
        'schema_errors_by_kind': dict(sorted(data['schema_errors_by_kind'].items())),
        'timing': {kind: timing[kind] for kind in TIMING_KINDS},
        'kernel_sampling_stride_counts': dict(sorted(data['kernel_sampling_strides'].items())),
        'kernel_sampling': kernel_sampling,
        'iteration_coverage': {
            'observed_iterations': observed_iterations,
            'observed_count': len(observed_iterations),
            'expected_profile_iterations': expected_iterations,
            'missing_expected_iterations': missing_iterations,
            'coverage_verified': (completion['status'] == 'complete'
                                  and expected_iterations is not None and not missing_iterations),
            'window_completeness': completion,
        },
        'summary': {
            'record_count': data['summary_count'],
            'invalid_record_count': data['summary_invalid_count'],
            'timing_event_accounting': {
                'observed_records_by_kind': observed_timing_by_kind,
                'observed_timing_record_count': observed_timing_total,
                'observed_successful_timing_record_count': observed_successful_timing_total,
                'summary_emitted_semantics': 'successfully timed rows; skip rows excluded',
                'reported_emitted_values': reported_emitted,
                'status': emitted_status,
            },
            'reported_values': {key: values for key, values in
                                sorted(data['summary_values'].items())},
            'missing_or_invalid_fields': sorted(data['summary_missing_fields']),
        },
    }


def inspect_pilot(root):
    """Read only node*/observations/rank_*.jsonl and setup_rank*.json files."""
    root = Path(root)
    observation_paths = sorted(root.glob('node*/observations/rank_*.jsonl'))
    setup_paths = sorted(root.glob('node*/observations/setup_rank*.json'))
    ranks = defaultdict(_empty_rank)
    cohort_rows = defaultdict(Counter)
    communicators = defaultdict(list)

    observation_files = [
        _observation_file(path, index, ranks, cohort_rows, communicators)
        for index, path in enumerate(observation_paths)
    ]
    setup_files = [_setup_record(path, index) for index, path in enumerate(setup_paths)]

    valid_setup = [record for record in setup_files
                   if record['json_status'] == 'valid' and not record['schema_errors']]
    profile_windows = {tuple(record['profile_window']) for record in valid_setup
                       if record['mode'] == 'profile' and record['profile_window']}
    if len(profile_windows) == 1 and len(valid_setup) == len(setup_files):
        first, last = next(iter(profile_windows))
        expected_iterations = list(range(first, last + 1))
        setup_window_status = 'consistent'
    elif len(profile_windows) == 1:
        first, last = next(iter(profile_windows))
        expected_iterations = list(range(first, last + 1))
        setup_window_status = 'partially_observed_consistent'
    elif len(profile_windows) > 1:
        expected_iterations = None
        setup_window_status = 'inconsistent'
    else:
        expected_iterations = None
        setup_window_status = 'unavailable'

    comm_checks = _communicator_checks(communicators)
    multi_rank_by_kind = Counter()
    cohort_count_by_kind = Counter()
    for signature, per_rank in cohort_rows.items():
        kind = signature[0]
        cohort_count_by_kind[kind] += 1
        if len(per_rank) >= 2:
            multi_rank_by_kind[kind] += 1
    valid_timing_samples = sum(
        counts['valid_elapsed'] for data in ranks.values()
        for counts in data['timing'].values())
    invalid_json_lines = sum(record['invalid_json_lines'] for record in observation_files)
    non_object_lines = sum(record['non_object_json_lines'] for record in observation_files)
    unattributed_files = sum(not record['reported_ranks'] for record in observation_files)
    unknown_kinds = Counter()
    for data in ranks.values():
        for kind, count in data['rows_by_kind'].items():
            if kind not in KNOWN_KINDS:
                unknown_kinds[kind] += count
    for record in observation_files:
        unknown_kinds.update(record.get('unknown_kinds', {}))

    return {
        'schema': 'flash-v4-pilot-inspection-1',
        'scope': 'collection_quality_and_schema_only',
        'files': {
            'observation_jsonl_count': len(observation_files),
            'setup_json_count': len(setup_files),
            'unattributed_observation_file_count': unattributed_files,
            'observation_jsonl': observation_files,
            'setup': {
                'valid_json_count': sum(f['json_status'] == 'valid' for f in setup_files),
                'invalid_or_non_object_json_count': sum(f['json_status'] != 'valid'
                                                        for f in setup_files),
                'schema_invalid_count': sum(bool(f['schema_errors']) for f in setup_files),
                'mode_counts': dict(sorted(Counter(
                    f['mode'] for f in setup_files if f['mode'] is not None).items())),
                'profile_window_status': setup_window_status,
                'expected_profile_iterations': expected_iterations,
                'records': setup_files,
            },
        },
        'ranks': [_rank_output(rank, ranks[rank], expected_iterations)
                  for rank in sorted(ranks)],
        'communicators': {
            'join_key': 'comm_id',
            'rank_field_scope': 'observer RANK value (global rank)',
            'group_count': len(comm_checks),
            'members_match_count': sum(c['membership_status'] == 'members_match'
                                       for c in comm_checks),
            'incomplete_or_inconsistent_count': sum(
                c['membership_status'] != 'members_match' for c in comm_checks),
            'checks': comm_checks,
        },
        'comparability': {
            'key_fields': ['kind', 'op', 'dtype', 'shape'],
            'sequence_used_for_matching': False,
            'interpretation': 'distribution cohorts only; not operator-instance matching',
            'cohort_count': sum(cohort_count_by_kind.values()),
            'multi_rank_cohort_count': sum(multi_rank_by_kind.values()),
            'cohort_count_by_kind': dict(sorted(cohort_count_by_kind.items())),
            'multi_rank_cohort_count_by_kind': dict(sorted(multi_rank_by_kind.items())),
        },
        'quality': {
            'rank_count_with_records': len(ranks),
            'invalid_json_line_count': invalid_json_lines,
            'non_object_json_line_count': non_object_lines,
            'unknown_kinds': dict(sorted(unknown_kinds.items())),
            'valid_timing_sample_count': valid_timing_samples,
            'sample_evidence': ('valid_timing_samples_present' if valid_timing_samples
                                else 'no_valid_timing_samples'),
            'communicator_groups': len(comm_checks),
        },
    }


def write_report(report, output=None, stream=None):
    text = json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n'
    if output is None:
        (stream or sys.stdout).write(text)
        return
    with Path(output).open('x', encoding='utf-8') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot_dir', type=Path)
    parser.add_argument('--output', type=Path,
                        help='write compact JSON to a new file (existing files are never replaced)')
    args = parser.parse_args()
    write_report(inspect_pilot(args.pilot_dir), args.output)


if __name__ == '__main__':
    main()
