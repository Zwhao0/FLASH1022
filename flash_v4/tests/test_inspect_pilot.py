import json
import tempfile
import unittest
from pathlib import Path

from flash_v4.inspect_pilot import MAX_RANK_BYTES, inspect_pilot, write_report


def observation(kind, rank, **extra):
    row = {
        'kind': kind,
        'rank': rank,
        'iteration': 4,
        'sequence': 0,
        'stream': 0,
        'comm_id': '0123456789abcdef',
        'op': 0,
        'numel': 64,
        'dtype': 1,
        'dtype_b': -1 if kind == 'allreduce' else 1,
        'dtype_c': -1 if kind == 'allreduce' else 1,
        'dtype_d': -1 if kind == 'allreduce' else 1,
        'compute': -1 if kind == 'allreduce' else 0,
        'batch': 0 if kind == 'allreduce' else 1,
        'shape': [2, 4, 8] if kind != 'allreduce' else [64, 0, 0],
        'elapsed_us': 10.0,
        'group': 0,
        'timing_scope': 'library_call_envelope',
        'skip': '',
    }
    row.update(extra)
    return row


def summary(rank, dropped=0):
    return {
        'kind': 'summary', 'rank': rank,
        'profile_seen': 2, 'sampled': 1, 'skipped': 0, 'dropped': dropped,
        'emitted': 2, 'pending_peak': 1, 'data_bytes': 100,
        'stride': 16, 'profile_iterations': 1, 'max_iterations': 8,
        'max_seconds': 30, 'max_bytes': MAX_RANK_BYTES,
        'blas_sampling': 'per_iteration_library_shape_dtypes_compute_transpose_batch_ordinal',
        'comparison': 'like_distributions_only;not_matched_operator_instances',
        'coverage': 'GemmEx,LtMatmul,blocking_AllReduce_only;no_graphs;no_kernel_coverage',
    }


def communicator(rank, comm_rank):
    return {'kind': 'communicator', 'rank': rank, 'comm_id': '0123456789abcdef',
            'comm_rank': comm_rank, 'members': 2, 'blocking': True,
            'mapping': 'join_global_rank_records_by_comm_id'}


def setup():
    return {'mode': 'profile', 'scheduling': 'fixed_offline_window_not_automatic_detection',
            'first_iteration': 4, 'last_iteration': 11,
            'timing_scope': 'sampled_library_envelopes_not_pure_kernel_execution',
            'comparison': 'matched_shapes_and_backend_distributions_not_operator_identity'}


class PilotInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def node_observations(self, node):
        directory = self.root / node / 'observations'
        directory.mkdir(parents=True)
        return directory

    def write_jsonl(self, path, *rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def test_counts_skips_coverage_comparability_and_communicators(self):
        d0 = self.node_observations('node0')
        d1 = self.node_observations('node1')
        self.write_jsonl(d0 / 'rank_0.jsonl',
                         observation('gemm', 0, sequence=0),
                         observation('allreduce', 0, elapsed_us=None,
                                     skip='event_query_error'),
                         communicator(0, 0), summary(0, dropped=1))
        self.write_jsonl(d1 / 'rank_1.jsonl',
                         observation('gemm', 1, sequence=32),
                         communicator(1, 1), summary(1))
        (d0 / 'setup_rank0.json').write_text(json.dumps(setup()), encoding='utf-8')
        (d1 / 'setup_rank1.json').write_text(json.dumps(setup()), encoding='utf-8')

        report = inspect_pilot(self.root)
        self.assertEqual(report['quality']['valid_timing_sample_count'], 2)
        self.assertEqual(report['quality']['sample_evidence'], 'valid_timing_samples_present')
        self.assertEqual(report['comparability']['key_fields'],
                         ['kind', 'op', 'dtype', 'shape'])
        self.assertFalse(report['comparability']['sequence_used_for_matching'])
        self.assertEqual(report['comparability']['multi_rank_cohort_count'], 1)
        self.assertEqual(report['communicators']['members_match_count'], 1)
        rank0 = report['ranks'][0]
        self.assertEqual(rank0['raw_bytes_limit_status'], 'within_limit')
        self.assertEqual(rank0['timing']['allreduce']['valid_elapsed'], 0)
        self.assertEqual(rank0['timing']['allreduce']['skip_rows'], 1)
        self.assertEqual(rank0['timing']['allreduce']['null_elapsed'], 1)
        self.assertEqual(rank0['summary']['reported_values']['dropped'], [1])
        self.assertEqual(report['files']['setup']['profile_window_status'], 'consistent')

    def test_skip_only_is_no_valid_sample(self):
        directory = self.node_observations('node0')
        self.write_jsonl(directory / 'rank_0.jsonl',
                         observation('gemm', 0, elapsed_us=None, skip='event_create_error'),
                         summary(0))
        (directory / 'setup_rank0.json').write_text(json.dumps(setup()), encoding='utf-8')

        report = inspect_pilot(self.root)
        self.assertEqual(report['quality']['valid_timing_sample_count'], 0)
        self.assertEqual(report['quality']['sample_evidence'], 'no_valid_timing_samples')
        self.assertEqual(report['comparability']['cohort_count'], 0)
        self.assertEqual(report['ranks'][0]['timing']['gemm']['skip_rows'], 1)
        self.assertEqual(report['ranks'][0]['timing']['gemm']['valid_elapsed'], 0)

    def test_iteration_presence_cannot_certify_deadline_truncated_or_legacy_window(self):
        directory = self.node_observations('node0')
        (directory / 'setup_rank0.json').write_text(json.dumps(setup()), encoding='utf-8')
        observations = [observation('gemm', 0, iteration=iteration) for iteration in range(4, 12)]
        for fields, status in (
                ({'completed_profile_iterations': 7, 'window_complete': False,
                  'stop_reason': 'time_budget'}, 'incomplete'),
                ({'completed_profile_iterations': 8, 'window_complete': True,
                  'stop_reason': 'window_closed'}, 'complete'),
                ({'completed_profile_iterations': 8, 'window_complete': True,
                  'stop_reason': 'time_budget'}, 'incomplete'),
                ({}, 'unknown')):
            with self.subTest(status=status, fields=fields):
                record = summary(0)
                record.update(profile_iterations=8, **fields)
                self.write_jsonl(directory / 'rank_0.jsonl', *observations, record)
                coverage = inspect_pilot(self.root)['ranks'][0]['iteration_coverage']
                self.assertEqual(coverage['missing_expected_iterations'], [])
                self.assertEqual(coverage['window_completeness']['status'], status)
                self.assertEqual(coverage['coverage_verified'], status == 'complete')

    def test_over_limit_file_is_not_parsed_and_output_is_exclusive(self):
        directory = self.node_observations('node0')
        path = directory / 'rank_0.jsonl'
        path.write_bytes(b'x' * (MAX_RANK_BYTES + 1))
        # Files outside the two declared input patterns are ignored.
        (self.root / 'actuation').mkdir()
        (self.root / 'actuation' / 'secret.json').write_text('{"rank":99}')

        report = inspect_pilot(self.root)
        self.assertEqual(report['files']['observation_jsonl_count'], 1)
        self.assertFalse(report['files']['observation_jsonl'][0]['within_1_mib'])
        self.assertEqual(report['files']['observation_jsonl'][0]['json_status'],
                         'not_checked_over_limit')
        self.assertEqual(report['quality']['sample_evidence'], 'no_valid_timing_samples')

        output = self.root / 'report.json'
        write_report(report, output)
        with self.assertRaises(FileExistsError):
            write_report(report, output)


if __name__ == '__main__':
    unittest.main()
