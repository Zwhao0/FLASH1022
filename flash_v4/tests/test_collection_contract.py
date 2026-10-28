import copy
import tempfile
import unittest
from pathlib import Path

from flash_v4.budget import ProfileBudget, encode_observation, selected
from flash_v4.protocol import (
    BASELINE_FILES,
    BLIND_TARGET,
    BUDGET,
    FAULT_STRATA,
    MPS_QUOTA_REQUESTS,
    make_protocol,
    validate,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in BASELINE_FILES:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('baseline fixture')
        self.plan = make_protocol(20261027, root)

    def test_mps_matrix_uses_independent_fault_sessions_and_bounded_budget(self):
        self.assertEqual(len(self.plan['sessions']), BUDGET['max_training_sessions'])
        self.assertEqual(sum(len(s['fault_cases']) for s in self.plan['sessions']), 36)
        healthy = [s for s in self.plan['sessions'] if not s['fault_cases']]
        self.assertEqual(len(healthy), 12)
        self.assertEqual(sum(s['healthy_windows'] for s in healthy), 60)
        fault_sessions = [s for s in self.plan['sessions'] if s['fault_cases']]
        self.assertEqual(len(fault_sessions), 36)
        self.assertTrue(all(len(s['fault_cases']) == 1 for s in fault_sessions))
        self.assertEqual(len({s['workload_seed'] for s in self.plan['sessions']}), 48)
        self.assertTrue(all(s['mps_mode'] == 'shared' for s in self.plan['sessions']))
        self.assertFalse(self.plan['blind_schedule_generated'])
        self.assertEqual(self.plan['blind_target'], BLIND_TARGET)
        self.assertFalse(self.plan['blind_target']['schedule_generated'])
        self.assertEqual({name: len(slots) for name, slots in
                          self.plan['partitions_by_physical_slot'].items()},
                         {'development': 4, 'validation': 6,
                          'blind_only_fault_devices': 6})
        self.assertEqual(self.plan['budget']['max_active_collection_seconds'],
                         36 * 180 + 12 * 5 * 60)
        self.assertIn('not a measured slowdown or error severity',
                      self.plan['severity_semantics'])

    def test_quota_requests_distinguish_development_from_unseen_validation(self):
        self.assertEqual(MPS_QUOTA_REQUESTS['development'],
                         {'light': 75.0, 'medium': 50.0, 'strong': 25.0})
        self.assertEqual(MPS_QUOTA_REQUESTS['validation'],
                         {'light': 88.0, 'medium': 63.0, 'strong': 38.0})
        self.assertTrue(set(MPS_QUOTA_REQUESTS['development'].values()).isdisjoint(
                        MPS_QUOTA_REQUESTS['validation'].values()))
        self.assertEqual({s['healthy_mps_quota_percent'] for s in self.plan['sessions']
                          if s['condition'] == 'healthy'}, {100.0})
        for s in self.plan['sessions']:
            for case in s['fault_cases']:
                self.assertEqual(case['mechanism'], 'mps_quota')
                self.assertEqual(case['quota_scope'], 'target_rank_process')
                self.assertEqual(case['apply'], 'static_at_client_start')
                self.assertIn(case['fault_stratum'], FAULT_STRATA)

    def test_device_leakage_is_rejected(self):
        p = copy.deepcopy(self.plan)
        session = next(s for s in p['sessions'] if s['fault_cases'])
        session['fault_cases'][0]['physical_slot'] = p['partitions_by_physical_slot']['validation'][0]
        with self.assertRaisesRegex(ValueError, 'holdout'):
            validate(p)

    def test_multiple_quota_settings_in_one_session_are_rejected(self):
        p = copy.deepcopy(self.plan)
        session = next(s for s in p['sessions'] if s['fault_cases'])
        session['fault_cases'].append(copy.deepcopy(session['fault_cases'][0]))
        with self.assertRaisesRegex(ValueError, 'independent session'):
            validate(p)

    def test_clock_mode_cannot_be_confused_with_mps_quota(self):
        p = copy.deepcopy(self.plan)
        p['injection_mechanism'] = 'clock'
        with self.assertRaisesRegex(ValueError, 'shared mps_quota'):
            validate(p)

    def test_seed_reuse_and_premature_blind_draw_are_rejected(self):
        p = copy.deepcopy(self.plan)
        p['sessions'][1]['workload_seed'] = p['sessions'][0]['workload_seed']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate(p)
        self.plan['blind_schedule_generated'] = True
        with self.assertRaisesRegex(ValueError, 'blind'):
            validate(self.plan)


class ObservationTests(unittest.TestCase):
    def row(self, iteration=10):
        return dict(kind='computation', rank=0, iteration=iteration,
                    op='gemm', dtype='bf16', shape=[16, 512, 512], elapsed_us=10.)

    def test_truth_fields_and_hidden_nested_payload_are_rejected(self):
        for extra in ({'fault_rank': 0}, {'clock_mhz': 900},
                      {'shape': {'truth': 0}}, {'elapsed_us': float('nan')}):
            with self.assertRaises(ValueError):
                encode_observation(self.row() | extra)

    def test_hard_byte_limit_stays_closed_after_overflow(self):
        row = self.row()
        size = len(encode_observation(row))
        b = ProfileBudget(10, 0., byte_limit=size)
        self.assertIsNotNone(b.consume(row, 1.))
        self.assertIsNone(b.consume(row, 2.))
        self.assertEqual(b.bytes_written, size)
        self.assertEqual(b.stop_reason, 'byte_budget')
        self.assertIsNone(b.consume(row, 3.))
        self.assertFalse(b.summary()['coverage_verified'])

    def test_eighth_iteration_is_allowed_ninth_is_not(self):
        b = ProfileBudget(10, 0.)
        self.assertIsNotNone(b.consume(self.row(17), 1.))
        self.assertIsNone(b.consume(self.row(18), 2.))
        self.assertEqual(b.stop_reason, 'iteration_budget')

    def test_deadline_and_oversized_event(self):
        b = ProfileBudget(10, 0.)
        self.assertIsNone(b.consume(self.row(), 30.))
        self.assertEqual(b.stop_reason, 'time_budget')
        with self.assertRaisesRegex(ValueError, 'serialization'):
            encode_observation(self.row() | {'op': 'x' * 1024})

    def test_paired_sampling_does_not_use_rank_identity(self):
        keys = [f'group0:allreduce:bf16:4096:{i}' for i in range(128)]
        self.assertEqual([selected(k) for k in keys], [selected(k) for k in keys])
        self.assertTrue(any(selected(k) for k in keys))
        self.assertTrue(any(not selected(k) for k in keys))
        with self.assertRaises(ValueError):
            selected('key', 0)


if __name__ == '__main__':
    unittest.main()
