"""Mechanism checks: topology isolation, rank symmetry, persistence, abstention."""
import copy
import math
import unittest

from flash_v4.health_centered_fusion import calibrate, decide


def prediction(topology, role='healthy_repeat_h2'):
    return {
        'batch_case': {'case_key': f'{topology}_development_01', 'topology': topology,
                       'reference_role': 'development_H1', 'condition_role': role},
        'branches': {branch: {'rank_order': [
            {'rank': rank, 'score_by_iteration': {
                str(i): math.exp((i - 3.5) * 0.01) for i in range(8)}}
            for rank in range(16)]} for branch in ('gemm', 'kernel')},
    }


class CalibrationMechanismTests(unittest.TestCase):
    def setUp(self):
        self.healthy = [prediction(t) for t in ('legacy', 'node_local_pp')]
        self.calibration = calibrate(self.healthy)

    def test_positive_compute_evidence_survives_a_negative_other_branch(self):
        p = prediction('legacy', 'fault_condition')
        for branch, factor in (('kernel', 2.0), ('gemm', 0.5)):
            p['branches'][branch]['rank_order'][6]['score_by_iteration'] = {
                str(i): factor for i in range(8)}
        self.assertEqual(decide(p, self.calibration)['alarm_rank'], 6)

    def test_rank_relabeling_preserves_localization(self):
        p = prediction('legacy', 'fault_condition')
        p['branches']['kernel']['rank_order'][6]['score_by_iteration'] = {
            str(i): 2.0 for i in range(8)}
        permuted = copy.deepcopy(p)
        for branch in permuted['branches'].values():
            for row in branch['rank_order']:
                row['rank'] = (row['rank'] + 5) % 16
        self.assertEqual(decide(permuted, self.calibration)['alarm_rank'], 11)

    def test_another_topology_cannot_change_legacy_calibration(self):
        changed = copy.deepcopy(self.healthy)
        for branch in changed[1]['branches'].values():
            for row in branch['rank_order']:
                row['score_by_iteration'] = {str(i): math.exp(row['rank'] + i) for i in range(8)}
        separate = calibrate(changed)
        self.assertEqual(separate['by_topology']['legacy'],
                         self.calibration['by_topology']['legacy'])

    def test_insufficient_iterations_and_transient_spikes_cannot_alarm(self):
        for count in (5, 8):
            p = prediction('legacy', 'fault_condition')
            p['branches']['kernel']['rank_order'][6]['score_by_iteration'] = {
                str(i): (2.0 if i < 5 else 0.9) for i in range(count)}
            self.assertIsNone(decide(p, self.calibration)['alarm_rank'])

    def test_equal_sustained_candidates_are_rejected(self):
        p = prediction('legacy', 'fault_condition')
        for rank in (6, 7):
            p['branches']['kernel']['rank_order'][rank]['score_by_iteration'] = {
                str(i): 2.0 for i in range(8)}
        result = decide(p, self.calibration)
        self.assertIsNone(result['alarm_rank'])
        self.assertIn('ambiguous_rank_margin', result['abstention_reasons'])

    def test_no_hardcoded_fault_rank_or_validation_inputs(self):
        for rank in range(16):
            p = prediction('legacy', 'fault_condition')
            p['branches']['kernel']['rank_order'][rank]['score_by_iteration'] = {
                str(i): 2.0 for i in range(8)}
            self.assertEqual(decide(p, self.calibration)['alarm_rank'], rank)
        invalid = prediction('legacy')
        invalid['batch_case']['reference_role'] = 'validation_H1'
        with self.assertRaisesRegex(ValueError, 'development'):
            decide(invalid, self.calibration)


if __name__ == '__main__':
    unittest.main()
