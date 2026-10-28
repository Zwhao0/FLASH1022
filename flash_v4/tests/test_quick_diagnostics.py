import unittest

from flash_v4.quick_diagnostics import timing_diagnostics


class TimingDiagnosisTests(unittest.TestCase):
    def rows(self, fault_ms=105):
        return [dict(iteration=i, time_ms=100 if i < 40 else fault_ms,
                     wall_iteration_ms=101 if i < 40 else fault_ms + 1) for i in range(1, 65)]

    def test_low_signal_is_a_hypothesis_not_a_correct_detection(self):
        result = timing_diagnostics(self.rows(), dict(first=40, last=63))
        self.assertIn('rank0_step_signal_below_v1_degradation_gate', result['flags'])
        self.assertAlmostEqual(result['relative_change']['time_ms']['mean'], .05)
        self.assertNotIn('outcome', result)

    def test_profile_overhead_is_excluded_from_monitor_signal(self):
        rows = self.rows()
        for row in rows:
            if 52 <= row['iteration'] <= 59:
                row['time_ms'] = 1000
        result = timing_diagnostics(rows, dict(first=40, last=63), 43, [52, 59])
        self.assertAlmostEqual(result['relative_change']['time_ms']['mean'], .05)
        self.assertEqual(result['planned_fault_profile_overlap_iterations'], 8)

    def test_strong_untriggered_signal_requires_detector_review(self):
        result = timing_diagnostics(self.rows(150), dict(first=40, last=63))
        self.assertIn('Rbeast_candidate_probability_or_filter_review_needed', result['flags'])

    def test_pretrigger_and_partial_window_are_reported(self):
        result = timing_diagnostics(self.rows(), dict(first=40, last=55), 35, [52, 59])
        self.assertIn('monitor_requested_before_injection', result['flags'])
        self.assertIn('profile_only_partially_overlapped_fault_episode', result['flags'])
        self.assertEqual(result['planned_fault_profile_overlap_iterations'], 4)


if __name__ == '__main__':
    unittest.main()
