import unittest

from flash_v4.episode_qualification import qualify


class QualificationTests(unittest.TestCase):
    contract = dict(baseline_steps=30, block_steps=8, max_baseline_cv=.1,
                    min_increase=.3, max_increase=.5, min_block_increase=.25,
                    max_fault_seconds=30)
    window = dict(first=40, last=71)

    def rows(self):
        return [dict(iteration=i, time_ms=100 if i < 40 else 140,
                     wall_iteration_ms=100 if i < 40 else 140, walltime=i * .14)
                for i in range(10, 72)]

    def test_clean_fixed_episode_qualifies_without_alarm_or_target(self):
        result = qualify(self.rows(), self.window, self.contract)
        self.assertTrue(result['qualified'])
        self.assertEqual(result['measured_steps'], 32)

    def test_profile_missing_duplicate_or_invalid_timing_rejected(self):
        self.assertFalse(qualify(self.rows(), self.window, self.contract, [52, 59])['qualified'])
        for rows in [self.rows()[:-1], self.rows() + self.rows()[-1:]]:
            self.assertFalse(qualify(rows, self.window, self.contract)['qualified'])
        for bad in [float('nan'), float('inf'), 0, -1, True]:
            rows = self.rows()
            rows[-1]['time_ms'] = bad
            self.assertFalse(qualify(rows, self.window, self.contract)['qualified'])

    def test_weak_burst_cannot_pass_as_sustained_main_positive(self):
        rows = self.rows()
        for row in rows:
            if 40 <= row['iteration'] < 48:
                row.update(time_ms=100, wall_iteration_ms=100)
        result = qualify(rows, self.window, self.contract)
        self.assertFalse(result['qualified'])
        self.assertIn('time_ms_not_sustained', result['reasons'])

    def test_time_budget_and_unstable_baseline_rejected(self):
        rows = self.rows()
        rows[-1]['walltime'] += 30
        self.assertIn('episode_exceeds_time_budget', qualify(rows, self.window, self.contract)['reasons'])
        rows = self.rows()
        rows[0]['time_ms'] = 300
        self.assertIn('time_ms_unstable_baseline', qualify(rows, self.window, self.contract)['reasons'])

    def test_intermediate_timestamp_and_profile_metadata_must_be_valid(self):
        for bad in [None, float('nan'), float('inf'), 0, True]:
            rows = self.rows()
            rows[35]['walltime'] = bad
            self.assertFalse(qualify(rows, self.window, self.contract)['qualified'])
        for profile in [[], [50], [55, 50], [50, True], '52']:
            self.assertFalse(qualify(self.rows(), self.window, self.contract, profile)['qualified'])


if __name__ == '__main__':
    unittest.main()
