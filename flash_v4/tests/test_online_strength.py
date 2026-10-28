import unittest
from flash_v4.online_strength import check


class StrengthTests(unittest.TestCase):
    contract = dict(baseline_steps=30, block_steps=8, min_clean_blocks=2, max_baseline_cv=.1,
                    min_increase=.3, max_increase=.5, min_block_increase=.25, max_fault_seconds=30)
    window = dict(first=40, last=71)

    def rows(self):
        return [dict(iteration=i, time_ms=100 if i < 40 else 140,
                     wall_iteration_ms=100 if i < 40 else 140, walltime=i * .14) for i in range(10, 72)]

    def test_profile_cost_never_enters_strength(self):
        rows = self.rows()
        for row in rows:
            if 52 <= row['iteration'] <= 59:
                row.update(time_ms=1000, wall_iteration_ms=1000)
        result = check(rows, self.window, self.contract, [52, 59])
        self.assertTrue(result['qualified'])
        self.assertEqual(result['clean_blocks'], [[40, 47], [64, 71]])
        self.assertEqual(result['measured_unprofiled_steps'], 16)

    def test_untriggered_weak_signal_is_not_validated_by_no_profile(self):
        rows = self.rows()
        for row in rows:
            if row['iteration'] >= 40:
                row.update(time_ms=105, wall_iteration_ms=105)
        self.assertFalse(check(rows, self.window, self.contract)['qualified'])

    def test_missing_timing_baseline_pollution_and_invalid_profile_rejected(self):
        self.assertFalse(check(self.rows()[:-1], self.window, self.contract)['qualified'])
        self.assertFalse(check(self.rows(), self.window, self.contract, [35, 42])['qualified'])
        self.assertFalse(check(self.rows(), self.window, self.contract, [52])['qualified'])
        rows = self.rows(); rows[0]['walltime'] = float('nan')
        self.assertFalse(check(rows, self.window, self.contract)['qualified'])


if __name__ == '__main__':
    unittest.main()
