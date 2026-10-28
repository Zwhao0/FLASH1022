import math
import unittest

from flash_v4.monitor_policy import fit_healthy, should_trigger


class MonitorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.calibration = fit_healthy([100.0] * 8)

    def test_healthy_samples_do_not_trigger(self):
        self.assertFalse(should_trigger([100.0, 100.0, 100.0], self.calibration))
        self.assertAlmostEqual(self.calibration["threshold"], 115.0)

    def test_three_sustained_slow_samples_trigger(self):
        self.assertTrue(should_trigger([116.0, 118.0, 120.0], self.calibration))

    def test_one_spike_does_not_trigger(self):
        self.assertFalse(should_trigger([130.0, 100.0, 100.0], self.calibration))

    def test_invalid_or_missing_recent_sample_prevents_trigger(self):
        invalid_windows = (
            [116.0, 118.0],
            [116.0, None, 120.0],
            [116.0, math.nan, 120.0],
            [116.0, math.inf, 120.0],
            [116.0, 0.0, 120.0],
            [116.0, -1.0, 120.0],
        )
        for recent in invalid_windows:
            with self.subTest(recent=recent):
                self.assertFalse(should_trigger(recent, self.calibration))

    def test_invalid_healthy_values_are_ignored_and_eight_valid_are_required(self):
        fitted = fit_healthy([100.0] * 8 + [None, math.nan, math.inf, 0.0, -1.0])
        self.assertEqual(fitted["sample_count"], 8)
        with self.assertRaises(ValueError):
            fit_healthy([100.0] * 7 + [math.nan])

    def test_threshold_and_trigger_are_scale_invariant(self):
        healthy = [80.0, 90.0, 95.0, 100.0, 100.0, 105.0, 110.0, 120.0]
        recent = [130.0, 132.0, 134.0]
        base = fit_healthy(healthy)
        factor = 3.0
        scaled = fit_healthy([value * factor for value in healthy])

        self.assertAlmostEqual(scaled["threshold"], base["threshold"] * factor)
        self.assertEqual(
            should_trigger(recent, base),
            should_trigger([value * factor for value in recent], scaled),
        )


if __name__ == "__main__":
    unittest.main()
