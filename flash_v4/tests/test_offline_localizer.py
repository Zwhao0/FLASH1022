"""Two focused mathematical checks for the exploratory localizer."""
import unittest

from flash_v4.offline_localizer import (
    _kernel_family_and_features,
    _predict,
    _robust_nnls,
    _weighted_nnls,
)


class OfflineLocalizerMathTests(unittest.TestCase):
    def test_nnls_predicts_a_shape_not_present_in_training_rows(self):
        features = [
            (1.0, 0.05, 0.30),
            (1.0, 0.12, 0.07),
            (1.0, 0.22, 0.41),
            (1.0, 0.37, 0.16),
            (1.0, 0.51, 0.62),
            (1.0, 0.73, 0.28),
        ]
        expected = (3.0, 2.0, 4.0)
        observations = [sum(a * b for a, b in zip(expected, row)) for row in features]
        coefficients = _weighted_nnls(features, observations, [1.0] * len(features))
        unseen = (1.0, 0.46, 0.53)
        self.assertAlmostEqual(_predict(coefficients, unseen),
                               sum(a * b for a, b in zip(expected, unseen)), places=8)

    def test_three_robust_reweight_rounds_limit_one_extreme_outlier(self):
        points = [
            (0.03, 0.41), (0.07, 0.10), (0.11, 0.22), (0.18, 0.55),
            (0.24, 0.14), (0.31, 0.37), (0.43, 0.63), (0.58, 0.25),
        ]
        truth = (2.0, 4.0, 3.0)
        samples = []
        for index, (flops, byte_count) in enumerate(points):
            elapsed = truth[0] + truth[1] * flops + truth[2] * byte_count
            if index == 7:
                elapsed += 1000.0
            samples.append({
                "shape": (index + 1, index + 2, index + 3),
                "features": (1.0, flops, byte_count),
                "elapsed_us": elapsed,
            })
        coefficients, weights = _robust_nnls(samples)
        expected = truth[0] + truth[1] * points[7][0] + truth[2] * points[7][1]
        self.assertLess(abs(_predict(coefficients, samples[7]["features"]) - expected), 2.0)
        self.assertLess(weights[7], 0.1)

    def test_kernel_model_predicts_a_grid_size_not_seen_in_health(self):
        samples = []
        for blocks in (1, 2, 4, 8):
            row = {
                "kind": "kernel",
                "kernel_id": "0123456789abcdef",
                "launch_api": "runtime",
                "grid": [blocks, 1, 1],
                "block": [256, 1, 1],
                "dynamic_shared_bytes": 0,
                "sampling_stride": 256,
                "elapsed_us": 3.0 + 0.02 * blocks * 256,
                "skip": "",
            }
            _, sample, reason = _kernel_family_and_features(row)
            self.assertIsNone(reason)
            samples.append(sample)
        coefficients, _ = _robust_nnls(samples)
        unseen = dict(samples[0])
        unseen["features"] = (1.0, 7 * 256)
        self.assertAlmostEqual(_predict(coefficients, unseen["features"]),
                               3.0 + 0.02 * 7 * 256, places=8)


if __name__ == "__main__":
    unittest.main()
