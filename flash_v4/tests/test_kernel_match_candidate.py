"""Fallback comparability, health-only support, rank symmetry and drift checks."""
import unittest

from flash_v4.kernel_match_candidate import fit_exact, normalize, score_kernel


def row(rank, iteration, elapsed=10, grid=1):
    return {'kind': 'kernel', 'rank': rank, 'iteration': iteration,
            'kernel_id': '0123456789abcdef', 'launch_api': 'runtime',
            'grid': [grid, 1, 1], 'block': [128, 1, 1],
            'dynamic_shared_bytes': 0, 'sampling_stride': 256,
            'elapsed_us': elapsed, 'skip': ''}


def dataset(ranks=16, iterations=8):
    return {rank: {'event_rows': [row(rank, i) for i in range(iterations)]}
            for rank in range(ranks)}


class ExactKernelTests(unittest.TestCase):
    def test_fixed_shape_family_needs_no_shape_regression(self):
        data = dataset()
        exact = fit_exact(data, {})
        records, coverage = score_kernel(data, {}, exact)
        self.assertEqual(len(exact), 1)
        self.assertEqual(records[0][0], [2560, 2560])
        self.assertEqual(coverage['0']['modeled_event_fraction'], 1)

    def test_no_extrapolation_or_retraining_from_condition(self):
        exact = fit_exact(dataset(), {})
        condition = dataset()
        condition[0]['event_rows'] = [row(0, i, elapsed=100, grid=2) for i in range(8)]
        records, coverage = score_kernel(condition, {}, exact)
        self.assertEqual(records[0], {})
        self.assertEqual(coverage['0']['unknown_events'], 8)
        self.assertEqual(next(iter(exact.values())), 10)

    def test_support_requires_multiple_ranks_and_six_iterations(self):
        self.assertEqual(fit_exact(dataset(ranks=1), {}), {})
        self.assertEqual(fit_exact(dataset(iterations=5), {}), {})

    def test_fitted_regression_has_priority(self):
        data = dataset()
        models = {('0123456789abcdef', 'runtime', 128): [5.0, 0.0]}
        self.assertEqual(fit_exact(data, models), {})
        records, coverage = score_kernel(data, models, {})
        self.assertEqual(records[0][0], [2560, 1280])
        self.assertEqual(coverage['0']['exact_fallback_events'], 0)

    def test_common_slowdown_does_not_create_a_slow_rank(self):
        data = dataset()
        exact = fit_exact(data, {})
        for rank in data:
            for r in data[rank]['event_rows']:
                r['elapsed_us'] *= 3
        records, _ = score_kernel(data, {}, exact)
        rows = normalize(records, {r: 1 for r in range(16)}, range(8))
        self.assertTrue(all(r['score_median'] == 1 for r in rows))

    def test_any_rank_can_have_the_same_detectable_deviation(self):
        exact = fit_exact(dataset(), {})
        for target in (0, 7, 15):
            data = dataset()
            for r in data[target]['event_rows']:
                r['elapsed_us'] *= 2
            records, _ = score_kernel(data, {}, exact)
            rows = normalize(records, {r: 1 for r in range(16)}, range(8))
            self.assertEqual(rows[0]['rank'], target)
            self.assertEqual(rows[0]['score_median'], 2)


if __name__ == '__main__':
    unittest.main()
