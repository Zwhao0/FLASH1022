import unittest
from unittest.mock import patch
import importlib
import sys
import types
from flash_v1.analysis import WindowDetector, localize, profile_seconds


class AnalysisTests(unittest.TestCase):
    def test_window_and_cp_mapping(self):
        # Candidate indices refer to the filtered series, not the raw iteration ID.
        d = WindowDetector(candidate_fn=lambda values: [(20, .99)])
        result = None
        for iteration in range(1, 34):
            result = d.add(iteration, 100 if iteration < 24 else 150)
        self.assertEqual(result['cp'], 24)
        self.assertEqual(result['window_start'], 4)
        self.assertEqual(result['baseline_mean'], 100)
        self.assertEqual(result['recovery_ratio'], 0)
        self.assertIsNone(d.add(34, 150))  # same event not reported twice

    def test_spike_and_recovery(self):
        d = WindowDetector(candidate_fn=lambda values: [(20, .99)])
        result = None
        for iteration in range(1, 34):
            result = d.add(iteration, 500 if iteration == 24 else 100)
        self.assertIsNone(result)

    def test_recovery_ratio_filter(self):
        d = WindowDetector(window=10, warmup=0, candidate_fn=lambda values: [(7, .99)])
        for i, value in enumerate([100]*7 + [200, 100, 100], 1):
            result = d.add(i, value)
        self.assertIsNone(result)

    def test_two_stage_selects_slowest_clique_and_fastest_rank(self):
        rows = [dict(tag='dp', clique=c, rank=r, duration_us=t)
                for c, values in [('a', [(0, 100), (1, 100)]),
                                  ('b', [(2, 400), (3, 10)])]
                for r, t in values]
        rows += [dict(tag='expert_dp', clique='x', rank=r, duration_us=t)
                 for r, t in [(1, 200), (3, 5)]]
        result = localize(rows)
        self.assertEqual(result['rank'], 3)
        self.assertEqual(len(result['evidence']), 2)

    def test_conflicting_evidence_uses_cumulative_score(self):
        rows = [dict(tag=tag, clique=tag, rank=r, duration_us=t)
                for tag, values in [('a', [(3, 75), (0, 125)]),
                                    ('b', [(3, 75), (1, 125)]),
                                    ('c', [(4, 10), (2, 190)])]
                for r, t in values]
        self.assertEqual(localize(rows)['rank'], 4)

    def test_controller_cursor_survives_redis_trim(self):
        # redis-py is not needed for this bounded-list controller regression.
        with patch.dict(sys.modules, {'redis': types.ModuleType('redis'),
                                      'numpy': types.ModuleType('numpy')}):
            control = importlib.import_module('flash_v1.control')
        class Client:
            polls = 0
            def exists(self, key):
                return key == 'shutdown' and self.polls >= 2
            def lrange(self, key, start, end):
                self.polls += 1
                first = self.polls
                return [f'{i}:100' for i in range(first, first + 500)]
            def get(self, key):
                return None
            def set(self, *args):
                pass
        seen = []
        class Detector:
            def add(self, iteration, duration):
                seen.append(iteration)
                return None
        with patch.object(control, 'WindowDetector', Detector), patch.object(control.time, 'sleep'):
            control.global_controller(None, Client(), None)
        self.assertEqual(seen, list(range(1, 502)))

    def test_no_separation_and_profile_units(self):
        rows = [dict(tag='dp', clique='a', rank=r, duration_us=100) for r in range(4)]
        self.assertIsNone(localize(rows)['rank'])
        self.assertEqual(profile_seconds(1000), 20)
        self.assertEqual(profile_seconds(10000), 22)
        with self.assertRaises(ValueError): profile_seconds(float('nan'))


if __name__ == '__main__':
    unittest.main()
