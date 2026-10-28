import unittest
import json
import tempfile
from pathlib import Path

from flash_v4.online_evaluate import actuation_quality, classify


class OutcomeTests(unittest.TestCase):
    def test_distinct_detection_and_localization_outcomes(self):
        q = dict(nodes_completed=True, quality_passed=True, profile_triggered=True, window=[52, 59])
        fault = dict(target_rank=6, window=dict(first=40, last=71))
        healthy = dict(target_rank=None)
        for prediction, truth, expected in [({'alarm_rank': 6}, fault, 'correct_alarm'),
                                             ({'alarm_rank': 7}, fault, 'mislocalized'),
                                             ({'alarm_rank': None}, fault, 'abstained'),
                                             ({'alarm_rank': 6}, healthy, 'healthy_false_alarm'),
                                             ({'alarm_rank': None}, healthy, 'healthy_trigger_abstained'),
                                             (None, fault, 'inference_failure')]:
            self.assertEqual(classify(q, prediction, truth), expected)
        self.assertEqual(classify(dict(q, profile_triggered=False), None, fault), 'not_triggered')
        self.assertEqual(classify(dict(q, profile_triggered=False), None, healthy), 'healthy_no_trigger')
        self.assertEqual(classify(dict(q, window=[36, 43]), {'alarm_rank': 6}, fault), 'pre_injection_trigger')
        self.assertEqual(classify(dict(q, window=[76, 83]), {'alarm_rank': 6}, fault), 'fault_window_missed')
        self.assertEqual(classify(dict(q, error='failed'), None, fault), 'collection_failure')

    def test_actor_checks_include_warmup_and_reject_wrong_rank_or_caps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = dict(target_rank=6, ratio=1, window=dict(first=4, last=5))
            for rank in range(16):
                directory = root / f'node{rank // 8}' / 'actuation'
                directory.mkdir(parents=True, exist_ok=True)
                rows = [dict(kind='configuration', ratio=1 if rank == 6 else 0,
                             max_fault_iterations=2, max_step_us=200000, max_fault_seconds=30)]
                rows += [dict(kind='iteration', iteration=i,
                              requested_delay_us=20 if rank == 6 and i >= 4 else 0)
                         for i in range(1, 6)]
                rows += [dict(kind='summary')]
                (directory / f'stream_delay_rank_{rank}.jsonl').write_text(
                    '\n'.join(json.dumps(r) for r in rows))
            self.assertTrue(actuation_quality(root, truth, 2)['passed'])
            self.assertTrue(actuation_quality(root, truth, 2, budget=dict(max_fault_iterations=2))['passed'])
            self.assertFalse(actuation_quality(root, truth, 2, budget=dict(max_fault_iterations=1))['passed'])
            self.assertFalse(actuation_quality(root, dict(truth, target_rank=7), 2)['passed'])
            self.assertFalse(actuation_quality(root, dict(target_rank=None), 2)['passed'])
            path = root / 'node0/actuation/stream_delay_rank_6.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[4]['cap_hits'] = 1
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertFalse(actuation_quality(root, truth, 2)['passed'])


if __name__ == '__main__':
    unittest.main()
