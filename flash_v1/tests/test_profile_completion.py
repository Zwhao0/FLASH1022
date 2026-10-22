"""A profile is complete only after its localization event has been persisted."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class ProfileCompletionTests(unittest.TestCase):
    def test_delayed_aggregation_completes_after_localization_even_without_samples(self):
        # No Redis server, detector fit, GPU, or real profiling wait is needed.
        with patch.dict(sys.modules, {'redis': types.ModuleType('redis')}):
            control = importlib.import_module('flash_v1.control')
        for rows in ([], [dict(rank=3, tag='dense_DP', clique='a', duration_us=100)]):
            with self.subTest(samples=len(rows)), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                event_path = output / 'events.jsonl'
                clock = types.SimpleNamespace(now=0.0)

                class Detector:
                    def __init__(self, **kwargs):
                        pass

                    def add(self, iteration, duration):
                        return dict(cp=iteration) if iteration == 1 else None

                class Client:
                    def __init__(self):
                        self.values = {}
                        self.latest = 1
                        self.result_polls = 0
                        self.waiting_snapshots = []
                        self.marker_events = None
                        self.operations = []

                    def exists(self, key):
                        if key == 'shutdown':
                            return self.values.get('processed_iteration') == '2'
                        if key == 'profile_result_1' and self.values.get('state') == '0':
                            self.result_polls += 1
                            if self.result_polls <= 2:
                                self.waiting_snapshots.append(dict(self.values))
                            elif key not in self.values:
                                self.values[key] = json.dumps(rows)
                                self.latest = 2  # A step completes while aggregation drains.
                                self.operations.append('result_delivered')
                        return key in self.values

                    def get(self, key):
                        return self.values.get(key)

                    def set(self, key, value):
                        self.values[key] = str(value)
                        if key == 'state' and str(value) == '1':
                            self.values['profile_started_1'] = '1'
                        if key == 'profile_completed_1':
                            self.marker_events = [json.loads(line) for line in event_path.read_text().splitlines()]
                            self.operations.append('completed_marker')
                        if key == 'processed_iteration' and str(value) == '2':
                            self.operations.append('latest_iteration_processed')

                    def lrange(self, key, start, end):
                        return [f'{self.latest}:100']

                client = Client()

                def advance(seconds):
                    clock.now += seconds

                predicted = dict(status='identified', rank=3)
                with patch.object(control, 'WindowDetector', Detector), \
                     patch.object(control, 'profile_seconds', return_value=0.2), \
                     patch.object(control.time, 'monotonic', side_effect=lambda: clock.now), \
                     patch.object(control.time, 'sleep', side_effect=advance), \
                     patch.object(control, 'localize', return_value=predicted) as localize, \
                     contextlib.redirect_stdout(io.StringIO()):
                    control.global_controller(None, client, output)

                self.assertEqual(len(client.waiting_snapshots), 2)
                for snapshot in client.waiting_snapshots:
                    self.assertEqual(snapshot['state'], '0')
                    self.assertNotIn('profile_result_1', snapshot)
                    self.assertNotIn('profile_completed_1', snapshot)
                self.assertEqual([event['event'] for event in client.marker_events],
                                 ['detection', 'localization'])
                event = client.marker_events[-1]
                self.assertEqual(event['epoch'], 1)
                self.assertEqual(client.values['profile_completed_1'], '1')
                self.assertEqual(client.values['processed_iteration'], '2')
                self.assertEqual(client.operations, ['result_delivered', 'completed_marker',
                                                     'latest_iteration_processed'])
                if rows:
                    localize.assert_called_once_with(rows)
                    self.assertEqual(event['result'], predicted)
                else:
                    localize.assert_not_called()
                    self.assertEqual(event['result'], dict(status='inconclusive', rank=None,
                                                          reason='no probe samples'))
                self.assertLess(clock.now, 1.0)


if __name__ == '__main__':
    unittest.main()
