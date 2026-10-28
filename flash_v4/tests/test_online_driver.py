import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from flash_v4 import online_driver as driver
from flash_v4.online_driver import control_integrity


class ControlIntegrityTests(unittest.TestCase):
    def evidence(self, profile=True, manual=False):
        summaries = [{'rank': r} for r in range(16)]
        ready = ([{'rank': r, 'first': 52, 'last': 59} for r in range(16)]
                 if profile else [])
        endings = [{'rank': r, 'profile_requested': profile, 'profile_complete': profile,
                    'monitor_status': ('manual_integration' if manual else
                                       'completed' if r == 0 else 'peer_completed'),
                    'worker_returncode': 0 if r == 0 and not manual else None}
                   for r in range(16)]
        requests = ([{'method': 'flash_v1_Rbeast', 'iteration': 49}]
                    if profile and not manual else [])
        return summaries, ready, endings, requests

    def test_automatic_manual_and_healthy_control_paths(self):
        for profile, manual in [(True, False), (True, True), (False, False)]:
            self.assertTrue(control_integrity(*self.evidence(profile, manual), manual=manual))

    def test_duplicate_rank_cannot_replace_missing_rank(self):
        for position in range(3):
            evidence = self.evidence()
            evidence[position][-1] = evidence[position][0].copy()
            self.assertFalse(control_integrity(*evidence))

    def test_profile_requires_real_earlier_bocd_request(self):
        for request in [[], [{'method': 'manual', 'iteration': 49}],
                        [{'method': 'flash_v1_Rbeast', 'iteration': 52}]]:
            summaries, ready, endings, _ = self.evidence()
            self.assertFalse(control_integrity(summaries, ready, endings, request))

    def test_monitor_failure_or_partial_profile_fails(self):
        for field, value in [('worker_returncode', 1), ('monitor_status', 'failed'),
                             ('profile_complete', False), ('profile_requested', False)]:
            evidence = self.evidence()
            evidence[2][0][field] = value
            self.assertFalse(control_integrity(*evidence))

    def test_no_profile_does_not_hide_monitor_failure(self):
        evidence = self.evidence(profile=False)
        evidence[2][0]['monitor_status'] = 'failed'
        self.assertFalse(control_integrity(*evidence))

    def test_terminal_bootstrap_failure_preserves_collection_failure_result(self):
        states = [dict(status='failed', returncode=2, private_mps_stopped=True,
                       profile_ready_ranks=0) for _ in range(2)]
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(driver.harness, 'LOCAL', Path(temp)), \
                patch.object(driver.harness, 'case_row', return_value={'case': 'c01'}), \
                patch.object(driver.harness, 'launch'), patch.object(driver.harness, 'fetch') as fetch, \
                patch.object(driver.harness, 'status', side_effect=lambda _: print(json.dumps(states))), \
                patch.object(driver, 'quality', side_effect=FileNotFoundError('no training started')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.execute('c01'), 1)
            result = json.loads((Path(temp) / 'driver_c01/result.json').read_text())
            self.assertFalse(result['quality_passed'])
            self.assertFalse(result['nodes_completed'])
            self.assertIn('FileNotFoundError', result['quality_error'])
            fetch.assert_called_once_with('c01', 'final')


if __name__ == '__main__':
    unittest.main()
