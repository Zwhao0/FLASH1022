import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flash_v4 import online_harness as harness
from flash_v4.health_centered_fusion import fingerprint


class LaunchPreflightTests(unittest.TestCase):
    def test_case_timeout_is_explicit_and_bounded(self):
        self.assertEqual(harness.case_timeout({}), 180)
        self.assertEqual(harness.case_timeout({'timeout': 60}), 60)
        for value in (0, 29, 181, True, 60.0):
            with self.assertRaises(ValueError): harness.case_timeout({'timeout': value})

    def test_run_identity_uses_full_release_and_case(self):
        from flash_v4.shared_node import check_name
        with patch.object(harness, 'RELEASE', 'v4_online_20261027_r24_automatic'):
            first = harness.runtime_run_id('c01')
            self.assertEqual(first, harness.runtime_run_id('c01'))
            self.assertNotEqual(first, harness.runtime_run_id('c02'))
            check_name(first)
        with patch.object(harness, 'RELEASE', 'v4_online_20261027_r25_automatic'):
            self.assertNotEqual(first, harness.runtime_run_id('c01'))

    def test_strong_actor_requires_explicit_plan_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / 'private.json'
            private.write_text(json.dumps(dict(cases=dict(c01=dict(
                target_rank=8, ratio=8, window=dict(first=40, last=63))))))
            plan = Path(temp) / 'plan.json'
            data = dict(private_execution=str(private),
                        development_GPU_sessions=[dict(case='c01', steps=96)])
            plan.write_text(json.dumps(data))
            with patch.object(harness, 'PLAN', plan):
                with self.assertRaisesRegex(ValueError, 'declared plan budget'):
                    harness.private_case('c01')
                data['actor_budget'] = dict(max_requested_ratio=8)
                plan.write_text(json.dumps(data))
                self.assertEqual(harness.private_case('c01')['ratio'], 8)
                data['actor_budget']['max_fault_iterations'] = 16
                plan.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'native iteration budget'):
                    harness.private_case('c01')

    def test_native_build_budget_rejected_before_remote_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / 'plan.json'
            for budget in [dict(max_step_us=1000001), dict(max_fault_iterations=33),
                           dict(max_step_us=True)]:
                plan.write_text(json.dumps(dict(actor_budget=budget)))
                with patch.object(harness, 'PLAN', plan), patch.object(harness, 'transport') as remote:
                    with self.assertRaises(ValueError):
                        harness.build_actor()
                    remote.assert_not_called()

    def test_invalid_private_actor_never_dispatches_or_creates_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / 'private.json'
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(private_execution=str(private),
                development_GPU_sessions=[dict(case='c01', steps=96)])))
            for truth in [dict(target_rank=8, ratio=2, window=dict(first=40, last=71)),
                          dict(target_rank=8, ratio=1, window=dict(first=40, last=72)),
                          dict(target_rank=None, ratio=1, window=dict(first=40, last=71))]:
                private.write_text(json.dumps(dict(cases=dict(c01=truth))))
                with patch.object(harness, 'PLAN', plan), patch.object(harness, 'LOCAL', root / 'runs'), \
                        patch.object(harness, 'transport') as remote:
                    with self.assertRaises(ValueError):
                        harness.launch('c01')
                    remote.assert_not_called()
                    self.assertFalse((root / 'runs').exists())

    def test_valid_bounded_actor_and_frozen_private_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / 'private.json'
            private.write_text(json.dumps(dict(cases=dict(c01=dict(
                target_rank=8, ratio=1, window=dict(first=40, last=71))))))
            plan = Path(temp) / 'plan.json'
            plan.write_text(json.dumps(dict(private_execution=str(private),
                private_execution_receipt=fingerprint(private),
                development_GPU_sessions=[dict(case='c01', steps=96)])))
            with patch.object(harness, 'PLAN', plan):
                self.assertEqual(harness.private_case('c01')['target_rank'], 8)
                private.write_text(private.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'changed after freeze'):
                    harness.private_case('c01')

    def test_early_bootstrap_failure_keeps_log_and_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'path with spaces' / 'node0'
            args = [sys.executable, '-c', 'print("bootstrap diagnostic"); raise SystemExit(23)']
            command = harness.bootstrap_command(args, str(output))
            result = subprocess.run(shlex.split(command), capture_output=True, text=True)
            self.assertEqual(result.returncode, 23)
            self.assertEqual((output.parent / 'node0_bootstrap_exit').read_text().strip(), '23')
            self.assertIn('bootstrap diagnostic', (output.parent / 'node0_bootstrap.log').read_text())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
