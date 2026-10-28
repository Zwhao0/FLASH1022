import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from flash_v4.profile_finalize import (CONTRACT, ProfileFinalizer,
                                      validate_contract, validate_summary)
from flash_v4.online_runtime import OnlineObserver


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.control = self.root / 'control'; self.control.mkdir()
        self.observations = self.root / 'observations'; self.observations.mkdir()
        self.row = dict(kind='summary', rank=12, window_complete=True,
                        completed_profile_iterations=8, dropped=0, skipped=0,
                        event_pool=dict(cleanup_complete=True))
        self.write_summary()
        self.native = Mock()
        self.order = []
        self.native.flush.side_effect = lambda: self.order.append('flush')
        self.device = lambda d: self.order.append(('device', d))

    def write_summary(self):
        (self.observations/'rank_12.jsonl').write_text(json.dumps(self.row)+'\n')

    def make(self, **kwargs):
        return ProfileFinalizer(self.native, self.control, self.observations,
                                12, 36, 43, 4, self.device, **kwargs)

    def finish(self, finalizer):
        finalizer.thread.join(2)
        self.assertFalse(finalizer.thread.is_alive())

    def test_normal_order_atomic_ready_one_native_owner(self):
        f = self.make(); f.start(); self.finish(f); f.shutdown(); f.shutdown()
        self.assertEqual(self.order, [('device', 4), 'flush'])
        self.assertFalse(f.thread.daemon)
        self.assertTrue(f.published)
        self.assertEqual(json.loads((self.control/'ready_rank12.json').read_text())['last'], 43)
        self.assertFalse(list(self.control.glob('*.tmp')))
        self.native.flush.assert_called_once()

    def blocked(self, **kwargs):
        entered, release = threading.Event(), threading.Event()
        def blocking():
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test did not release worker')
        self.native.flush.side_effect = blocking
        f = self.make(**kwargs); f.start()
        self.assertTrue(entered.wait(1))
        self.addCleanup(release.set)
        return f, release

    def test_overlap_cancel_does_not_stop_native_or_publish_late(self):
        f, release = self.blocked()
        self.assertFalse(f.published)
        f.cancel(); release.set(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertFalse((self.control/'ready_rank12.json').exists())
        self.native.flush.assert_called_once()

    def test_timeout_keeps_owner_live_and_never_publishes_late(self):
        now = [0.0]
        f, release = self.blocked(clock=lambda: now[0])
        now[0] = 11
        with self.assertRaises(TimeoutError): f.shutdown()
        self.assertTrue(f.thread.is_alive())
        release.set(); self.finish(f)
        self.assertFalse(f.published)
        self.assertFalse((self.control/'ready_rank12.json').exists())

    def test_deadline_without_shutdown_also_rejects_late_ready(self):
        now = [0.0]
        f, release = self.blocked(clock=lambda: now[0])
        now[0] = 11; release.set(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertFalse(f.published)

    def test_device_failure_never_enters_native(self):
        f = self.make(); f.set_device = Mock(side_effect=ValueError('device'))
        f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertFalse(f.native_entered)
        self.native.flush.assert_not_called()
        self.assertFalse(f.published)

    def test_native_failure_no_ready(self):
        self.native.flush.side_effect = OSError('native')
        f = self.make(); f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertTrue(f.native_entered)
        self.assertFalse(f.published)

    def test_thread_start_failure_no_owner_or_ready(self):
        f = self.make()
        with patch.object(f.thread, 'start', side_effect=RuntimeError('start')):
            with self.assertRaises(RuntimeError): f.start()
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertFalse(f.started)
        self.native.flush.assert_not_called()

    def test_summary_missing_incomplete_dropped_or_bad_pool_rejected(self):
        for key, value in [('window_complete', False), ('completed_profile_iterations', 7),
                           ('dropped', 1), ('skipped', 1), ('rank', 4),
                           ('event_pool', {'cleanup_complete': False})]:
            with self.subTest(key=key):
                old = self.row[key]; self.row[key] = value; self.write_summary()
                with self.assertRaises(RuntimeError): validate_summary(self.observations, 12)
                self.row[key] = old
        (self.observations/'rank_12.jsonl').unlink()
        f = self.make(); f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertFalse(f.published)

    def test_partial_and_oversize_summary_rejected(self):
        p = self.observations/'rank_12.jsonl'
        p.write_text(json.dumps(self.row))
        with self.assertRaises(RuntimeError): validate_summary(self.observations, 12)
        p.write_bytes(b'x'*1048577)
        with self.assertRaises(RuntimeError): validate_summary(self.observations, 12)

    def test_existing_ready_never_overwritten(self):
        p = self.control/'ready_rank12.json'; p.write_text('frozen')
        f = self.make(); f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertEqual(p.read_text(), 'frozen')
        self.assertFalse(f.published)

    def test_existing_temp_never_deleted(self):
        p = self.control/'.ready_rank12.r68.tmp'; p.write_text('frozen')
        f = self.make(); f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertEqual(p.read_text(), 'frozen')
        self.assertFalse(f.published)

    def test_failed_publish_no_ready(self):
        f = self.make()
        with patch('flash_v4.profile_finalize.os.link', side_effect=OSError('disk')):
            f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): f.shutdown()
        self.assertFalse(f.published)

    def test_opt_in_contract_online_and_supervised_only(self):
        validate_contract(None, 'off', None)
        validate_contract(CONTRACT, 'online', CONTRACT)
        for contract, mode, supervision in [(CONTRACT, 'off', CONTRACT),
                ('bad', 'online', CONTRACT), (CONTRACT, 'online', None)]:
            with self.assertRaises(ValueError): validate_contract(contract, mode, supervision)

    def observer(self):
        obs = OnlineObserver(self.native, self.control, 12, profile_finalize=CONTRACT)
        obs.group, obs.dist, obs.payload = object(), Mock(), [0]
        return obs

    def test_online_native_end_before_background_device_and_no_budget_reapply(self):
        obs = self.observer()
        native_end = []
        self.native.end_fn.side_effect = lambda i: native_end.append(i)
        raw = Mock(side_effect=lambda d: self.assertEqual(native_end[-1], 11))
        wrapped = Mock(); wrapped._flash_v4_device_only = raw
        torch = Mock(); torch.cuda.set_device = wrapped; torch.cuda.current_device.return_value = 4
        with patch.dict('sys.modules', {'torch': torch}), patch.dict(os.environ,
                FLASH_V4_OUTPUT_DIR=str(self.observations)):
            obs.state.start(4)
            for i in range(1, 12): obs.begin(i); obs.end(i)
        self.finish(obs.finalizer)
        obs.flush(); obs.flush()
        raw.assert_called_once_with(4); wrapped.assert_not_called()
        self.native.flush.assert_called_once()
        records = [json.loads(s) for s in (self.control/'online_rank12.jsonl').read_text().splitlines()]
        self.assertEqual([r['kind'] for r in records], ['profile_complete', 'monitor_end'])
        self.assertTrue(records[-1]['profile_complete'])
        self.assertTrue(obs.audit.closed)

    def test_monitor_has_no_finalizer_thread_or_device_work(self):
        obs = self.observer()
        with patch('flash_v4.profile_finalize.threading.Thread') as thread:
            for i in range(1, 12): obs.begin(i); obs.end(i)
            thread.assert_not_called()
        self.assertIsNone(obs.finalizer)
        self.native.begin_fn.assert_not_called()
        obs.flush(); self.native.flush.assert_called_once()

    def test_online_timeout_does_not_close_audit_or_reflush_native(self):
        obs = self.observer(); obs.first, obs.last = 36, 43
        now = [0.0]
        f, release = self.blocked(clock=lambda: now[0]); obs.finalizer = f
        now[0] = 11
        with self.assertRaises(TimeoutError): obs.flush()
        self.assertFalse(obs.closed); self.assertFalse(obs.audit.closed)
        self.assertNotIn('monitor_end', (self.control/'online_rank12.jsonl').read_text())
        release.set(); self.finish(f)
        with self.assertRaises(RuntimeError): obs.flush()
        self.assertTrue(obs.audit.closed)
        self.assertFalse((self.control/'ready_rank12.json').exists())
        self.native.flush.assert_called_once()

    def test_joined_failure_audit_marks_profile_incomplete(self):
        obs = self.observer(); obs.first, obs.last = 36, 43; obs.state.complete = True
        self.native.flush.side_effect = OSError('flush')
        f = self.make(); obs.finalizer = f; f.start(); self.finish(f)
        with self.assertRaises(RuntimeError): obs.flush()
        records = [json.loads(s) for s in (self.control/'online_rank12.jsonl').read_text().splitlines()]
        self.assertEqual(records[-1]['monitor_status'], 'failed')
        self.assertFalse(records[-1]['profile_complete']); self.assertTrue(obs.closed)
        self.native.flush.assert_called_once()


if __name__ == '__main__': unittest.main()
