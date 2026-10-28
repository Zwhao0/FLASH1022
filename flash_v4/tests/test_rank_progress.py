"""CPU-only contracts; all commands/processes in watcher tests are fakes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flash_v4.rank_progress import CONTRACT, ProgressObserver, validate_receipt
from flash_v4 import online_harness as harness
from results.v4.preflight import real_clock_host_watch_r66 as watcher

RUN_ID = 'v4_' + 'a' * 32


def receipt(**extra):
    return dict(dict(contract=CONTRACT, run_id=RUN_ID, rank=12, local_rank=4,
                     node_rank=1, iteration=49, status='running', monotonic_seconds=100), **extra)


class ProgressContracts(unittest.TestCase):
    def test_receipts_reject_wrong_identity_invalid_stale_and_regression(self):
        self.assertEqual(validate_receipt(receipt(), run_id=RUN_ID, rank=12, now=101), 49)
        for bad in (dict(rank=4), dict(node_rank=0), dict(local_rank=3), dict(run_id='old'),
                    dict(status='closed'), dict(iteration=True), dict(iteration=513),
                    dict(iteration=48), dict(monotonic_seconds=97),
                    dict(monotonic_seconds=102), dict(monotonic_seconds=float('nan'))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_receipt(receipt(**bad), run_id=RUN_ID, rank=12, now=101, previous=49)

    def test_writer_only_publishes_successful_schedule_ends_and_closes_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            delegate = Mock(first=0, last=0)
            observer = ProgressObserver(delegate, temp, 12, 4, RUN_ID, lambda: 100)
            observer.begin(1)
            self.assertFalse(observer.path.exists())
            observer.end(1)
            data = json.loads(observer.path.read_text())
            self.assertEqual(data['iteration'], 1)
            self.assertLess(observer.path.stat().st_size, 1024)
            self.assertFalse(observer.temp.exists())
            observer.begin(2)
            delegate.end.side_effect = RuntimeError('schedule observer failed')
            with self.assertRaises(RuntimeError): observer.end(2)
            self.assertEqual(json.loads(observer.path.read_text())['iteration'], 1)
            observer.flush(); observer.flush()
            self.assertEqual(json.loads(observer.path.read_text())['status'], 'closed')
            delegate.flush.assert_called_once()
            with self.assertRaises(FileExistsError): ProgressObserver(delegate, temp, 12, 4, RUN_ID)

    def test_writer_rejects_wrong_local_rank_and_iteration_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): ProgressObserver(Mock(), temp, 12, 3, RUN_ID)
            observer = ProgressObserver(Mock(), temp, 0, 0, RUN_ID)
            with self.assertRaises(ValueError): observer.begin(2)
            observer.iteration = 512
            with self.assertRaises(ValueError): observer.begin(513)

    def stream(self, raw, *, remove_after_first=False, terminal=False, later=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'control/progress/rank12.json'
            path.parent.mkdir(parents=True)
            if raw is not None: path.write_text(raw)
            if terminal: (root/'node_status.json').write_text('{"status":"completed"}')
            now = [100.]
            def sleep(_):
                now[0] += .1 if raw is not None else 181
                if remove_after_first: path.unlink(missing_ok=True)
                if later: path.write_text(json.dumps(later))
            output = io.StringIO()
            with patch('time.monotonic', side_effect=lambda: now[0]), patch('time.sleep', side_effect=sleep), contextlib.redirect_stdout(output):
                exec(watcher.STREAM, dict(ROOT=str(root), TARGET_RANK=12, RUN_ID=RUN_ID))
            return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_stream_uses_rank12_without_train_log_then_rejects_stale(self):
        rows = self.stream(json.dumps(receipt()))
        self.assertEqual(rows[0]['iteration'], 49)
        self.assertEqual(rows[0]['progress_rank'], 12)
        self.assertIn('stale', rows[-1]['progress_error'])

    def test_stream_rejects_invalid_wrong_node_oversize_and_missing(self):
        for raw in ('invalid', 'x'*1025, json.dumps(receipt(node_rank=0)), None):
            with self.subTest(raw=str(raw)[:20]):
                rows = self.stream(raw)
                self.assertIn('progress_error', rows[-1])
                self.assertFalse(any('iteration' in row for row in rows))
        rows = self.stream(json.dumps(receipt()), remove_after_first=True)
        self.assertIn('missing', rows[-1]['progress_error'])

    def test_stream_rejects_regressed_iteration_and_terminal_stops(self):
        rows = self.stream(json.dumps(receipt()), later=receipt(iteration=48))
        self.assertIn('nonmonotonic', rows[-1]['progress_error'])
        rows = self.stream(None, terminal=True)
        self.assertEqual(rows, [dict(terminal=dict(status='completed'))])

    def test_stream_rejects_timestamp_regression(self):
        rows = self.stream(json.dumps(receipt()), later=receipt(iteration=50,monotonic_seconds=99))
        self.assertIn('nonmonotonic progress timestamp', rows[-1]['progress_error'])

    def test_valid_harness_opt_in_propagates_identity_to_both_nodes(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/'plan.json'
            plan.write_text(json.dumps(dict(development_GPU_sessions=[dict(case='c01',steps=104,mode='online',
                topology='legacy',seed=1,rank_progress_observation=CONTRACT)])))
            with patch.object(harness,'PLAN',plan), patch.object(harness,'LOCAL',Path(temp)/'runs'), patch.object(harness,'transport', return_value='') as remote:
                harness.launch('c01')
                self.assertEqual(remote.call_count,2)
                for call in remote.call_args_list:
                    self.assertIn('FLASH_V4_RANK_PROGRESS='+CONTRACT,call.args[1])
                    self.assertIn('FLASH_V4_PROGRESS_RUN_ID='+harness.runtime_run_id('c01'),call.args[1])

    def test_invalid_harness_opt_in_never_dispatches(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/'plan.json'
            for contract, steps, mode in [('bad',104,'online'), (CONTRACT,510,'online'), (CONTRACT,104,'off')]:
                plan.write_text(json.dumps(dict(development_GPU_sessions=[dict(case='c01', steps=steps,
                    mode=mode, rank_progress_observation=contract)])))
                with patch.object(harness,'PLAN',plan), patch.object(harness,'LOCAL',Path(temp)/'runs'), patch.object(harness,'transport') as remote:
                    with self.assertRaises(ValueError): harness.launch('c01')
                    remote.assert_not_called()
                    self.assertFalse((Path(temp)/'runs').exists())


class WatcherCleanup(unittest.TestCase):
    def run_fake(self, rows, *, tick=.1):
        with tempfile.TemporaryDirectory() as temp:
            cfg = dict(gpu_uuid='GPU-test', original_sm_mhz=1980, lowered_sm_mhz=345,
                target_rank=12, logical_node_rank=1, first=50, last=81,
                container_command=['fake-docker'], container_output='/case/node1',
                host_output=str(Path(temp)/'evidence'), progress_contract=CONTRACT, progress_run_id=RUN_ID)
            commands, processes = [], []
            clock = [0.]
            class Process:
                def __init__(self, lines=''):
                    class LiveStream(io.StringIO):
                        def readline(self):
                            raw = super().readline()
                            if not raw: return raw
                            row = json.loads(raw)
                            if 'iteration' in row: row.setdefault('progress_monotonic_seconds', clock[0])
                            return json.dumps(row)+'\n'
                    self.stdout = LiveStream(lines)
                    self.stopped = False
                def poll(self): return 0 if self.stopped else None
                def terminate(self): self.stopped = True
                def kill(self): self.stopped = True
                def wait(self, timeout): self.stopped = True; return 0
            def spawn(args, **kwargs):
                proc = Process('\n'.join(json.dumps(row) for row in rows)+'\n' if not processes else '')
                processes.append(proc); return proc
            class Selector:
                def register(self, *args): pass
                def unregister(self, *args): pass
                def close(self): pass
                def select(self, **kwargs): clock[0] += tick; return [(None, None)]
            def command(args, **kwargs):
                commands.append(args)
                return SimpleNamespace(returncode=0, stdout='GPU-test, 1980, 1980, 0\n', stderr='')
            rc = watcher.run(cfg, command_runner=command, popen_factory=spawn,
                             selector_factory=Selector, monotonic=lambda: clock[0], wall_clock=lambda: 1000+clock[0])
            result = json.loads((Path(cfg['host_output'])/'clock_result.json').read_text())
            events = [json.loads(line) for line in (Path(cfg['host_output'])/'clock_events.jsonl').read_text().splitlines()]
            self.assertTrue(all(proc.stopped for proc in processes))
            self.assertEqual(result['cleanup_errors'], [])
            return rc, result, events, commands

    def rows(self):
        binding = dict(node_rank=1, local_rank_to_gpu_uuid=['GPU-'+str(i) if i!=4 else 'GPU-test' for i in range(8)])
        return [dict(binding=binding), dict(iteration=49, progress_rank=12,
                  progress_node_rank=1, progress_run_id=RUN_ID),
                dict(iteration=81, progress_rank=12, progress_node_rank=1, progress_run_id=RUN_ID),
                dict(terminal=dict(status='completed'))]

    def test_rank12_normal_lower_restore_and_cleanup(self):
        rc, result, events, commands = self.run_fake(self.rows())
        self.assertEqual(rc, 0)
        self.assertTrue(result['changed'] and result['restored'] and result['physical_binding_verified'])
        self.assertEqual(sum('345,345' in cmd for cmd in commands), 1)
        self.assertTrue(any(e['kind']=='restored_iteration' for e in events))

    def test_invalid_progress_before_lower_never_changes_clock(self):
        rows = self.rows()
        rows[1]['progress_node_rank'] = 0
        rc, result, _, commands = self.run_fake(rows)
        self.assertEqual(rc, 1)
        self.assertFalse(result['changed'])
        self.assertFalse(any('-lgc' in cmd for cmd in commands))

    def test_stale_host_pipe_progress_never_lowers_clock(self):
        rows=self.rows()
        rows[1]['progress_monotonic_seconds']=-100
        rc,result,_,commands=self.run_fake(rows)
        self.assertEqual(rc,1)
        self.assertFalse(result['changed'])
        self.assertFalse(any('-lgc' in cmd for cmd in commands))

    def test_stream_rejection_after_lower_restores_and_cleans(self):
        rc, result, _, _ = self.run_fake(self.rows()[:2]+[dict(progress_error='stale')])
        self.assertEqual(rc, 1)
        self.assertTrue(result['changed'] and result['restored'])
        self.assertIn('stale', result['error'])
        self.assertFalse(result['independent_watchdog_left_armed'])

    def test_fault_time_budget_restores_even_without_end_iteration(self):
        rows = self.rows()[:2]+[dict(iteration=50,progress_rank=12,progress_node_rank=1,progress_run_id=RUN_ID),{},dict(terminal={})]
        _, result, events, _ = self.run_fake(rows, tick=20)
        self.assertTrue(result['restored'])
        self.assertTrue(any(e['kind']=='restored_fault_budget' for e in events))


if __name__ == '__main__': unittest.main()
