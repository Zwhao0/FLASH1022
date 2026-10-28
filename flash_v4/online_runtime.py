"""Runtime-only bridge: FLASH/v1 Rbeast request -> one bounded PROFILE.

The CPU analyzer consumes existing rank-zero iteration logs. A sparse Gloo
broadcast coordinates the same schedule iteration on all ranks. MONITOR adds
no CUDA events. Localization remains the separately frozen v3 model.
"""
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class ProfileState:
    """One window per session, followed by permanent session cooldown."""
    def __init__(self, stride=8):
        self.stride = stride
        self.first = self.last = 0
        self.complete = False
        self.last_iteration = 0

    def check_due(self, iteration):
        return not self.first and iteration >= 4 and (iteration - 4) % self.stride == 0

    def start(self, iteration):
        if self.first or self.complete or iteration <= self.last_iteration:
            raise RuntimeError('duplicate or late PROFILE request')
        self.first, self.last = iteration, iteration + 7

    def begin(self, iteration):
        if iteration != self.last_iteration + 1:
            raise RuntimeError('schedule iteration skipped or repeated')
        self.last_iteration = iteration
        return bool(self.first and self.first <= iteration <= self.last and not self.complete)

    def end(self, iteration):
        if iteration != self.last_iteration:
            raise RuntimeError('schedule completion mismatch')
        if self.first and iteration == self.last:
            self.complete = True
            return True
        return False


def child_environment(env):
    # A CPU consumer must not inherit GPU instrumentation or actuator payloads.
    result = {k: v for k, v in env.items()
              if k != 'LD_PRELOAD' and not k.endswith('_PRIVATE')}
    result.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return result


class OnlineObserver:
    scheduling = 'FLASH_v1_Rbeast_request_sparse_Gloo_single_bounded_window'

    def __init__(self, native, directory, rank, force_at=0, profile_finalize=None):
        if force_at and (force_at < 4 or (force_at - 4) % 8):
            raise ValueError('manual integration trigger must be a control boundary')
        self.native, self.directory, self.rank = native, Path(directory), rank
        self.state = ProfileState()
        self.first = self.last = 0
        self.force_at = force_at
        self.degradation = float(os.environ.get('FLASH_V4_MONITOR_DEGRADATION', '0.2'))
        if not 0 < self.degradation < 1:
            raise ValueError('monitor degradation must be between 0 and 1')
        self.closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audit = (self.directory / f'online_rank{rank}.jsonl').open('x', buffering=1)
        self.worker = self.worker_log = None
        self.group = self.payload = self.dist = None
        self.store_control = None
        self.active = False
        self.control_cpu_ms = 0.0
        self.control_wall_ms = 0.0
        self.control_checks = 0
        self.monitor_status = 'initializing'
        self.profile_finalize = profile_finalize
        self.finalizer = None

    def record(self, kind, **fields):
        self.audit.write(json.dumps({'kind': kind, 'rank': self.rank,
                                     'wall_time': time.time(), **fields}) + '\n')

    def initialize(self):
        from flash_v4.execution_budget import monitor_worker_seconds
        worker_seconds = monitor_worker_seconds(os.environ.get('FLASH_V4_EXECUTION_BUDGET'))
        import torch
        import torch.distributed as dist
        self.dist = dist
        control_contract = os.environ.get('FLASH_V4_MONITOR_CONTROL')
        if control_contract == 'r92_store_v1':
            from flash_v4.monitor_store import create_control
            self.store_control = create_control(dist, os.environ.get('FLASH_V4_MONITOR_RUN_ID'), self.rank)
            self.group = self.store_control
            self.scheduling = 'FLASH_v1_Rbeast_request_Store_same_iteration_single_bounded_window'
        elif control_contract is None:
            self.group = dist.new_group(backend='gloo', timeout=timedelta(seconds=30))
        else:
            from flash_v4.monitor_control import create_group
            self.group = create_group(dist, control_contract)
        self.payload = torch.zeros(1, dtype=torch.int64, device='cpu')
        self.record('monitor_start', trigger='manual_integration_only' if self.force_at else 'FLASH_v1_Rbeast',
                    check_stride=8, extra_cuda_events=0, degradation_threshold=self.degradation,
                    monitor_worker_seconds=worker_seconds,
                    control_contract=control_contract or 'legacy_gloo2')
        command = 0
        if self.rank == 0 and not self.force_at:
            try:
                self.worker_log = (self.directory / 'bocd_worker.log').open('x')
                worker_command = [sys.executable, '-m', 'flash_v4.bocd_monitor',
                       '--training-log', str(self.directory.parent / 'training/train.jsonl'),
                       '--request', str(self.directory / 'profile_request.json'),
                       '--audit', str(self.directory / 'bocd_audit.jsonl'),
                       '--stop', str(self.directory / 'monitor_stop'), '--max-seconds', str(worker_seconds),
                       '--degradation', str(self.degradation)]
                self.worker = subprocess.Popen(worker_command, env=child_environment(os.environ),
                                           stdout=self.worker_log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except Exception as error:
                command = -1
                self.record('monitor_error', error=type(error).__name__, at='worker_start')
        if self.rank == 0:
            self.payload[0] = command
        self.exchange_command(0)
        if int(self.payload[0]) < 0:
            raise RuntimeError('BOCD worker startup failed on rank zero')
        self.monitor_status = 'running'

    def exchange_command(self, iteration):
        if self.store_control is None:
            self.dist.broadcast(self.payload, src=0, group=self.group)
        elif self.rank == 0:
            self.store_control.publish(iteration, int(self.payload[0]))
        else:
            self.payload[0] = self.store_control.receive(iteration)

    def begin(self, iteration):
        if self.closed:
            raise RuntimeError('online observer is closed')
        if self.group is None:
            self.initialize()
        if self.state.check_due(iteration):
            cpu, wall = time.process_time(), time.perf_counter()
            command = 0
            if self.rank == 0:
                request = self.directory / 'profile_request.json'
                if self.force_at:
                    command = int(iteration == self.force_at)
                elif request.exists():
                    try:
                        data = json.loads(request.read_text())
                        if data.get('method') != 'flash_v1_Rbeast':
                            raise ValueError('unexpected monitor request method')
                        command = 1
                    except Exception as error:
                        command = -1
                        self.record('monitor_error', error=type(error).__name__, at='request_read')
                if self.worker is not None and self.worker.poll() not in (None, 0):
                    # Spread the error to all ranks before any training collective.
                    command = -1
                self.payload[0] = command
            self.exchange_command(iteration)
            command = int(self.payload[0])
            self.control_cpu_ms += (time.process_time() - cpu) * 1000
            self.control_wall_ms += (time.perf_counter() - wall) * 1000
            self.control_checks += 1
            if command < 0:
                raise RuntimeError('BOCD worker failed; inspect retained worker log')
            if command:
                self.state.start(iteration)
                self.first, self.last = self.state.first, self.state.last
                self.record('profile_start', first=self.first, last=self.last,
                            manual_integration_only=bool(self.force_at))
        self.active = self.state.begin(iteration)
        if self.active:
            self.native.begin_fn(iteration, 1)

    def end(self, iteration):
        if self.active:
            self.native.end_fn(iteration)
        if self.state.end(iteration):
            if self.profile_finalize is not None:
                from flash_v4.profile_finalize import ProfileFinalizer
                import torch
                self.finalizer = ProfileFinalizer(self.native, self.directory,
                    os.environ['FLASH_V4_OUTPUT_DIR'], self.rank, self.first, self.last,
                    torch.cuda.current_device(), getattr(torch.cuda.set_device,
                        '_flash_v4_device_only', torch.cuda.set_device))
                self.finalizer.start()
            else:
                self.native.flush()
                self.record('profile_complete', first=self.first, last=self.last)
                # Historical synchronous contract remains unchanged.
                ready = self.directory / f'ready_rank{self.rank}.json'
                with ready.open('x') as stream:
                    json.dump({'rank': self.rank, 'first': self.first, 'last': self.last,
                               'ready_wall_time': time.time()}, stream)
        self.active = False

    def flush(self):
        if self.closed:
            return
        finalize_error = None
        if self.finalizer is not None:
            try:
                self.finalizer.shutdown()
            except TimeoutError:
                # Native is still active. Do not close audit, reuse native or
                # claim monitor_end. Non-daemon owner + supervisor bound exit.
                self.record('profile_finalize_timeout')
                raise
            except BaseException as error:
                finalize_error = error
            if not self.finalizer.native_entered:
                # Start/device failure: joined thread never touched native.
                self.native.flush()
            if finalize_error is None:
                self.record('profile_complete', first=self.first, last=self.last,
                            finalize_contract=self.profile_finalize)
            else:
                self.record('profile_finalize_error', error=type(finalize_error).__name__)
        try:
            if self.worker is not None:
                (self.directory / 'monitor_stop').touch(exist_ok=True)
                try:
                    self.worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.worker.terminate()
                    try:
                        self.worker.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.worker.kill()
                        self.worker.wait(timeout=2)
                audit_path = self.directory / 'bocd_audit.jsonl'
                try:
                    audit = json.loads(audit_path.read_text())
                    self.monitor_status = ('completed' if self.worker.returncode == 0
                        and audit.get('finish_reason') in ('stop', 'request') else 'failed')
                except (OSError, ValueError):
                    self.monitor_status = 'failed'
            else:
                self.monitor_status = 'manual_integration' if self.force_at else 'peer_completed'
            if self.finalizer is None:
                self.native.flush()
            if finalize_error is not None:
                self.monitor_status = 'failed'
            self.record('monitor_end', profile_requested=bool(self.first),
                        profile_complete=self.state.complete and finalize_error is None, checks=self.control_checks,
                        control_cpu_ms=self.control_cpu_ms, control_wall_ms=self.control_wall_ms,
                        worker_returncode=self.worker.returncode if self.worker else None,
                        monitor_status=self.monitor_status)
        finally:
            self.closed = True
            self.audit.close()
            if self.worker_log:
                self.worker_log.close()
        if finalize_error is not None:
            raise finalize_error
