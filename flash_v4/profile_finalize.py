"""Explicit, supervised one-shot PROFILE finalization. No MONITOR polling.

Native CUDA calls cannot be interrupted safely. A non-daemon owner prevents
normal interpreter teardown while native code is active; shared_node supplies
the outer process-group timeout when a native call does not return.
"""
import json
import os
from pathlib import Path
import threading
import time

CONTRACT = 'r68_async_finalize_v1'
SECONDS = 10.0


def validate_contract(contract, mode, supervised):
    if contract is not None and (contract != CONTRACT or mode != 'online'
                                 or supervised != CONTRACT):
        raise ValueError('async PROFILE finalization requires supervised online v1 contract')


def validate_summary(directory, rank):
    path = Path(directory) / f'rank_{rank}.jsonl'
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if not 0 < size <= 1048576:
            raise RuntimeError('invalid native observation size')
        stream.seek(max(0, size - 2048))
        tail = stream.read(2048)
    if not tail.endswith(b'\n'):
        raise RuntimeError('native summary not terminated')
    summary = json.loads(tail.splitlines()[-1])
    if not (summary.get('kind') == 'summary' and summary.get('rank') == rank
            and summary.get('window_complete') is True
            and summary.get('completed_profile_iterations') == 8
            and summary.get('dropped') == summary.get('skipped') == 0
            and summary.get('event_pool', {}).get('cleanup_complete') is True):
        raise RuntimeError('native PROFILE summary incomplete')


class ProfileFinalizer:
    def __init__(self, native, directory, observations, rank, first, last,
                 device, set_device, *, clock=time.monotonic):
        self.native = native
        self.directory, self.observations = Path(directory), Path(observations)
        self.rank, self.first, self.last = rank, first, last
        self.device, self.set_device = device, set_device
        self.clock = clock
        self.deadline = clock() + SECONDS
        self.lock = threading.Lock()
        self.cancelled = False
        self.error = None
        self.published = False
        self.started = False
        self.native_entered = False
        self.thread = threading.Thread(target=self._run, name='flash-profile-finalize', daemon=False)

    def start(self):
        try:
            self.thread.start()
            self.started = True
        except BaseException as error:
            self.error = error
            self.cancel()
            raise

    def cancel(self):
        with self.lock:
            self.cancelled = True

    def _run(self):
        temporary = self.directory / f'.ready_rank{self.rank}.r68.tmp'
        temporary_owned = False
        try:
            # Select the caller's logical device before native clear_cached's
            # first cudaGetDevice. Never infer device from global rank.
            self.set_device(self.device)
            self.native_entered = True
            self.native.flush()
            validate_summary(self.observations, self.rank)
            payload = json.dumps(dict(rank=self.rank, first=self.first, last=self.last,
                                      ready_wall_time=time.time()))
            if len(payload.encode()) > 512:
                raise RuntimeError('ready payload exceeds bound')
            with temporary.open('x') as stream:
                temporary_owned = True
                stream.write(payload)
            # Publication and cancellation have one linearization point.
            # link is atomic, exclusive and on the same filesystem.
            with self.lock:
                if self.cancelled or self.clock() >= self.deadline:
                    raise RuntimeError('PROFILE finalization cancelled or expired')
                os.link(temporary, self.directory / f'ready_rank{self.rank}.json')
                self.published = True
        except BaseException as error:
            self.error = error
        finally:
            if temporary_owned:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass  # A hidden temp cannot be consumed as readiness.

    def shutdown(self):
        # An exit while finalization is unfinished cancels publication. The
        # worker still owns native cleanup; no fallback concurrent flush.
        with self.lock:
            if not self.published:
                self.cancelled = True
        if self.started:
            self.thread.join(max(0.0, self.deadline - self.clock()))
            if self.thread.is_alive():
                raise TimeoutError('PROFILE native finalizer still active; supervisor must terminate')
        if self.error is not None:
            raise RuntimeError('PROFILE finalization failed') from self.error
        if not self.published:
            raise RuntimeError('PROFILE finalization cancelled before readiness')
