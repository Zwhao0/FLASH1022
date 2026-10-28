"""Megatron-specific runtime adapter for the private native stream-delay actor."""
from __future__ import annotations

import ctypes
import functools
import importlib
import inspect
import os
from pathlib import Path
import stat


SCHEDULES_MODULE = "megatron.core.pipeline_parallel.schedules"


class NativeStreamDelayActor:
    """Small binding for the independently preloaded native actor ABI."""

    def __init__(self, ratio: float, private_log_path: Path, library=None,
                 first: int = 4, last: int = 11):
        if (type(first) is not int or type(last) is not int
                or not (4 <= first <= last <= 120) or last - first + 1 > 32):
            raise ValueError("invalid bounded stream-delay actor window")
        self.first, self.last = first, last
        lib = ctypes.CDLL(None) if library is None else library
        try:
            self.configure_fn = lib.flash_v4_delay_configure
            self.begin_fn = lib.flash_v4_delay_begin
            self.phase_fn = lib.flash_v4_delay_phase
            self.end_fn = lib.flash_v4_delay_end
            self.flush_fn = lib.flash_v4_delay_flush
        except AttributeError:
            raise RuntimeError("native stream-delay actor was not preloaded") from None

        self.configure_fn.argtypes = [ctypes.c_double, ctypes.c_char_p]
        self.configure_fn.restype = ctypes.c_int
        self.begin_fn.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        self.begin_fn.restype = None
        self.phase_fn.argtypes = [ctypes.c_int]
        self.phase_fn.restype = None
        self.end_fn.argtypes = [ctypes.c_uint64]
        self.end_fn.restype = None
        self.flush_fn.argtypes = []
        self.flush_fn.restype = None

        status = self.configure_fn(float(ratio), os.fsencode(private_log_path))
        if status != 0:
            raise RuntimeError(f"native stream-delay actor configure failed ({status})")
        self.closed = False

    def begin(self, iteration: int) -> None:
        if self.closed:
            raise RuntimeError("native stream-delay actor is already closed")
        calibration = int(1 <= iteration <= 3)
        fault_window = int(self.first <= iteration <= self.last)
        self.begin_fn(iteration, calibration, fault_window)

    def phase(self, phase: int) -> None:
        if not self.closed:
            self.phase_fn(phase)

    def end(self, iteration: int) -> None:
        if not self.closed:
            self.end_fn(iteration)

    def flush(self) -> None:
        if not self.closed:
            try:
                self.flush_fn()
            finally:
                self.closed = True


def _forward_only_position(schedule):
    try:
        parameters = list(inspect.signature(schedule).parameters.values())
    except (TypeError, ValueError):
        return None
    for index, parameter in enumerate(parameters):
        if parameter.name == "forward_only" and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return index
    return None


def _is_forward_only(schedule, args, kwargs, position):
    if kwargs.get("forward_only", False):
        return True
    return position is not None and len(args) > position and bool(args[position])


def _wrap_schedule(schedule, actor, call_counter):
    if getattr(schedule, "_flash_v4_stream_delay_schedule", False) is True:
        raise RuntimeError("Megatron schedule already has a stream-delay actor")
    forward_only_position = _forward_only_position(schedule)

    @functools.wraps(schedule)
    def actor_schedule(*args, **kwargs):
        if _is_forward_only(schedule, args, kwargs, forward_only_position):
            raise RuntimeError(
                "stream-delay actor supports training schedules only; "
                "forward_only evaluation was rejected"
            )
        call_counter[0] += 1
        iteration = call_counter[0]
        actor.begin(iteration)
        try:
            return schedule(*args, **kwargs)
        finally:
            actor.end(iteration)

    actor_schedule._flash_v4_stream_delay_schedule = True
    return actor_schedule


def _patch_phase_functions(schedules_module, actor):
    for name, phase in (("forward_step", 1), ("backward_step", 2)):
        function = getattr(schedules_module, name, None)
        if not callable(function):
            raise RuntimeError(f"Megatron {name} audit hook is unavailable")
        if getattr(function, "_flash_v4_stream_delay_phase", False) is True:
            raise RuntimeError(f"Megatron {name} already has a stream-delay phase hook")

        def make_wrapper(original, selected_phase):
            @functools.wraps(original)
            def phase_audit(*args, **kwargs):
                actor.phase(selected_phase)
                try:
                    return original(*args, **kwargs)
                finally:
                    actor.phase(0)

            phase_audit._flash_v4_stream_delay_phase = True
            return phase_audit

        setattr(schedules_module, name, make_wrapper(function, phase))


def _patch_schedule_factory(schedules_module, actor, call_counter):
    factory = getattr(schedules_module, "get_forward_backward_func", None)
    if not callable(factory):
        raise RuntimeError("Megatron schedule factory is unavailable")
    if getattr(factory, "_flash_v4_stream_delay_factory", False) is True:
        raise RuntimeError("Megatron schedule factory already has a stream-delay actor")

    @functools.wraps(factory)
    def actor_factory(*args, **kwargs):
        schedule = factory(*args, **kwargs)
        if not callable(schedule):
            raise RuntimeError("Megatron schedule factory did not return a callable")
        return _wrap_schedule(schedule, actor, call_counter)

    actor_factory._flash_v4_stream_delay_factory = True
    schedules_module.get_forward_backward_func = actor_factory
    _patch_phase_functions(schedules_module, actor)


def install_megatron_schedule_hook(adapter_module, actor, schedule_importer=None):
    """Patch after v1 group setup and before train_pipeline imports schedules."""
    initialize_groups = getattr(adapter_module, "initialize_groups", None)
    if not callable(initialize_groups):
        raise RuntimeError("v1 Megatron adapter has no initialize_groups entry")
    if getattr(initialize_groups, "_flash_v4_stream_delay_init_hook", False) is True:
        raise RuntimeError("v1 Megatron initialize_groups already has a stream-delay hook")
    schedule_importer = schedule_importer or importlib.import_module
    call_counter = [0]
    hook_installed = False

    @functools.wraps(initialize_groups)
    def initialize_groups_then_hook(*args, **kwargs):
        nonlocal hook_installed
        result = initialize_groups(*args, **kwargs)
        if not hook_installed:
            schedules_module = schedule_importer(SCHEDULES_MODULE)
            _patch_schedule_factory(schedules_module, actor, call_counter)
            hook_installed = True
        return result

    initialize_groups_then_hook._flash_v4_stream_delay_init_hook = True
    adapter_module.initialize_groups = initialize_groups_then_hook


def _assert_cuda_not_initialized(training_module) -> None:
    torch_module = getattr(training_module, "torch", None)
    cuda_module = getattr(torch_module, "cuda", None)
    is_initialized = getattr(cuda_module, "is_initialized", None)
    if callable(is_initialized) and is_initialized():
        raise RuntimeError("stream-delay actor must be configured before CUDA initialization")


def install(training_module, startup, library=None):
    """Configure and hook one rank, or return None when actor env is absent."""
    if not startup.stream_delay_enabled:
        return None
    if startup.requested_quota_percent != 100:
        raise RuntimeError("stream-delay actor requires CUDA MPS quota 100 on every rank")
    if startup.actuation_dir is None:
        raise RuntimeError("stream-delay actor requires private actuation evidence storage")

    adapter_module = importlib.import_module("flash_v1.megatron_adapter")
    _assert_cuda_not_initialized(training_module)
    private_dir = Path(startup.actuation_dir)
    private_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(private_dir.stat().st_mode) & 0o077:
        raise RuntimeError("private actuation directory permissions are too broad")
    private_log_path = private_dir / f"stream_delay_rank_{startup.rank}.jsonl"
    actor = NativeStreamDelayActor(
        startup.stream_delay_ratio, private_log_path, library,
        first=getattr(startup, "stream_delay_first", 4),
        last=getattr(startup, "stream_delay_last", 11),
    )
    try:
        install_megatron_schedule_hook(adapter_module, actor)
    except BaseException:
        actor.flush()
        raise
    return actor
