"""Megatron-specific bounded observation hooks for the v1 training path."""
import ctypes
import functools
import importlib
import inspect
import json
import os
from pathlib import Path


class RuntimeObserver:
    def __init__(self, mode, first=4, length=8, library=None):
        if mode not in ('monitor', 'profile') or first < 1 or length != 8:
            raise ValueError('invalid bounded observation request')
        self.mode, self.first, self.last = mode, first, first + length - 1
        # CDLL(None) requires actual preload; merely dlopening an unused probe
        # after torch has bound its library symbols must not count as profiling.
        lib = ctypes.CDLL(None) if library is None else library
        try:
            self.begin_fn = lib.flash_v4_begin_iteration
            self.end_fn = lib.flash_v4_end_iteration
            self.flush_fn = lib.flash_v4_flush
        except AttributeError:
            raise RuntimeError('v4 observer was not preloaded') from None
        self.begin_fn.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.begin_fn.restype = None
        self.end_fn.argtypes = [ctypes.c_uint64]
        self.end_fn.restype = None
        self.flush_fn.argtypes = []
        self.flush_fn.restype = None
        self.closed = False

    def begin(self, iteration):
        if self.closed:
            raise RuntimeError('observer already closed')
        enabled = self.mode == 'profile' and self.first <= iteration <= self.last
        self.begin_fn(iteration, int(enabled))

    def flush(self):
        if not self.closed:
            self.flush_fn()
            self.closed = True

    def end(self, iteration):
        if self.closed:
            raise RuntimeError('observer already closed')
        self.end_fn(iteration)


def _forward_only_position(schedule):
    try:
        parameters = list(inspect.signature(schedule).parameters.values())
    except (TypeError, ValueError):
        return None
    for index, parameter in enumerate(parameters):
        if parameter.name == 'forward_only' and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return index
    return None


def _wrap_schedule(schedule, observer, call_counter):
    if getattr(schedule, '_flash_v4_observed_megatron_schedule', False) is True:
        return schedule
    forward_only_index = _forward_only_position(schedule)

    @functools.wraps(schedule)
    def observed_schedule(*args, **kwargs):
        forward_only = kwargs.get('forward_only', False)
        if (forward_only_index is not None and len(args) > forward_only_index
                and args[forward_only_index]):
            forward_only = True
        if forward_only:
            raise RuntimeError(
                'v4 Megatron observer supports training schedules only; '
                'forward_only evaluation was rejected'
            )
        call_counter[0] += 1
        observer.begin(call_counter[0])
        result = schedule(*args, **kwargs)
        observer.end(call_counter[0])
        return result

    observed_schedule._flash_v4_observed_megatron_schedule = True
    return observed_schedule


def _patch_schedule_factory(schedules_module, observer, call_counter):
    factory = getattr(schedules_module, 'get_forward_backward_func', None)
    if not callable(factory):
        raise RuntimeError('Megatron schedule factory is unavailable')
    if getattr(factory, '_flash_v4_observed_megatron_factory', False) is True:
        raise RuntimeError('Megatron schedule factory was already adapted')

    @functools.wraps(factory)
    def observed_factory(*args, **kwargs):
        schedule = factory(*args, **kwargs)
        if not callable(schedule):
            raise RuntimeError('Megatron schedule factory did not return a callable')
        return _wrap_schedule(schedule, observer, call_counter)

    observed_factory._flash_v4_observed_megatron_factory = True
    schedules_module.get_forward_backward_func = observed_factory


def install_megatron_schedule_hook(adapter_module, observer, schedule_importer=None):
    """Patch after normal group initialization, before train_pipeline imports the factory."""
    initialize_groups = getattr(adapter_module, 'initialize_groups', None)
    if not callable(initialize_groups):
        raise RuntimeError('v1 Megatron adapter has no initialize_groups entry')
    if getattr(initialize_groups, '_flash_v4_megatron_init_hook', False) is True:
        raise RuntimeError('v1 Megatron initialize_groups is already adapted')
    schedule_importer = schedule_importer or importlib.import_module
    call_counter = [0]
    hook_installed = False

    @functools.wraps(initialize_groups)
    def initialize_groups_then_hook(*args, **kwargs):
        nonlocal hook_installed
        result = initialize_groups(*args, **kwargs)
        if not hook_installed:
            schedules_module = schedule_importer(
                'megatron.core.pipeline_parallel.schedules'
            )
            _patch_schedule_factory(schedules_module, observer, call_counter)
            hook_installed = True
        return result

    initialize_groups_then_hook._flash_v4_megatron_init_hook = True
    adapter_module.initialize_groups = initialize_groups_then_hook


def install(training_module, delay_class):
    """Install the Megatron schedule hook; the v1 delay object is not an observer clock."""
    del training_module, delay_class
    mode = os.environ.get('FLASH_V4_OBSERVE_MODE', 'off')
    if mode == 'off':
        return None
    finalize_contract = os.environ.get('FLASH_V4_PROFILE_FINALIZE')
    if finalize_contract is not None:
        from flash_v4.profile_finalize import validate_contract
        validate_contract(finalize_contract, mode,
                          os.environ.get('FLASH_V4_PROFILE_FINALIZE_SUPERVISED'))
    if mode == 'online':
        from flash_v4.online_runtime import OnlineObserver
        observer = OnlineObserver(
            RuntimeObserver('monitor'), os.environ['FLASH_V4_ONLINE_LOG_DIR'],
            int(os.environ['RANK']), int(os.environ.get('FLASH_V4_FORCE_PROFILE_AT', '0')),
            profile_finalize=finalize_contract)
    else:
        observer = RuntimeObserver(mode, first=int(os.environ.get('FLASH_V4_PROFILE_START', '4')))
    progress_contract = os.environ.get('FLASH_V4_RANK_PROGRESS')
    if progress_contract is not None:
        from flash_v4.rank_progress import CONTRACT, ProgressObserver
        if progress_contract != CONTRACT or mode != 'online':
            raise ValueError('rank progress requires explicit online v1 contract')
        observer = ProgressObserver(observer, os.environ['FLASH_V4_ONLINE_LOG_DIR'],
            int(os.environ['RANK']), int(os.environ['LOCAL_RANK']),
            os.environ['FLASH_V4_PROGRESS_RUN_ID'])
    adapter_module = importlib.import_module('flash_v1.megatron_adapter')
    install_megatron_schedule_hook(adapter_module, observer)

    directory = Path(os.environ['FLASH_V4_OUTPUT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    setup = dict(
        mode=mode,
        runtime_adapter='Megatron-specific pipeline schedule wrapper',
        framework_api='megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
        initialization_boundary='after megatron_adapter.initialize_groups returns',
        scheduling=getattr(observer, 'scheduling', 'fixed_offline_window_not_automatic_detection'),
        collection_scope='successful_training_schedule_calls;excludes_optimizer',
        first_iteration=observer.first,
        last_iteration=observer.last,
        timing_scope='sampled_library_and_launch_envelopes_not_pure_kernel_execution',
        comparison='matched_shapes_and_backend_distributions_not_operator_identity',
    )
    with (directory / f'setup_rank{rank}.json').open('x') as stream:
        json.dump(setup, stream)
    return observer
