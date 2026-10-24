"""Calibrated, stream-local fault work; never touches model tensors."""
import ctypes
from pathlib import Path
from statistics import median


MODES = ('attention_bwd_sleep', 'experts_bwd_sleep',
         'attention_fwd_sleep', 'attention_bwd_compute', 'whole_compute')


class StreamWorkload:
    def __init__(self, torch, engine):
        self.torch, self.engine = torch, engine
        self.prepared = {}
        if engine == 'cuda_sleep':
            self.units_per_ms = None
        elif engine == 'cuda_fp32_fma':
            path = Path(__file__).parent / 'build' / 'libflashcompute.so'
            self.library = ctypes.CDLL(str(path))
            self.library.flash_compute_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                          ctypes.c_int, ctypes.c_int]
            self.library.flash_compute_launch.restype = ctypes.c_int
            self.blocks = 2 * torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            self.output = torch.empty(self.blocks * 128, device='cuda', dtype=torch.float32)
            self.units_per_ms = None
        else:
            raise ValueError(f'unknown workload engine: {engine}')

    def _launch(self, units):
        if self.engine == 'cuda_sleep':
            self.torch.cuda._sleep(units)
        else:
            error = self.library.flash_compute_launch(
                self.torch.cuda.current_stream().cuda_stream,
                self.output.data_ptr(), self.blocks, units)
            if error:
                raise RuntimeError(f'CUDA compute launch failed: {error}')

    def _measure(self, units):
        samples = []
        for _ in range(3):
            start, end = [self.torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            self._launch(units)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return median(samples)

    def prepare(self, milliseconds):
        if milliseconds <= 0:
            raise ValueError('fault delay must be positive')
        if milliseconds in self.prepared:
            return self.prepared[milliseconds]
        if self.units_per_ms is None:
            initial = 2_000_000 if self.engine == 'cuda_sleep' else 4096
            self._launch(initial)  # Warm the kernel before measuring.
            elapsed = self._measure(initial)
            if elapsed <= 0:
                raise ValueError('invalid CUDA workload calibration')
            self.units_per_ms = initial / elapsed
        units = max(1, round(self.units_per_ms * milliseconds))
        observed = self._measure(units)
        for _ in range(3):
            if abs(observed - milliseconds) <= max(.05, milliseconds * .1):
                break
            units = max(1, round(units * milliseconds / max(observed, 1e-9)))
            observed = self._measure(units)
        if abs(observed - milliseconds) > max(.5, milliseconds * .25):
            raise ValueError(f'CUDA workload calibration mismatch: {milliseconds}, {observed}')
        result = dict(units=units, requested_ms=milliseconds, observed_ms=observed,
                      engine=self.engine, measurement_samples=3)
        if self.engine == 'cuda_sleep':
            result['cycles_per_ms'] = units / observed
        else:
            result.update(rounds=units, blocks=self.blocks, threads=128)
        self.prepared[milliseconds] = result
        return result

    def launch(self, milliseconds):
        # Only look up precomputed work; no events, allocations, or synchronization.
        calibration = self.prepared[milliseconds]
        self._launch(calibration['units'])
        return calibration

    def launch_units(self, units):
        # Prepared proportional timing model may issue a variable round count.
        self._launch(units)
