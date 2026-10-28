"""Low-cost trigger policy for training-rank timing samples."""

from collections import deque
from math import isfinite
from statistics import median
from typing import Any, Deque, Dict, Iterable, Optional, Union


MIN_HEALTHY_SAMPLES = 8
TRIGGER_WINDOW = 3
MAD_SCALE = 1.4826


def _positive_finite(value: Any) -> Optional[float]:
    """Return a numeric sample as float, or None when it is not usable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    if not isfinite(result) or result <= 0.0:
        return None
    return result


def fit_healthy(values: Iterable[float]) -> Dict[str, Union[float, int]]:
    """Fit a fixed robust threshold from at least eight valid healthy samples.

    Invalid, non-finite, and non-positive values are ignored. ``scaled_mad``
    is 1.4826 times the median absolute deviation.
    """
    try:
        samples = []
        for value in values:
            sample = _positive_finite(value)
            if sample is not None:
                samples.append(sample)
    except TypeError as exc:
        raise ValueError("healthy values must be an iterable of numeric samples") from exc
    if len(samples) < MIN_HEALTHY_SAMPLES:
        raise ValueError(f"at least {MIN_HEALTHY_SAMPLES} valid healthy samples are required")

    center = float(median(samples))
    raw_mad = float(median(abs(sample - center) for sample in samples))
    scaled_mad = MAD_SCALE * raw_mad
    threshold = max(1.15 * center, center + 3.5 * scaled_mad)
    return {
        "median": center,
        "scaled_mad": scaled_mad,
        "threshold": threshold,
        "sample_count": len(samples),
    }


def should_trigger(recent: Iterable[float], calibration: Any) -> bool:
    """Trigger only when the last three consecutive samples exceed threshold.

    An invalid or missing value in that trailing window prevents a trigger.
    Only three recent values are retained while scanning ``recent``.
    """
    if not isinstance(calibration, dict):
        return False
    threshold = _positive_finite(calibration.get("threshold"))
    if threshold is None:
        return False

    tail: Deque[Any] = deque(maxlen=TRIGGER_WINDOW)
    try:
        for value in recent:
            tail.append(value)
    except TypeError:
        return False
    if len(tail) != TRIGGER_WINDOW:
        return False

    samples = [_positive_finite(value) for value in tail]
    return all(sample is not None and sample > threshold for sample in samples)
