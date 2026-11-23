"""Primitivas leves de profiling por subrotina."""

from __future__ import annotations

from dataclasses import dataclass
import time
import tracemalloc
from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class ProfileResult(Generic[R]):
    value: R
    elapsed_seconds: float
    peak_bytes: int


def profile_call(function: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> ProfileResult[R]:
    """Executa uma chamada e mede tempo de parede e pico de alocação Python."""

    was_tracing = tracemalloc.is_tracing()
    baseline_peak = tracemalloc.get_traced_memory()[1] if was_tracing else 0
    if not was_tracing:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        value = function(*args, **kwargs)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        peak = max(0, peak - baseline_peak)
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return ProfileResult(value, elapsed, peak)
