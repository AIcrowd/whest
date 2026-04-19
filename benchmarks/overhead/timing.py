"""Timing primitives for overhead benchmarks."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns
from typing import Callable, Sequence


@dataclass(frozen=True)
class SampleSummary:
    """Summary statistics for a set of timing samples."""

    best_ns: int
    median_ns: float
    sample_count: int


def summarize_samples(samples: Sequence[int]) -> SampleSummary:
    """Return the best and median timing for the given samples."""
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    return SampleSummary(
        best_ns=ordered[0],
        median_ns=median(ordered),
        sample_count=len(ordered),
    )


def _time_iterations(func: Callable[[], object], iterations: int) -> int:
    start = perf_counter_ns()
    for _ in range(iterations):
        func()
    return perf_counter_ns() - start


def calibrate_iterations(
    func: Callable[[], object],
    minimum_elapsed_ns: int = 200_000,
    max_iterations: int = 1 << 20,
) -> tuple[int, int]:
    """Find a batch size that runs long enough for stable measurements."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    iterations = 1
    elapsed_ns = _time_iterations(func, iterations)
    while elapsed_ns < minimum_elapsed_ns and iterations < max_iterations:
        iterations = min(iterations * 2, max_iterations)
        elapsed_ns = _time_iterations(func, iterations)
    return iterations, elapsed_ns


def measure_samples(
    func: Callable[[], object],
    warmup_samples: int,
    measured_samples: int,
    minimum_elapsed_ns: int,
) -> tuple[list[int], int]:
    """Measure repeated batches with warmup and GC management."""
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for _ in range(warmup_samples):
            func()

        iterations, _ = calibrate_iterations(
            func, minimum_elapsed_ns=minimum_elapsed_ns
        )
        samples = [
            _time_iterations(func, iterations) for _ in range(measured_samples)
        ]
        return samples, iterations
    finally:
        if was_enabled:
            gc.enable()
