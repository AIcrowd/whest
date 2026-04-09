"""Benchmark window function operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

WINDOW_OPS: list[str] = ["bartlett", "blackman", "hamming", "hanning", "kaiser"]

_ANALYTICAL_COST: dict[str, int] = {
    "bartlett": 1,  # multiplied by n
    "blackman": 3,  # multiplied by n
    "hamming": 1,  # multiplied by n
    "hanning": 1,  # multiplied by n
    "kaiser": 3,  # multiplied by n
}


def benchmark_window(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> dict[str, float]:
    """Benchmark window ops, returning alpha(op) = measured/analytical.

    Parameters
    ----------
    n : int
        Window size.
    dtype : str
        Accepted for interface consistency; window functions always produce
        float64.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    dict[str, float]
        Mapping from op name to median FP ops per analytical operation.
    """
    results: dict[str, float] = {}

    for op in WINDOW_OPS:
        dist_values: list[float] = []
        analytical = _ANALYTICAL_COST[op] * n

        # Window functions are deterministic (no input distribution),
        # but we still take 3 measurements with different seeds for
        # consistency (they'll be nearly identical).
        for seed in [42, 123, 999]:
            setup = "import numpy as np"
            if op == "kaiser":
                bench = f"np.kaiser({n}, 14.0)"
            else:
                bench = f"np.{op}({n})"

            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (analytical * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)

    return results
