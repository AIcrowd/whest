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

_FORMULA_STRINGS: dict[str, str] = {
    "bartlett": "n",
    "blackman": "3*n",
    "hamming": "n",
    "hanning": "n",
    "kaiser": "3*n",
}


def benchmark_window(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
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
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        alpha(op). ``details`` maps op name to a dict of raw benchmark
        metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in WINDOW_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        analytical = _ANALYTICAL_COST[op] * n
        bench = ""

        # Window functions are deterministic (no input distribution),
        # but we still take 3 measurements with different seeds for
        # consistency (they'll be nearly identical).
        for _seed in [42, 123, 999]:
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
            dist_raw_totals.append(result.total_flops)

        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "counted_custom",
                "measurement_mode": "custom",
                "analytical_formula": _FORMULA_STRINGS[op],
                "analytical_flops": analytical,
                "benchmark_size": f"output: ({n},)",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
