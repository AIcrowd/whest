"""Benchmark reduction operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

REDUCTION_OPS: list[str] = [
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "max",
    "min",
    "argmax",
    "argmin",
    "any",
    "all",
    "cumsum",
    "cumprod",
    "nansum",
    "nanmean",
    "nanstd",
    "nanvar",
    "nanmax",
    "nanmin",
    "nanprod",
    "median",
    "nanmedian",
    "percentile",
    "nanpercentile",
    "quantile",
    "nanquantile",
    "count_nonzero",
    "average",
]

# Ops that require an extra argument.
_SPECIAL_ARGS: dict[str, str] = {
    "percentile": ", 50",
    "nanpercentile": ", 50",
    "quantile": ", 0.5",
    "nanquantile": ", 0.5",
}


def benchmark_reductions(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> dict[str, float]:
    """Benchmark all reduction ops, returning FP ops per element.

    Parameters
    ----------
    n : int
        Array size.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    dict[str, float]
        Mapping from op name to median FP ops per element.
    """
    results: dict[str, float] = {}

    setups = [
        f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
        f"import numpy as np; x = np.random.default_rng(42).uniform(0.01, 100, size={n}).astype(np.{dtype})",
        f"import numpy as np; x = np.random.default_rng(42).uniform(-1000, 1000, size={n}).astype(np.{dtype})",
    ]

    for op in REDUCTION_OPS:
        dist_values: list[float] = []
        extra = _SPECIAL_ARGS.get(op, "")
        for setup in setups:
            bench = f"np.{op}(x{extra})"
            result = measure_flops(setup, bench, repeats=repeats)
            dist_values.append(result.total_flops / (n * repeats))
        results[op] = statistics.median(dist_values)

    return results
