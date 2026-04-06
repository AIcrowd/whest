"""Benchmark random number generation operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

RANDOM_OPS: list[str] = [
    "random.standard_normal",
    "random.uniform",
    "random.standard_exponential",
    "random.standard_cauchy",
    "random.standard_gamma",
    "random.standard_t",
    "random.poisson",
    "random.binomial",
    "random.permutation",
    "random.shuffle",
]

# Ops that need extra arguments beyond size.
_EXTRA_ARGS: dict[str, str] = {
    "random.uniform": "0.0, 1.0, ",
    "random.standard_gamma": "1.0, ",
    "random.standard_t": "3, ",
    "random.poisson": "5.0, ",
    "random.binomial": "10, 0.5, ",
}


def benchmark_random(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> dict[str, float]:
    """Benchmark random ops, returning FP ops per element.

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
    seeds = [42, 123, 999]

    for op in RANDOM_OPS:
        dist_values: list[float] = []
        extra = _EXTRA_ARGS.get(op, "")

        for seed in seeds:
            setup = f"import numpy as np; np.random.seed({seed})"

            if op == "random.shuffle":
                setup += f"; x = np.arange({n}, dtype=np.{dtype})"
                bench = "np.random.shuffle(x)"
            elif op == "random.permutation":
                bench = f"np.random.permutation({n})"
            else:
                bench = f"np.{op}({extra}{n})"

            result = measure_flops(setup, bench, repeats=repeats)
            dist_values.append(result.total_flops / (n * repeats))

        results[op] = statistics.median(dist_values)

    return results
