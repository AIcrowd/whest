"""Benchmark scipy.stats-compatible distribution methods (pdf/cdf/ppf).

These ops use raw numpy internally (no ufunc layer), so measurement_mode
is ``custom`` (no overhead subtraction). The analytical cost is
``numel(input)`` for all methods — the weight captures the per-element
FP cost.
"""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

# 8 distributions × 3 methods = 24 ops
DISTRIBUTIONS = [
    "norm",
    "uniform",
    "expon",
    "cauchy",
    "logistic",
    "laplace",
    "lognorm",
    "truncnorm",
]
METHODS = ["pdf", "cdf", "ppf"]

_FORMULA_STRINGS: dict[str, str] = {
    f"stats.{dist}.{method}": "numel(input)"
    for dist in DISTRIBUTIONS
    for method in METHODS
}


def _setup_code(dist: str, method: str, n: int, di: int) -> str:
    """Build setup code for a stats benchmark.

    The import of the stats distribution object is done HERE in setup
    (not in bench_code) so the import cost is not measured — only the
    actual computation is in the hot loop.
    """
    seeds = [42, 123, 7]
    seed = seeds[di % len(seeds)]

    # Import the distribution in setup so it's available in the bench loop
    import_line = f"from whest.stats import {dist} as _dist"

    if method == "ppf":
        return (
            f"import numpy as np; {import_line}; "
            f"rng = np.random.default_rng({seed}); x = rng.uniform(0.01, 0.99, {n})"
        )
    else:
        setups = [
            f"import numpy as np; {import_line}; x = np.random.default_rng({seed}).standard_normal({n})",
            f"import numpy as np; {import_line}; x = np.random.default_rng({seed}).uniform(-5, 5, {n})",
            f"import numpy as np; {import_line}; x = np.random.default_rng({seed}).uniform(0.01, 10, {n})",
        ]
        return setups[di % len(setups)]


def _bench_code(dist: str, method: str) -> str:
    """Build benchmark code — just the computation, no imports.

    The distribution object is imported as ``_dist`` in setup_code.
    We call _compute_* to avoid budget.deduct overhead in measurement.
    """
    if dist == "lognorm":
        return f"_dist._compute_{method}(x, s=0.5)"
    elif dist == "truncnorm":
        return f"_dist._compute_{method}(x, a=-2, b=2)"
    else:
        return f"_dist._compute_{method}(x)"


def benchmark_stats(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
    **kwargs,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark all stats distribution methods.

    Parameters
    ----------
    n : int
        Array size for benchmarks.
    dtype : str
        Ignored (stats always use float64).
    repeats : int
        Repetitions per distribution.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        (alphas, details) — alphas maps op name to median alpha,
        details maps op name to benchmark metadata.
    """
    distributions = 3
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for dist in DISTRIBUTIONS:
        for method in METHODS:
            op_name = f"stats.{dist}.{method}"
            dist_values: list[float] = []
            dist_raw_totals: list[int] = []

            for di in range(distributions):
                setup = _setup_code(dist, method, n, di)
                bench = _bench_code(dist, method)

                try:
                    result = measure_flops(setup, bench, repeats=repeats)
                except RuntimeError:
                    continue

                # Analytical cost = numel(input) = n
                alpha = result.total_flops / (n * repeats)
                dist_values.append(alpha)
                dist_raw_totals.append(result.total_flops)

            if dist_values:
                results[op_name] = statistics.median(dist_values)
                details[op_name] = {
                    "category": "counted_custom",
                    "measurement_mode": "custom",
                    "analytical_formula": "numel(input)",
                    "analytical_flops": n,
                    "benchmark_size": f"x: ({n},)",
                    "bench_code": _bench_code(dist, method),
                    "repeats": repeats,
                    "perf_instructions_total": dist_raw_totals,
                    "distribution_alphas": dist_values,
                }

    return results, details
