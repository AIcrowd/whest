"""Benchmark reduction operations.

Cumulative ops (cumsum, cumprod) use pre-allocated output via ``out=``
to match the pointwise measurement methodology.
"""

from __future__ import annotations

import statistics

import numpy as np

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
    # --- added in Step 2.4 ---
    "nanargmax",
    "nanargmin",
    "ptp",
    "nancumprod",
    "nancumsum",
]

# Add NumPy 2.x array API cumulative ops if available.
if hasattr(np, "cumulative_sum"):
    REDUCTION_OPS.extend(["cumulative_sum", "cumulative_prod"])

# Ops that require an extra argument.
_SPECIAL_ARGS: dict[str, str] = {
    "percentile": ", 50",
    "nanpercentile": ", 50",
    "quantile": ", 0.5",
    "nanquantile": ", 0.5",
}

# Cumulative ops return same-size array — pre-allocate output.
_CUMULATIVE_OPS = {"cumsum", "cumprod", "nancumprod", "nancumsum"}
if hasattr(np, "cumulative_sum"):
    _CUMULATIVE_OPS |= {"cumulative_sum", "cumulative_prod"}
_CUMULATIVE_OPS = frozenset(_CUMULATIVE_OPS)


def benchmark_reductions(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark all reduction ops, returning raw measurement per element.

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
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        measurement per element. ``details`` maps op name to a dict of
        raw benchmark metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    base_setups = [
        f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
        f"import numpy as np; x = np.random.default_rng(42).uniform(0.01, 100, size={n}).astype(np.{dtype})",
        f"import numpy as np; x = np.random.default_rng(42).uniform(-1000, 1000, size={n}).astype(np.{dtype})",
    ]

    for op in REDUCTION_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        extra = _SPECIAL_ARGS.get(op, "")
        bench = ""

        for base_setup in base_setups:
            if op in _CUMULATIVE_OPS:
                setup = base_setup + f"; _out = np.empty({n}, dtype=np.{dtype})"
                bench = f"np.{op}(x, out=_out)"
            else:
                setup = base_setup
                bench = f"np.{op}(x{extra})"
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))
            dist_raw_totals.append(result.total_flops)
        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "counted_reduction",
                "analytical_formula": "numel(input)",
                "analytical_flops": n,
                "benchmark_size": f"n={n}",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
