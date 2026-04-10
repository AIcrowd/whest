"""Benchmark sorting and related operations."""

from __future__ import annotations

import math
import statistics

from benchmarks._perf import measure_flops

SORTING_OPS: list[str] = [
    "sort",
    "argsort",
    "lexsort",
    "partition",
    "argpartition",
    "searchsorted",
    "unique",
    # Set operations
    "in1d",
    "isin",
    "intersect1d",
    "setdiff1d",
    "setxor1d",
    "union1d",
    # NumPy 2.x unique variants
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
]

_FORMULA_STRINGS: dict[str, str] = {
    "sort": "n * ceil(log2(n))",
    "argsort": "n * ceil(log2(n))",
    "unique": "n * ceil(log2(n))",
    "unique_all": "n * ceil(log2(n))",
    "unique_counts": "n * ceil(log2(n))",
    "unique_inverse": "n * ceil(log2(n))",
    "unique_values": "n * ceil(log2(n))",
    "lexsort": "k * n * ceil(log2(n))",
    "searchsorted": "m * ceil(log2(n))",
    "partition": "n",
    "argpartition": "n",
    "in1d": "(n+m) * ceil(log2(n+m))",
    "isin": "(n+m) * ceil(log2(n+m))",
    "intersect1d": "(n+m) * ceil(log2(n+m))",
    "setdiff1d": "(n+m) * ceil(log2(n+m))",
    "setxor1d": "(n+m) * ceil(log2(n+m))",
    "union1d": "(n+m) * ceil(log2(n+m))",
}


def _analytical_cost(op: str, n: int, **kwargs: int) -> int:
    """Return the analytical operation count for a sorting-related op.

    Parameters
    ----------
    op : str
        Operation name (e.g. ``"sort"``, ``"lexsort"``).
    n : int
        Primary array size.
    **kwargs
        Extra parameters: ``k`` for lexsort key count, ``m`` for second
        array size in set operations / searchsorted queries.

    Returns
    -------
    int
        Analytical FP-operation count used as the normalization denominator.
    """
    nlogn = n * math.ceil(math.log2(n)) if n > 1 else n

    if op in ("sort", "argsort", "unique"):
        return nlogn

    if op in ("unique_all", "unique_counts", "unique_inverse", "unique_values"):
        return nlogn

    if op == "lexsort":
        k = kwargs.get("k", 2)
        return k * nlogn

    if op == "searchsorted":
        m = kwargs.get("m", n)
        return m * math.ceil(math.log2(n)) if n > 1 else m

    if op in ("partition", "argpartition"):
        return n

    # Set operations: in1d, isin, intersect1d, setdiff1d, setxor1d, union1d
    if op in ("in1d", "isin", "intersect1d", "setdiff1d", "setxor1d", "union1d"):
        m = kwargs.get("m", n)
        total = n + m
        return total * math.ceil(math.log2(total)) if total > 1 else total

    # Fallback: per-element
    return n


def benchmark_sorting(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark sorting ops, returning FP ops per analytical operation.

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
        ``(alphas, details)`` where *alphas* maps op name to median alpha
        and *details* maps op name to a dict of per-op measurement metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in SORTING_OPS:
        dist_values: list[float] = []
        perf_instructions: list[int] = []

        if op == "lexsort":
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"x = rng.standard_normal({n}).astype(np.{dtype}); "
                    f"y = rng.standard_normal({n}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; "
                    f"x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})); "
                    f"y = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))"
                ),
                (
                    f"import numpy as np; "
                    f"x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))[::-1].copy(); "
                    f"y = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))[::-1].copy()"
                ),
            ]
            bench = "np.lexsort((x, y))"
            analytical = _analytical_cost(op, n, k=2)
        elif op == "searchsorted":
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"x = np.sort(rng.standard_normal({n}).astype(np.{dtype})); "
                    f"q = rng.standard_normal({n}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; "
                    f"x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})); "
                    f"q = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))"
                ),
                (
                    f"import numpy as np; "
                    f"x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})); "
                    f"q = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))[::-1].copy()"
                ),
            ]
            bench = "np.searchsorted(x, q)"
            analytical = _analytical_cost(op, n, m=n)
        elif op in ("partition", "argpartition"):
            kth = n // 2
            setups = [
                f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))[::-1].copy()",
            ]
            bench = f"np.{op}(x, {kth})"
            analytical = _analytical_cost(op, n)
        elif op in ("in1d", "isin"):
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.standard_normal({n}).astype(np.{dtype}); "
                    f"b = rng.standard_normal({n}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = np.sort(rng.standard_normal({n}).astype(np.{dtype})); "
                    f"b = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = np.sort(rng.standard_normal({n}).astype(np.{dtype}))[::-1].copy(); "
                    f"b = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))[::-1].copy()"
                ),
            ]
            bench = f"np.{op}(a, b)"
            analytical = _analytical_cost(op, n, m=n)
        elif op in ("intersect1d", "setdiff1d", "setxor1d", "union1d"):
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = rng.standard_normal({n}).astype(np.{dtype}); "
                    f"b = rng.standard_normal({n}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = np.sort(rng.standard_normal({n}).astype(np.{dtype})); "
                    f"b = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"a = np.sort(rng.standard_normal({n}).astype(np.{dtype}))[::-1].copy(); "
                    f"b = np.sort(np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}))[::-1].copy()"
                ),
            ]
            bench = f"np.{op}(a, b)"
            analytical = _analytical_cost(op, n, m=n)
        elif op in ("unique_all", "unique_counts", "unique_inverse", "unique_values"):
            setups = [
                f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))[::-1].copy()",
            ]
            bench = f"np.{op}(x)"
            analytical = _analytical_cost(op, n)
        else:
            # sort, argsort, unique
            setups = [
                f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))[::-1].copy()",
            ]
            bench = f"np.{op}(x)"
            analytical = _analytical_cost(op, n)

        for setup in setups:
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            perf_instructions.append(result.total_flops)
            dist_values.append(result.total_flops / (analytical * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)
            # Find the perf_instructions corresponding to the median alpha
            median_idx = dist_values.index(statistics.median(dist_values))
            details[op] = {
                "category": "counted_custom",
                "analytical_formula": _FORMULA_STRINGS.get(op, "n"),
                "analytical_flops": analytical,
                "benchmark_size": f"n={n}",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": perf_instructions[median_idx],
                "distribution_alphas": dist_values,
            }

    return results, details
