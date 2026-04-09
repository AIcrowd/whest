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
) -> dict[str, float]:
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
    dict[str, float]
        Mapping from op name to median FP ops per analytical operation.
    """
    results: dict[str, float] = {}

    for op in SORTING_OPS:
        dist_values: list[float] = []

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
            dist_values.append(result.total_flops / (analytical * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)

    return results
