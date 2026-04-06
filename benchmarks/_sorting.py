"""Benchmark sorting and related operations."""

from __future__ import annotations

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
]


def benchmark_sorting(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> dict[str, float]:
    """Benchmark sorting ops, returning FP ops per element.

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
        elif op in ("partition", "argpartition"):
            kth = n // 2
            setups = [
                f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))[::-1].copy()",
            ]
            bench = f"np.{op}(x, {kth})"
        else:
            # sort, argsort, unique
            setups = [
                f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))",
                f"import numpy as np; x = np.sort(np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}))[::-1].copy()",
            ]
            bench = f"np.{op}(x)"

        for setup in setups:
            result = measure_flops(setup, bench, repeats=repeats)
            dist_values.append(result.total_flops / (n * repeats))

        results[op] = statistics.median(dist_values)

    return results
