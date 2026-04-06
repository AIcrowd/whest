"""Benchmark polynomial operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

POLYNOMIAL_OPS: list[str] = [
    "polyval",
    "polyfit",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyder",
    "polyint",
    "poly",
    "roots",
]

# Ops that normalize by n (large-array ops).
_NORMALIZE_BY_N = {"polyval", "polyfit"}


def benchmark_polynomial(
    n: int = 1_000_000,
    dtype: str = "float64",
    repeats: int = 10,
    degree: int = 100,
) -> dict[str, float]:
    """Benchmark polynomial ops, returning raw measurement per element.

    For large-array ops (polyval, polyfit): normalizes by n.
    For coefficient ops: uses degree=100 to amortize overhead,
    normalizes by degree+1 (the coefficient array size).

    Parameters
    ----------
    n : int
        Array size for polyval/polyfit.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.
    degree : int
        Polynomial degree (higher = less overhead-dominated for coeff ops).

    Returns
    -------
    dict[str, float]
        Mapping from op name to median raw measurement per element.
    """
    results: dict[str, float] = {}

    # 3 distributions with varying coefficient magnitudes
    coeff_setups = [
        f"c = rng.standard_normal({degree + 1}).astype(np.{dtype})",
        f"c = (rng.standard_normal({degree + 1}) * 100).astype(np.{dtype})",
        f"c = (rng.standard_normal({degree + 1}) * 0.01).astype(np.{dtype})",
    ]

    for op in POLYNOMIAL_OPS:
        dist_values: list[float] = []

        for ci, c_setup in enumerate(coeff_setups):
            seed = 42 + ci
            base_setup = f"import numpy as np; rng = np.random.default_rng({seed}); {c_setup}"

            if op == "polyval":
                setup = (
                    base_setup
                    + f"; x = rng.standard_normal({n}).astype(np.{dtype})"
                )
                bench = "np.polyval(c, x)"
                normalizer = n
            elif op == "polyfit":
                setup = (
                    base_setup
                    + f"; x = np.linspace(-1, 1, {n}).astype(np.{dtype})"
                    + f"; y = np.polyval(c, x) + rng.standard_normal({n}).astype(np.{dtype}) * 0.01"
                )
                bench = f"np.polyfit(x, y, {degree})"
                normalizer = n
            elif op == "poly":
                setup = (
                    base_setup
                    + f"; r = rng.standard_normal({degree}).astype(np.{dtype})"
                )
                bench = "np.poly(r)"
                normalizer = degree
            elif op == "roots":
                setup = base_setup
                bench = "np.roots(c)"
                normalizer = degree
            elif op in ("polyadd", "polysub"):
                setup = (
                    base_setup
                    + f"; d = rng.standard_normal({degree + 1}).astype(np.{dtype})"
                )
                bench = f"np.{op}(c, d)"
                normalizer = degree + 1
            elif op in ("polymul", "polydiv"):
                setup = (
                    base_setup
                    + f"; d = rng.standard_normal({degree + 1}).astype(np.{dtype})"
                )
                bench = f"np.{op}(c, d)"
                normalizer = degree + 1
            elif op == "polyder":
                setup = base_setup
                bench = "np.polyder(c)"
                normalizer = degree + 1
            elif op == "polyint":
                setup = base_setup
                bench = "np.polyint(c)"
                normalizer = degree + 1
            else:
                setup = base_setup
                bench = f"np.{op}(c)"
                normalizer = degree + 1

            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (normalizer * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)

    return results
