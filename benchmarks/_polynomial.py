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


def _analytical_cost(op: str, n: int, degree: int) -> int:
    """Return the analytical FLOP cost for a polynomial operation.

    These formulas match mechestim's runtime cost model so that the
    benchmark denominator and the budget deduction use the same formula.
    """
    if op == "polyval":
        return 2 * n * degree
    elif op == "polyfit":
        return 2 * n * (degree + 1) ** 2
    elif op == "roots":
        return 10 * degree**3
    elif op in ("polymul", "polydiv"):
        return (degree + 1) ** 2
    elif op in ("polyadd", "polysub"):
        return degree + 1
    elif op in ("polyder", "polyint"):
        return degree
    elif op == "poly":
        return degree**2
    else:
        raise ValueError(f"Unknown polynomial op: {op!r}")


def benchmark_polynomial(
    n: int = 1_000_000,
    dtype: str = "float64",
    repeats: int = 10,
    degree: int = 100,
) -> dict[str, float]:
    """Benchmark polynomial ops, returning raw measurement per element.

    Each op is normalized by its analytical FLOP cost from
    ``_analytical_cost(op, n, degree)`` so the returned value
    represents raw perf-counter FLOPs per analytical FLOP.

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
            base_setup = (
                f"import numpy as np; rng = np.random.default_rng({seed}); {c_setup}"
            )

            if op == "polyval":
                setup = (
                    base_setup + f"; x = rng.standard_normal({n}).astype(np.{dtype})"
                )
                bench = "np.polyval(c, x)"
            elif op == "polyfit":
                setup = (
                    base_setup
                    + f"; x = np.linspace(-1, 1, {n}).astype(np.{dtype})"
                    + f"; y = np.polyval(c, x) + rng.standard_normal({n}).astype(np.{dtype}) * 0.01"
                )
                bench = f"np.polyfit(x, y, {degree})"
            elif op == "poly":
                setup = (
                    base_setup
                    + f"; r = rng.standard_normal({degree}).astype(np.{dtype})"
                )
                bench = "np.poly(r)"
            elif op == "roots":
                setup = base_setup
                bench = "np.roots(c)"
            elif op in ("polyadd", "polysub"):
                setup = (
                    base_setup
                    + f"; d = rng.standard_normal({degree + 1}).astype(np.{dtype})"
                )
                bench = f"np.{op}(c, d)"
            elif op in ("polymul", "polydiv"):
                setup = (
                    base_setup
                    + f"; d = rng.standard_normal({degree + 1}).astype(np.{dtype})"
                )
                bench = f"np.{op}(c, d)"
            elif op == "polyder":
                setup = base_setup
                bench = "np.polyder(c)"
            elif op == "polyint":
                setup = base_setup
                bench = "np.polyint(c)"
            else:
                setup = base_setup
                bench = f"np.{op}(c)"

            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            analytical = _analytical_cost(op, n, degree)
            dist_values.append(result.total_flops / (analytical * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)

    return results
