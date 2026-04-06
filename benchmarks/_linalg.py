"""Benchmark linear algebra operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

LINALG_OPS: list[str] = [
    "linalg.cholesky",
    "linalg.qr",
    "linalg.eig",
    "linalg.eigh",
    "linalg.eigvals",
    "linalg.eigvalsh",
    "linalg.svd",
    "linalg.svdvals",
    "linalg.solve",
    "linalg.inv",
    "linalg.lstsq",
    "linalg.pinv",
    "linalg.det",
    "linalg.slogdet",
]

# Ops that need symmetric positive-definite matrices.
_SPD_OPS = {"linalg.cholesky", "linalg.eigh", "linalg.eigvalsh"}


def _analytical_cost(op_name: str, n: int) -> int:
    """Return the textbook FLOP count for *op_name* on an (n, n) matrix.

    Parameters
    ----------
    op_name : str
        Operation name (e.g. ``"linalg.cholesky"``).
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Analytical FLOP count.
    """
    m = n  # square matrices
    short = op_name.split(".")[-1]
    costs: dict[str, int] = {
        "cholesky": n**3 // 3,
        "qr": 2 * m * n**2 - 2 * n**3 // 3,
        "eig": 10 * n**3,
        "eigh": 4 * n**3 // 3,
        "eigvals": 7 * n**3,
        "eigvalsh": 4 * n**3 // 3,
        "svd": m * n * min(m, n),
        "svdvals": m * n * min(m, n),
        "solve": 2 * n**3 // 3 + 2 * n**2,
        "inv": n**3,
        "lstsq": m * n * min(m, n),
        "pinv": m * n * min(m, n),
        "det": 2 * n**3 // 3,
        "slogdet": 2 * n**3 // 3,
    }
    return costs[short]


def benchmark_linalg(
    n: int = 1024,
    dtype: str = "float64",
    repeats: int = 10,
) -> dict[str, float]:
    """Benchmark linalg ops, returning raw measurement per analytical FLOP.

    In perf mode this is actual FP ops / analytical FLOPs (correction factor).
    In timing mode this is nanoseconds / analytical FLOPs (same units as
    pointwise — the runner normalizes against baseline to get relative weights).

    Parameters
    ----------
    n : int
        Matrix dimension (n x n).
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    dict[str, float]
        Mapping from op name to median raw measurement per analytical FLOP.
    """
    results: dict[str, float] = {}

    for op in LINALG_OPS:
        dist_values: list[float] = []

        if op in _SPD_OPS:
            # SPD matrices: A@A.T + n*I
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_A = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"A = _A @ _A.T + {n} * np.eye({n}, dtype=np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_A = rng.uniform(0.1, 1.0, size=({n}, {n})).astype(np.{dtype}); "
                    f"A = _A @ _A.T + {n} * np.eye({n}, dtype=np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_A = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"A = _A @ _A.T + {n * 100} * np.eye({n}, dtype=np.{dtype})"
                ),
            ]
        else:
            # General, well-conditioned, ill-conditioned
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal(({n}, {n})).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"A = A + {n} * np.eye({n}, dtype=np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_u = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"_s = np.logspace(0, -10, {n}, dtype=np.{dtype}); "
                    f"A = _u * _s @ _u.T"
                ),
            ]

        # Build bench code
        if op == "linalg.solve":
            bench_suffix = "; b = np.ones({n}, dtype=np.{dtype})".format(
                n=n, dtype=dtype
            )
            bench = f"np.linalg.solve(A, b)"
        elif op == "linalg.lstsq":
            bench_suffix = "; b = np.ones({n}, dtype=np.{dtype})".format(
                n=n, dtype=dtype
            )
            bench = f"np.linalg.lstsq(A, b, rcond=None)"
        else:
            bench_suffix = ""
            bench = f"np.{op}(A)"

        analytical = _analytical_cost(op, n)

        for setup in setups:
            full_setup = setup + bench_suffix
            try:
                result = measure_flops(full_setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            measured = result.total_flops / repeats
            dist_values.append(measured / analytical if analytical else 0.0)

        if dist_values:
            results[op] = statistics.median(dist_values)

    return results
