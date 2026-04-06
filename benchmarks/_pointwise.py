"""Benchmark pointwise (element-wise) unary and binary operations."""

from __future__ import annotations

import statistics

import numpy as np

from benchmarks._perf import measure_flops

UNARY_OPS: list[str] = [
    "abs",
    "negative",
    "positive",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "cbrt",
    "square",
    "reciprocal",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "ceil",
    "floor",
    "trunc",
    "rint",
    "sign",
    "signbit",
    "fabs",
    "deg2rad",
    "rad2deg",
    "degrees",
    "radians",
    "logical_not",
    "bitwise_invert",
]

BINARY_OPS: list[str] = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "power",
    "float_power",
    "mod",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "arctan2",
    "hypot",
    "copysign",
    "nextafter",
    "logaddexp",
    "logaddexp2",
    "ldexp",
]


def _make_inputs_unary(n: int, dtype: str) -> list[np.ndarray]:
    """Return 3 input arrays with different distributions."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal(n).astype(dtype),
        rng.uniform(0.01, 100, size=n).astype(dtype),
        rng.uniform(-1000, 1000, size=n).astype(dtype),
    ]


def _make_inputs_binary(
    n: int, dtype: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return 3 (a, b) tuples with different distributions."""
    rng = np.random.default_rng(42)
    return [
        (
            rng.standard_normal(n).astype(dtype),
            rng.standard_normal(n).astype(dtype),
        ),
        (
            rng.uniform(0.01, 100, size=n).astype(dtype),
            rng.uniform(0.01, 100, size=n).astype(dtype),
        ),
        (
            rng.uniform(-1000, 1000, size=n).astype(dtype),
            rng.uniform(-1000, 1000, size=n).astype(dtype),
        ),
    ]


def benchmark_pointwise(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
    distributions: int = 3,
) -> dict[str, float]:
    """Benchmark all pointwise ops, returning FP ops per element.

    Parameters
    ----------
    n : int
        Array size.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.
    distributions : int
        Number of input distributions to measure (median is taken).

    Returns
    -------
    dict[str, float]
        Mapping from op name to median FP ops per element.
    """
    results: dict[str, float] = {}

    # --- Unary ops ---
    for op in UNARY_OPS:
        dist_values: list[float] = []
        setups = [
            f"import numpy as np; x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
            f"import numpy as np; x = np.random.default_rng(42).uniform(0.01, 100, size={n}).astype(np.{dtype})",
            f"import numpy as np; x = np.random.default_rng(42).uniform(-1000, 1000, size={n}).astype(np.{dtype})",
        ]
        for setup in setups[:distributions]:
            bench = f"np.{op}(x)"
            result = measure_flops(setup, bench, repeats=repeats)
            dist_values.append(result.total_flops / (n * repeats))
        results[op] = statistics.median(dist_values)

    # --- Binary ops ---
    for op in BINARY_OPS:
        dist_values = []
        setups = [
            (
                f"import numpy as np; rng = np.random.default_rng(42); "
                f"a = rng.standard_normal({n}).astype(np.{dtype}); "
                f"b = rng.standard_normal({n}).astype(np.{dtype})"
            ),
            (
                f"import numpy as np; rng = np.random.default_rng(42); "
                f"a = rng.uniform(0.01, 100, size={n}).astype(np.{dtype}); "
                f"b = rng.uniform(0.01, 100, size={n}).astype(np.{dtype})"
            ),
            (
                f"import numpy as np; rng = np.random.default_rng(42); "
                f"a = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype}); "
                f"b = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype})"
            ),
        ]
        for setup in setups[:distributions]:
            bench = f"np.{op}(a, b)"
            result = measure_flops(setup, bench, repeats=repeats)
            dist_values.append(result.total_flops / (n * repeats))
        results[op] = statistics.median(dist_values)

    return results
