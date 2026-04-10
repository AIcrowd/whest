"""Benchmark pointwise (element-wise) unary and binary operations.

All benchmarks pre-allocate output arrays and use ``out=`` to eliminate
memory allocation overhead from measurements, isolating pure compute cost.
"""

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
    # --- added in Step 2.3 ---
    "frexp",
    "modf",
    "sinc",
    "i0",
    "spacing",
    "nan_to_num",
    "isneginf",
    "isposinf",
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

# Special pointwise ops that don't follow the standard unary/binary pattern.
SPECIAL_OPS: list[str] = [
    "isclose",
    "heaviside",
    "clip",
]

# Ops whose output dtype is bool (need bool pre-allocation for out=)
_BOOL_UNARY = frozenset({"signbit", "logical_not", "isneginf", "isposinf"})
_BOOL_BINARY = frozenset(
    {
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "equal",
        "not_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
    }
)

# Ops that return tuples — benchmark without out= parameter.
_TUPLE_RETURN_OPS = frozenset({"frexp", "modf"})

# Ops that require positive input.
_POSITIVE_INPUT_OPS = frozenset({"i0"})

# Ops that benefit from NaN/inf values in input.
_NAN_INPUT_OPS = frozenset({"nan_to_num"})


def _make_inputs_unary(n: int, dtype: str) -> list[np.ndarray]:
    """Return 3 input arrays with different distributions."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal(n).astype(dtype),
        rng.uniform(0.01, 100, size=n).astype(dtype),
        rng.uniform(-1000, 1000, size=n).astype(dtype),
    ]


def _make_inputs_binary(n: int, dtype: str) -> list[tuple[np.ndarray, np.ndarray]]:
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


# sin/cos have a fast path for small inputs (near 0) that skips range
# reduction.  Use uniform(-100, 100) instead of standard_normal for
# distribution 0 so all three distributions exercise the full code path.
_WIDE_INPUT_OPS = frozenset({"sin", "cos"})


def _unary_setup(n: int, dtype: str, op: str, dist_idx: int) -> str:
    """Build setup code for a unary op with pre-allocated output."""
    if op in _WIDE_INPUT_OPS and dist_idx == 0:
        dists_0 = f"x = np.random.default_rng(42).uniform(-100, 100, size={n}).astype(np.{dtype})"
    else:
        dists_0 = f"x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})"
    dists = [
        dists_0,
        f"x = np.random.default_rng(42).uniform(0.01, 100, size={n}).astype(np.{dtype})",
        f"x = np.random.default_rng(42).uniform(-1000, 1000, size={n}).astype(np.{dtype})",
    ]
    setup = f"import numpy as np; {dists[dist_idx]}"

    # i0 only works on positive input.
    if op in _POSITIVE_INPUT_OPS:
        setup += "; x = np.abs(x)"

    # nan_to_num benefits from NaN/inf values in one distribution.
    if op in _NAN_INPUT_OPS and dist_idx == 0:
        setup += (
            f"; x[:{n}//100] = np.nan"
            f"; x[{n}//100:{n}//50] = np.inf"
            f"; x[{n}//50:{n}//50+{n}//100] = -np.inf"
        )

    # Tuple-return ops (frexp, modf) don't use out=.
    if op in _TUPLE_RETURN_OPS:
        return setup

    out_dtype = "bool" if op in _BOOL_UNARY else f"np.{dtype}"
    setup += f"; _out = np.empty({n}, dtype={out_dtype})"
    return setup


def _binary_setup(n: int, dtype: str, op: str, dist_idx: int) -> str:
    """Build setup code for a binary op with pre-allocated output."""
    dists = [
        (
            f"rng = np.random.default_rng(42); "
            f"a = rng.standard_normal({n}).astype(np.{dtype}); "
            f"b = rng.standard_normal({n}).astype(np.{dtype})"
        ),
        (
            f"rng = np.random.default_rng(42); "
            f"a = rng.uniform(0.01, 100, size={n}).astype(np.{dtype}); "
            f"b = rng.uniform(0.01, 100, size={n}).astype(np.{dtype})"
        ),
        (
            f"rng = np.random.default_rng(42); "
            f"a = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype}); "
            f"b = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype})"
        ),
    ]
    out_dtype = "bool" if op in _BOOL_BINARY else f"np.{dtype}"
    return (
        f"import numpy as np; {dists[dist_idx]}; "
        f"_out = np.empty({n}, dtype={out_dtype})"
    )


def benchmark_pointwise(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
    distributions: int = 3,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark all pointwise ops, returning raw measurement per element.

    All operations use pre-allocated output (``out=``) to eliminate memory
    allocation overhead from measurements, isolating pure compute cost.

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
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        measurement per element. ``details`` maps op name to a dict of
        raw benchmark metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    # --- Unary ops ---
    for op in UNARY_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        bench = ""
        for di in range(distributions):
            setup = _unary_setup(n, dtype, op, di)
            if op in _TUPLE_RETURN_OPS:
                bench = f"np.{op}(x)"
            else:
                bench = f"np.{op}(x, out=_out)"
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))
            dist_raw_totals.append(result.total_flops)
        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "counted_unary",
                "analytical_formula": "numel(output)",
                "analytical_flops": n,
                "benchmark_size": f"n={n}",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    # --- Binary ops ---
    for op in BINARY_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        bench = ""
        for di in range(distributions):
            setup = _binary_setup(n, dtype, op, di)
            bench = f"np.{op}(a, b, out=_out)"
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))
            dist_raw_totals.append(result.total_flops)
        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "counted_binary",
                "analytical_formula": "numel(output)",
                "analytical_flops": n,
                "benchmark_size": f"n={n}",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    # --- Special ops (non-standard patterns) ---
    for op in SPECIAL_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        bench = ""
        for di in range(distributions):
            if op == "isclose":
                # Binary comparison returning bool.
                setup = _binary_setup(n, dtype, op, di)
                bench = "np.isclose(a, b)"
                category = "counted_binary"
            elif op == "heaviside":
                # Binary with scalar second argument.
                setup = _unary_setup(n, dtype, op, di)
                bench = "np.heaviside(x, 0.5)"
                category = "counted_binary"
            elif op == "clip":
                # Ternary: clip(x, min, max).
                setup = _unary_setup(n, dtype, op, di)
                bench = "np.clip(x, -1.0, 1.0)"
                category = "counted_unary"
            else:
                continue
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))
            dist_raw_totals.append(result.total_flops)
        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": category,
                "analytical_formula": "numel(output)",
                "analytical_flops": n,
                "benchmark_size": f"n={n}",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
