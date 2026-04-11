"""Benchmark bitwise and integer operations (timing mode only).

These ops operate on integers, so ``fp_arith_inst_retired`` perf counters
read 0. We force timing mode (``MECHESTIM_FORCE_TIMING=1``) and measure
wall-clock time per element instead.

Also includes ``isnat`` which operates on datetime64 arrays.
"""

from __future__ import annotations

import os
import statistics

from benchmarks._perf import measure_flops

# --- Operation lists -------------------------------------------------------

UNARY_OPS: list[str] = [
    "bitwise_not",
    "bitwise_invert",
    "bitwise_count",
    "invert",
]

BINARY_OPS: list[str] = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "gcd",
    "lcm",
]

SHIFT_OPS: list[str] = [
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
]

SPECIAL_OPS: list[str] = [
    "isnat",
]

BITWISE_OPS: list[str] = UNARY_OPS + BINARY_OPS + SHIFT_OPS + SPECIAL_OPS

# --- Analytical formula strings (all cost = n) ----------------------------

_FORMULA_STRINGS: dict[str, str] = {op: "n" for op in BITWISE_OPS}


def _analytical_cost(op: str, n: int) -> int:  # noqa: ARG001
    """Return analytical FLOP cost for *op* on arrays of length *n*."""
    return n


# --- Setup helpers ----------------------------------------------------------


def _unary_setup(n: int, dist_idx: int) -> str:
    """Build setup code for a unary integer op."""
    seeds = [42, 123, 7]
    seed = seeds[dist_idx]
    return (
        f"import numpy as np; "
        f"x = np.random.default_rng({seed}).integers(-1_000_000, 1_000_000, "
        f"size={n}, dtype=np.int64)"
    )


def _binary_setup(n: int, dist_idx: int) -> str:
    """Build setup code for a binary integer op."""
    seeds = [42, 123, 7]
    seed = seeds[dist_idx]
    return (
        f"import numpy as np; "
        f"rng = np.random.default_rng({seed}); "
        f"a = rng.integers(-1_000_000, 1_000_000, size={n}, dtype=np.int64); "
        f"b = rng.integers(-1_000_000, 1_000_000, size={n}, dtype=np.int64)"
    )


def _gcd_lcm_setup(n: int, dist_idx: int) -> str:
    """Build setup code for gcd/lcm (positive integers)."""
    seeds = [42, 123, 7]
    seed = seeds[dist_idx]
    return (
        f"import numpy as np; "
        f"rng = np.random.default_rng({seed}); "
        f"a = rng.integers(1, 1_000_000, size={n}, dtype=np.int64); "
        f"b = rng.integers(1, 1_000_000, size={n}, dtype=np.int64)"
    )


def _shift_setup(n: int, dist_idx: int) -> str:
    """Build setup code for shift ops (second operand 0-10)."""
    seeds = [42, 123, 7]
    seed = seeds[dist_idx]
    return (
        f"import numpy as np; "
        f"rng = np.random.default_rng({seed}); "
        f"a = rng.integers(-1_000_000, 1_000_000, size={n}, dtype=np.int64); "
        f"b = rng.integers(0, 11, size={n}, dtype=np.int64)"
    )


def _isnat_setup(n: int, dist_idx: int) -> str:
    """Build setup code for isnat (datetime64 input with some NaTs)."""
    seeds = [42, 123, 7]
    seed = seeds[dist_idx]
    # Create datetime64 array with ~1/3 NaT values
    return (
        f"import numpy as np; "
        f"x = np.array(['2020-01-01', 'NaT', '2020-06-15'] * ({n} // 3), "
        f"dtype='datetime64')"
    )


# --- Main benchmark function -----------------------------------------------


def benchmark_bitwise(
    n: int = 10_000_000,
    dtype: str = "int64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark bitwise/integer ops using timing mode only.

    Parameters
    ----------
    n : int
        Array size.
    dtype : str
        Ignored (always uses int64 for bitwise ops). Kept for interface
        consistency with other benchmark modules.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        timing per element (nanoseconds). ``details`` maps op name to a
        dict of raw benchmark metadata.
    """
    distributions = 3
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    # Force timing mode — perf counters read 0 for integer ops.
    orig = os.environ.get("MECHESTIM_FORCE_TIMING")
    os.environ["MECHESTIM_FORCE_TIMING"] = "1"
    try:
        # --- Unary ops ---
        for op in UNARY_OPS:
            dist_values: list[float] = []
            dist_raw_totals: list[int] = []
            bench = ""
            for di in range(distributions):
                setup = _unary_setup(n, di)
                bench = f"np.{op}(x)"
                try:
                    result = measure_flops(setup, bench, repeats=repeats)
                except RuntimeError:
                    continue
                dist_values.append(result.total_flops / (n * repeats))
                dist_raw_totals.append(result.total_flops)
            if dist_values:
                results[op] = statistics.median(dist_values)
                details[op] = {
                    "category": "timed_unary",
                    "analytical_formula": _FORMULA_STRINGS[op],
                    "analytical_flops": n,
                    "benchmark_size": f"x: ({n},)",
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
                if op in ("gcd", "lcm"):
                    setup = _gcd_lcm_setup(n, di)
                else:
                    setup = _binary_setup(n, di)
                bench = f"np.{op}(a, b)"
                try:
                    result = measure_flops(setup, bench, repeats=repeats)
                except RuntimeError:
                    continue
                dist_values.append(result.total_flops / (n * repeats))
                dist_raw_totals.append(result.total_flops)
            if dist_values:
                results[op] = statistics.median(dist_values)
                details[op] = {
                    "category": "timed_binary",
                    "analytical_formula": _FORMULA_STRINGS[op],
                    "analytical_flops": n,
                    "benchmark_size": f"a: ({n},), b: ({n},)",
                    "bench_code": bench,
                    "repeats": repeats,
                    "perf_instructions_total": dist_raw_totals,
                    "distribution_alphas": dist_values,
                }

        # --- Shift ops ---
        for op in SHIFT_OPS:
            dist_values: list[float] = []
            dist_raw_totals: list[int] = []
            bench = ""
            for di in range(distributions):
                setup = _shift_setup(n, di)
                bench = f"np.{op}(a, b)"
                try:
                    result = measure_flops(setup, bench, repeats=repeats)
                except RuntimeError:
                    continue
                dist_values.append(result.total_flops / (n * repeats))
                dist_raw_totals.append(result.total_flops)
            if dist_values:
                results[op] = statistics.median(dist_values)
                details[op] = {
                    "category": "timed_shift",
                    "analytical_formula": _FORMULA_STRINGS[op],
                    "analytical_flops": n,
                    "benchmark_size": f"a: ({n},), b: ({n},) (values 0-10)",
                    "bench_code": bench,
                    "repeats": repeats,
                    "perf_instructions_total": dist_raw_totals,
                    "distribution_alphas": dist_values,
                }

        # --- Special ops ---
        # isnat: operates on datetime64 arrays
        op = "isnat"
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        bench = ""
        for di in range(distributions):
            setup = _isnat_setup(n, di)
            bench = "np.isnat(x)"
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))
            dist_raw_totals.append(result.total_flops)
        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "timed_special",
                "analytical_formula": _FORMULA_STRINGS[op],
                "analytical_flops": n,
                "benchmark_size": f"x: ({n},) datetime64 with NaTs",
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    finally:
        # Restore original MECHESTIM_FORCE_TIMING value.
        if orig is None:
            os.environ.pop("MECHESTIM_FORCE_TIMING", None)
        else:
            os.environ["MECHESTIM_FORCE_TIMING"] = orig

    return results, details
