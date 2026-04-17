"""Benchmark bitwise and integer operations via ``instructions`` counter.

These ops operate on integers, so ``fp_arith_inst_retired`` perf counters
read 0. We use ``perf stat -e instructions`` (total retired instructions)
as the hardware-counter fallback — more stable and deterministic than
wall-clock timing. Falls back to timing if ``perf`` is unavailable.

Also includes ``isnat`` which operates on datetime64 arrays.
"""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_instructions

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

_FORMULA_STRINGS: dict[str, str] = dict.fromkeys(BITWISE_OPS, "n")


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
    seeds[dist_idx]
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

    def _bench_op(
        op: str, setup_fn, bench_code: str, category: str, size_desc: str
    ) -> None:
        """Benchmark a single op across distributions using instructions counter."""
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        for di in range(distributions):
            setup = setup_fn(n, di)
            try:
                result = measure_instructions(setup, bench_code, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))
            dist_raw_totals.append(result.total_flops)
        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": category,
                "measurement_mode": "instructions",
                "analytical_formula": _FORMULA_STRINGS[op],
                "analytical_flops": n,
                "benchmark_size": size_desc,
                "bench_code": bench_code,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    # --- Unary ops ---
    for op in UNARY_OPS:
        _bench_op(op, _unary_setup, f"np.{op}(x)", "instructions_unary", f"x: ({n},)")

    # --- Binary ops ---
    for op in BINARY_OPS:
        setup_fn = _gcd_lcm_setup if op in ("gcd", "lcm") else _binary_setup
        _bench_op(
            op,
            setup_fn,
            f"np.{op}(a, b)",
            "instructions_binary",
            f"a: ({n},), b: ({n},)",
        )

    # --- Shift ops ---
    for op in SHIFT_OPS:
        _bench_op(
            op,
            _shift_setup,
            f"np.{op}(a, b)",
            "instructions_shift",
            f"a: ({n},), b: ({n},) (values 0-10)",
        )

    # --- Special ops ---
    # isnat: operates on datetime64 arrays
    op = "isnat"
    dist_values: list[float] = []
    dist_raw_totals: list[int] = []
    bench = "np.isnat(x)"
    for di in range(distributions):
        setup = _isnat_setup(n, di)
        try:
            result = measure_instructions(setup, bench, repeats=repeats)
        except RuntimeError:
            continue
        dist_values.append(result.total_flops / (n * repeats))
        dist_raw_totals.append(result.total_flops)
    if dist_values:
        results[op] = statistics.median(dist_values)
        details[op] = {
            "category": "instructions_special",
            "measurement_mode": "instructions",
            "analytical_formula": _FORMULA_STRINGS[op],
            "analytical_flops": n,
            "benchmark_size": f"x: ({n},) datetime64 with NaTs",
            "bench_code": bench,
            "repeats": repeats,
            "perf_instructions_total": dist_raw_totals,
            "distribution_alphas": dist_values,
        }

    return results, details
