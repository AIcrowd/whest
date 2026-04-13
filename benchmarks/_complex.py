"""Benchmark complex-number operations.

Most ops use perf mode with complex128 input (they DO retire FP instructions
on complex data).  Two type-check ops (``iscomplexobj``, ``isrealobj``) use
the ``instructions`` counter because they inspect the dtype, not the array
elements (so ``fp_arith_inst_retired`` reads 0).
"""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops, measure_instructions

COMPLEX_OPS: list[str] = [
    "angle",
    "conj",
    "conjugate",
    "imag",
    "real",
    "real_if_close",
    "iscomplex",
    "isreal",
    "sort_complex",
    "iscomplexobj",
    "isrealobj",
]

# Ops that use the ``instructions`` counter instead of ``fp_arith_inst_retired``
# because they inspect the dtype, not the array elements.
_INSTRUCTIONS_OPS: frozenset[str] = frozenset({"iscomplexobj", "isrealobj"})

_FORMULA_STRINGS: dict[str, str] = {
    "angle": "numel(output)",
    "conj": "numel(output)",
    "conjugate": "numel(output)",
    "imag": "numel(output)",
    "real": "numel(output)",
    "real_if_close": "numel(output)",
    "iscomplex": "numel(output)",
    "isreal": "numel(output)",
    "sort_complex": "numel(output)",
    "iscomplexobj": "numel(output)",
    "isrealobj": "numel(output)",
}

# Seeds for the 3 input distributions.
_DIST_SEEDS: list[tuple[int, int]] = [
    (42, 43),
    (100, 101),
    (200, 201),
]


def _complex_setup(n: int, seed_real: int, seed_imag: int) -> str:
    """Build setup code that creates a complex128 array from two RNGs."""
    return (
        f"import numpy as np; "
        f"x = np.random.default_rng({seed_real}).standard_normal({n}).astype(np.float64) "
        f"+ 1j * np.random.default_rng({seed_imag}).standard_normal({n}).astype(np.float64)"
    )


def _real_if_close_setup(n: int, dist_idx: int) -> str:
    """Build setup for ``real_if_close``.

    Distribution 0 has negligible imaginary parts (tests the "close to real"
    path).  Distributions 1 and 2 have substantial imaginary parts.
    """
    if dist_idx == 0:
        # Negligible imaginary part — real_if_close may strip it.
        return (
            f"import numpy as np; "
            f"x = np.random.default_rng(42).standard_normal({n}).astype(np.float64) "
            f"+ 1j * np.random.default_rng(43).standard_normal({n}).astype(np.float64) * 1e-15"
        )
    seed_r, seed_i = _DIST_SEEDS[dist_idx]
    return _complex_setup(n, seed_r, seed_i)


def _timing_setup(n: int) -> str:
    """Build setup for type-check ops (``iscomplexobj`` / ``isrealobj``)."""
    return (
        f"import numpy as np; "
        f"x = np.random.default_rng(42).standard_normal({n}) "
        f"+ 1j * np.random.default_rng(43).standard_normal({n})"
    )


def _bench_code(op: str) -> str:
    """Return the benchmark statement for *op*."""
    return f"np.{op}(x)"


def benchmark_complex(
    n: int = 10_000_000,
    dtype: str = "complex128",
    repeats: int = 10,
    distributions: int = 3,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark complex-number ops, returning raw measurement per element.

    Parameters
    ----------
    n : int
        Array size (element count).
    dtype : str
        NumPy dtype string (unused — always complex128, kept for API parity).
    repeats : int
        Number of repetitions per measurement.
    distributions : int
        Number of input distributions to measure (median is taken).

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        ``(alphas, details)`` — *alphas* maps op name to median measurement
        per analytical FLOP; *details* maps op name to benchmark metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in COMPLEX_OPS:
        # --- Determine n for this op ---
        op_n = 1_000_000 if op == "sort_complex" else n

        # --- Choose measurement function ---
        use_instructions = op in _INSTRUCTIONS_OPS
        measure_fn = measure_instructions if use_instructions else measure_flops

        dist_values: list[float] = []
        dist_raw_totals: list[int] = []
        bench = _bench_code(op)

        for di in range(distributions):
            # --- Build setup code ---
            if use_instructions:
                setup = _timing_setup(op_n)
            elif op == "real_if_close":
                setup = _real_if_close_setup(op_n, di)
            else:
                seed_r, seed_i = _DIST_SEEDS[di]
                setup = _complex_setup(op_n, seed_r, seed_i)

            try:
                result = measure_fn(setup, bench, repeats=repeats)
            except RuntimeError:
                continue

            # Analytical cost = numel(output) = op_n for all complex ops.
            analytical = op_n
            dist_values.append(result.total_flops / (analytical * repeats))
            dist_raw_totals.append(result.total_flops)

        if dist_values:
            results[op] = statistics.median(dist_values)

            if use_instructions:
                bm_size = f"x: ({op_n},) complex128 (instructions counter)"
            else:
                bm_size = f"x: ({op_n},) complex128"

            mm = "instructions" if op in _INSTRUCTIONS_OPS else "ufunc_unary"
            details[op] = {
                "category": "counted_complex",
                "measurement_mode": mm,
                "analytical_formula": _FORMULA_STRINGS[op],
                "analytical_flops": op_n,
                "benchmark_size": bm_size,
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
