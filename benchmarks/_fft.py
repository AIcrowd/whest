"""Benchmark FFT operations."""

from __future__ import annotations

import math
import statistics

from benchmarks._perf import measure_flops

FFT_OPS: list[str] = [
    "fft.fft",
    "fft.ifft",
    "fft.rfft",
    "fft.irfft",
    "fft.fft2",
    "fft.ifft2",
    "fft.rfft2",
    "fft.irfft2",
    "fft.fftn",
    "fft.ifftn",
    "fft.rfftn",
    "fft.irfftn",
    "fft.hfft",
    "fft.ihfft",
]

_RFFT_OPS = {
    "fft.rfft",
    "fft.irfft",
    "fft.rfft2",
    "fft.irfft2",
    "fft.rfftn",
    "fft.irfftn",
}

_FORMULA_STRINGS: dict[str, str] = {
    "fft.fft": "5*n*ceil(log2(n))",
    "fft.ifft": "5*n*ceil(log2(n))",
    "fft.rfft": "5*(n/2)*ceil(log2(n))",
    "fft.irfft": "5*(n/2)*ceil(log2(n))",
    "fft.fft2": "5*n*ceil(log2(n))",
    "fft.ifft2": "5*n*ceil(log2(n))",
    "fft.rfft2": "5*(n/2)*ceil(log2(n))",
    "fft.irfft2": "5*(n/2)*ceil(log2(n))",
    "fft.fftn": "5*n*ceil(log2(n))",
    "fft.ifftn": "5*n*ceil(log2(n))",
    "fft.rfftn": "5*(n/2)*ceil(log2(n))",
    "fft.irfftn": "5*(n/2)*ceil(log2(n))",
    "fft.hfft": "5*n*ceil(log2(n))",
    "fft.ihfft": "5*(n/2)*ceil(log2(n))",
}


def _ceil_log2(n: int) -> int:
    """Return ceil(log2(n)), minimum 1."""
    if n <= 1:
        return 1
    return math.ceil(math.log2(n))


def _analytical_cost(op_name: str, n: int) -> int:
    """Return the analytical FLOP count for an FFT operation.

    Parameters
    ----------
    op_name : str
        Operation name (e.g. ``"fft.fft"``).
    n : int
        Input size.

    Returns
    -------
    int
        Analytical FLOP count.
    """
    cl2 = _ceil_log2(n)
    if op_name in _RFFT_OPS:
        return 5 * (n // 2) * cl2
    return 5 * n * cl2


def benchmark_fft(
    n: int = 2**20,
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark FFT ops, returning correction factors (measured / analytical).

    Parameters
    ----------
    n : int
        Input size.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        correction factor. ``details`` maps op name to a dict of raw
        benchmark metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    # 2D/nD ops use sqrt(n) x sqrt(n)
    side = int(math.isqrt(n))

    for op in FFT_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []

        # Determine dimensionality
        is_2d = "2" in op.split(".")[-1]
        is_nd = op.endswith("n") or op.endswith("fftn")
        is_multi = is_2d or is_nd

        if is_multi:
            shape_str = f"({side}, {side})"
            effective_n = side * side
            benchmark_size = f"x: ({side},{side})"
        else:
            shape_str = f"({n},)"
            effective_n = n
            benchmark_size = f"x: ({n},)"

        # Determine input type needed
        short = op.split(".")[-1]

        # irfft variants need rfft output (complex, half-size)
        needs_rfft_input = short in ("irfft", "irfft2", "irfftn")
        # ifft variants and hfft need complex input
        needs_complex = short in ("ifft", "ifft2", "ifftn", "hfft")

        setups = []
        if needs_rfft_input:
            # Generate input by applying rfft to real data
            rfft_func = short.replace("i", "", 1)  # irfft -> rfft
            setups = [
                f"import numpy as np; _r = np.random.default_rng(42).standard_normal({shape_str}).astype(np.{dtype}); x = np.fft.{rfft_func}(_r)",
                f"import numpy as np; _r = np.random.default_rng(43).uniform(-1, 1, size={shape_str}).astype(np.{dtype}); x = np.fft.{rfft_func}(_r)",
                f"import numpy as np; _r = np.random.default_rng(99).standard_normal({shape_str}).astype(np.{dtype}) * 100; x = np.fft.{rfft_func}(_r)",
            ]
        elif needs_complex:
            setups = [
                (
                    f"import numpy as np; x = np.random.default_rng(42).standard_normal({shape_str}).astype(np.{dtype}) "
                    f"+ 1j * np.random.default_rng(43).standard_normal({shape_str}).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; t = np.linspace(0, 2*np.pi, {effective_n}).reshape({shape_str}).astype(np.{dtype}); "
                    f"x = np.sin(t) + 1j * np.cos(t)"
                ),
                (
                    f"import numpy as np; x = (np.random.default_rng(99).uniform(-1, 1, size={shape_str}).astype(np.{dtype}) "
                    f"+ 1j * np.random.default_rng(100).uniform(-1, 1, size={shape_str}).astype(np.{dtype}))"
                ),
            ]
        else:
            # Real input: fft, rfft, fftn, rfftn, ihfft
            setups = [
                f"import numpy as np; x = np.random.default_rng(42).standard_normal({shape_str}).astype(np.{dtype})",
                (
                    f"import numpy as np; t = np.linspace(0, 2*np.pi, {effective_n}).reshape({shape_str}).astype(np.{dtype}); "
                    f"x = np.sin(5*t) + 0.5*np.sin(13*t)"
                ),
                f"import numpy as np; x = np.random.default_rng(99).uniform(-1, 1, size={shape_str}).astype(np.{dtype})",
            ]

        bench = f"np.{op}(x)"
        analytical = _analytical_cost(op, effective_n)

        for setup in setups:
            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            measured = result.total_flops / repeats
            dist_values.append(measured / analytical if analytical else 0.0)
            dist_raw_totals.append(result.total_flops)

        if dist_values:
            results[op] = statistics.median(dist_values)
            details[op] = {
                "category": "counted_custom",
                "measurement_mode": "custom",
                "analytical_formula": _FORMULA_STRINGS.get(op, ""),
                "analytical_flops": analytical,
                "benchmark_size": benchmark_size,
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
