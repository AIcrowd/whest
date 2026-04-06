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
) -> dict[str, float]:
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
    dict[str, float]
        Mapping from op name to median correction factor.
    """
    results: dict[str, float] = {}

    # 2D/nD ops use sqrt(n) x sqrt(n)
    side = int(math.isqrt(n))

    for op in FFT_OPS:
        dist_values: list[float] = []

        # Determine dimensionality
        is_2d = "2" in op.split(".")[-1]
        is_nd = op.endswith("n") or op.endswith("fftn")
        is_multi = is_2d or is_nd

        if is_multi:
            shape_str = f"({side}, {side})"
            effective_n = side * side
        else:
            shape_str = f"({n},)"
            effective_n = n

        # Need complex input for ifft/ihfft/hfft variants
        short = op.split(".")[-1]
        needs_complex = short.startswith("i") or short == "hfft"

        setups = []
        if needs_complex:
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
            result = measure_flops(setup, bench, repeats=repeats)
            measured = result.total_flops / repeats
            dist_values.append(measured / analytical if analytical else 0.0)

        results[op] = statistics.median(dist_values)

    return results
