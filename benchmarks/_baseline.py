"""Baseline measurement: FP ops per element for np.add."""

from __future__ import annotations

from benchmarks._perf import measure_flops


def measure_baseline(n: int = 10_000_000, dtype: str = "float64", repeats: int = 10) -> float:
    """Measure FP operations per element for np.add (the 1.0 reference).

    Parameters
    ----------
    n : int
        Array size.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions.

    Returns
    -------
    float
        FP operations per element for np.add.
    """
    setup = f"x = np.random.randn({n}).astype(np.{dtype}); y = np.random.randn({n}).astype(np.{dtype})"
    bench = "np.add(x, y, out=x)"
    result = measure_flops(setup, bench, repeats=repeats)
    return result.total_flops / (n * repeats)
