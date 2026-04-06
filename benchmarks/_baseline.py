"""Baseline measurement: FP ops per element for np.add with pre-allocated output.

Uses the same methodology as pointwise benchmarks (out= to eliminate allocation).
"""

from __future__ import annotations

from benchmarks._perf import measure_flops


def measure_baseline(
    n: int = 10_000_000, dtype: str = "float64", repeats: int = 10
) -> float:
    """Measure raw cost per element for np.add (the 1.0 reference).

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
        Raw measurement per element for np.add.
    """
    setup = (
        f"x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}); "
        f"y = np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}); "
        f"_out = np.empty({n}, dtype=np.{dtype})"
    )
    bench = "np.add(x, y, out=_out)"
    result = measure_flops(setup, bench, repeats=repeats)
    return result.total_flops / (n * repeats)
