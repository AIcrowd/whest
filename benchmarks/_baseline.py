"""Baseline measurement: alpha(add) — the correction factor for np.add.

Uses the same methodology as pointwise benchmarks (``out=`` to eliminate
allocation overhead).  The returned value is::

    alpha(add) = total_measured / (numel * repeats)

Since the analytical FLOP cost of element-wise addition is ``numel``,
dividing the total measurement by ``numel * repeats`` gives the per-FLOP
correction factor directly.  This value is stored in the output as
``meta.methodology.baseline_alpha`` and used to normalise all other
operations:  ``weight(op) = alpha(op) / alpha(add)``.
"""

from __future__ import annotations

from benchmarks._perf import measure_flops


def measure_baseline(
    n: int = 10_000_000, dtype: str = "float64", repeats: int = 10
) -> float:
    """Return alpha(add): the correction factor for ``np.add``.

    This is the ratio of measured hardware cost (FP instructions in perf
    mode, or elapsed nanoseconds in timing mode) to the analytical FLOP
    count (``n``) per repetition::

        alpha(add) = total_measured / (n * repeats)

    The runner uses this as the normalisation constant so that
    ``weight(add) = 1.0``.

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
        alpha(add) — raw measurement per analytical FLOP for np.add.
    """
    import statistics

    # Use the SAME 3 distributions as the pointwise benchmark so the
    # per-element overhead from numpy's ufunc dispatch cancels out when
    # computing weight(op) = alpha(op) / alpha(add).
    setups = [
        (
            f"x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype}); "
            f"y = np.random.default_rng(43).standard_normal({n}).astype(np.{dtype}); "
            f"_out = np.empty({n}, dtype=np.{dtype})"
        ),
        (
            f"rng = np.random.default_rng(42); "
            f"x = rng.uniform(0.01, 100, size={n}).astype(np.{dtype}); "
            f"y = rng.uniform(0.01, 100, size={n}).astype(np.{dtype}); "
            f"_out = np.empty({n}, dtype=np.{dtype})"
        ),
        (
            f"rng = np.random.default_rng(42); "
            f"x = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype}); "
            f"y = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype}); "
            f"_out = np.empty({n}, dtype=np.{dtype})"
        ),
    ]
    bench = "np.add(x, y, out=_out)"
    dist_alphas = []
    for setup in setups:
        result = measure_flops(setup, bench, repeats=repeats)
        dist_alphas.append(result.total_flops / (n * repeats))
    return statistics.median(dist_alphas)
