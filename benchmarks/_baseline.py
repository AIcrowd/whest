"""Baseline measurements for overhead-subtracted weight normalization.

Measures three baselines:

1. **alpha(add)** — raw FP instructions per element for ``np.add`` (includes
   ufunc overhead). Used to derive the binary ufunc overhead.
2. **alpha(abs)** — raw FP instructions per element for ``np.abs``. Since abs
   on float64 is a bitwise sign-bit clear (NOT an FP instruction), all
   measured FP instructions are pure **unary ufunc overhead**.
3. **Binary ufunc overhead** = ``alpha(add) - 1.0`` (since one add = exactly
   one FP instruction; the rest is overhead).

The runner subtracts the appropriate overhead from each operation's raw
alpha before storing it as the weight::

    weight(op) = max(alpha_raw(op) - overhead_for_category, 1.0)

This replaces the old ``weight(op) = alpha(op) / alpha(add)`` formula which
penalized BLAS ops (that bypass the ufunc layer) with ufunc overhead they
don't have.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from benchmarks._perf import measure_flops


@dataclass(frozen=True)
class BaselineResult:
    """All baseline measurements needed for overhead-subtracted normalization."""

    alpha_add: float
    """Raw alpha for np.add (FP instructions per element, including overhead)."""

    alpha_abs: float
    """Raw alpha for np.abs (pure unary ufunc overhead — abs is bitwise)."""

    @property
    def overhead_ufunc_unary(self) -> float:
        """Unary ufunc overhead per element (from abs measurement)."""
        return self.alpha_abs

    @property
    def overhead_ufunc_binary(self) -> float:
        """Binary ufunc overhead per element.

        Derived as alpha(add) - 1.0, since one add = exactly 1 FP instruction.
        """
        return max(self.alpha_add - 1.0, 0.0)

    @property
    def overhead_ufunc_reduction(self) -> float:
        """Reduction ufunc overhead (same iterator structure as unary)."""
        return self.alpha_abs

    def overhead_for_mode(self, mode: str) -> float:
        """Return the overhead to subtract for a given measurement mode."""
        return {
            "ufunc_unary": self.overhead_ufunc_unary,
            "ufunc_binary": self.overhead_ufunc_binary,
            "ufunc_reduction": self.overhead_ufunc_reduction,
            "blas": 0.0,
            "linalg": 0.0,
            "custom": 0.0,
            "instructions": 0.0,
        }.get(mode, 0.0)

    def to_dict(self) -> dict:
        """Serialize for weights.json metadata."""
        return {
            "alpha_add_raw": self.alpha_add,
            "alpha_abs_raw": self.alpha_abs,
            "overhead_ufunc_unary": self.overhead_ufunc_unary,
            "overhead_ufunc_binary": self.overhead_ufunc_binary,
            "overhead_ufunc_reduction": self.overhead_ufunc_reduction,
            "normalization": "subtract per-category ufunc overhead, clamp to min 1.0",
        }


def _measure_alpha(setups: list[str], bench: str, n: int, repeats: int) -> float:
    """Measure median alpha across distributions."""
    dist_alphas = []
    for setup in setups:
        result = measure_flops(setup, bench, repeats=repeats)
        dist_alphas.append(result.total_flops / (n * repeats))
    return statistics.median(dist_alphas)


def _unary_setups(n: int, dtype: str) -> list[str]:
    return [
        f"x = np.random.default_rng(42).standard_normal({n}).astype(np.{dtype})",
        (
            f"rng = np.random.default_rng(42); "
            f"x = rng.uniform(0.01, 100, size={n}).astype(np.{dtype})"
        ),
        (
            f"rng = np.random.default_rng(42); "
            f"x = rng.uniform(-1000, 1000, size={n}).astype(np.{dtype})"
        ),
    ]


def _binary_setups(n: int, dtype: str) -> list[str]:
    return [
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


def measure_baseline(
    n: int = 10_000_000, dtype: str = "float64", repeats: int = 10
) -> float:
    """Return alpha(add) for backwards compatibility.

    Prefer :func:`measure_baselines` which returns the full
    :class:`BaselineResult` with overhead measurements.
    """
    return _measure_alpha(
        _binary_setups(n, dtype), "np.add(x, y, out=_out)", n, repeats
    )


def measure_baselines(
    n: int = 10_000_000, dtype: str = "float64", repeats: int = 10
) -> BaselineResult:
    """Measure all baselines needed for overhead-subtracted normalization.

    Returns
    -------
    BaselineResult
        Contains alpha(add), alpha(abs), and derived overhead values.
    """
    alpha_add = _measure_alpha(
        _binary_setups(n, dtype), "np.add(x, y, out=_out)", n, repeats
    )
    alpha_abs = _measure_alpha(_unary_setups(n, dtype), "np.abs(x)", n, repeats)

    result = BaselineResult(alpha_add=alpha_add, alpha_abs=alpha_abs)
    print(f"  alpha(add) = {alpha_add:.4f}")
    print(f"  alpha(abs) = {alpha_abs:.4f} (pure unary ufunc overhead)")
    print("  Derived overheads:")
    print(f"    ufunc_unary:     {result.overhead_ufunc_unary:.4f}")
    print(f"    ufunc_binary:    {result.overhead_ufunc_binary:.4f}")
    print(f"    ufunc_reduction: {result.overhead_ufunc_reduction:.4f}")
    print("    blas/linalg:     0.0000")
    return result
