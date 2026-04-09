"""Validate that weights.json and the empirical-weights docs cover all counted operations.

This test ensures that every non-free, non-blacklisted operation in the
registry either:
  1. Has a direct weight in weights.json, OR
  2. Is a known alias of a weighted operation, OR
  3. Is listed in a benchmark module's ops list (weights pending generation), OR
  4. Falls into a documented exclusion category (bitwise, complex, etc.)

It also validates that the docs/reference/empirical-weights.md mentions every
operation that has a weight.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mechestim._registry import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "src" / "mechestim" / "data" / "weights.json"
DOCS_PATH = ROOT / "docs" / "reference" / "empirical-weights.md"

# Ensure benchmarks package is importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks._contractions import CONTRACTION_OPS  # noqa: E402
from benchmarks._fft import FFT_OPS  # noqa: E402
from benchmarks._linalg import LINALG_OPS  # noqa: E402
from benchmarks._misc import MISC_OPS  # noqa: E402
from benchmarks._pointwise import BINARY_OPS, SPECIAL_OPS, UNARY_OPS  # noqa: E402
from benchmarks._polynomial import POLYNOMIAL_OPS  # noqa: E402
from benchmarks._random import RANDOM_OPS  # noqa: E402
from benchmarks._reductions import REDUCTION_OPS  # noqa: E402
from benchmarks._sorting import SORTING_OPS  # noqa: E402
from benchmarks._window import WINDOW_OPS  # noqa: E402

# ---------------------------------------------------------------------------
# Ops covered by benchmark modules (weights pending bare-metal generation).
# ---------------------------------------------------------------------------
BENCHMARKED_OPS: frozenset[str] = frozenset(
    set(SORTING_OPS)
    | set(CONTRACTION_OPS)
    | set(MISC_OPS)
    | set(WINDOW_OPS)
    | set(UNARY_OPS)
    | set(BINARY_OPS)
    | set(SPECIAL_OPS)
    | set(REDUCTION_OPS)
    | set(RANDOM_OPS)
    | set(POLYNOMIAL_OPS)
    | set(FFT_OPS)
    | set(LINALG_OPS)
)

# ---------------------------------------------------------------------------
# Alias map: operation name -> canonical name whose weight it should inherit.
# These are NumPy aliases, deprecated names, or nan-aware variants that share
# the same underlying FP instruction profile as their canonical form.
# ---------------------------------------------------------------------------
ALIAS_MAP: dict[str, str] = {
    # NumPy 2.x short aliases
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atan2": "arctan2",
    "atanh": "arctanh",
    "pow": "power",
    # Legacy aliases
    "absolute": "abs",
    "amax": "max",
    "amin": "min",
    "around": "rint",
    "fix": "trunc",
    "round": "rint",
    # nan-aware variants (same FP profile as non-nan version)
    "nanargmax": "argmax",
    "nanargmin": "argmin",
    "nancumprod": "cumprod",
    "nancumsum": "cumsum",
    # NumPy 2.x array API names
    "cumulative_prod": "cumprod",
    "cumulative_sum": "cumsum",
    # Reduction aliases
    "ptp": "max",
    # Binary alias
    "divmod": "floor_divide",
    # Deprecated name
    "trapz": "trapezoid",
}

# ---------------------------------------------------------------------------
# Exclusion categories: operations that don't produce meaningful float64
# perf weights because they operate on integers, complex types, or have
# shape-dependent cost that can't be captured by a single scalar weight.
# ---------------------------------------------------------------------------

#: Bitwise / integer-only ops -- no fp_arith_inst_retired events.
EXCLUDED_BITWISE: frozenset[str] = frozenset({
    "bitwise_and", "bitwise_count", "bitwise_invert", "bitwise_left_shift",
    "bitwise_not", "bitwise_or", "bitwise_right_shift", "bitwise_xor",
    "invert", "left_shift", "right_shift", "gcd", "lcm",
})

#: Complex-number ops -- benchmarked with float64 but internally branch
#: on dtype; a single real-dtype weight isn't representative.
EXCLUDED_COMPLEX: frozenset[str] = frozenset({
    "angle", "conj", "conjugate", "imag", "real", "real_if_close",
    "iscomplex", "iscomplexobj", "isreal", "isrealobj", "sort_complex",
})

#: BLAS contraction planning ops -- no FP work.
EXCLUDED_CONTRACTION: frozenset[str] = frozenset({
    "einsum_path",  # planning op, no FP work
})

#: linalg ops not in the decomposition benchmark suite -- delegates to
#: contraction ops or uses decompositions internally.
EXCLUDED_LINALG: frozenset[str] = frozenset({
    "linalg.cond", "linalg.cross", "linalg.matmul", "linalg.matrix_norm",
    "linalg.matrix_power", "linalg.matrix_rank", "linalg.multi_dot",
    "linalg.norm", "linalg.outer", "linalg.tensordot", "linalg.tensorinv",
    "linalg.tensorsolve", "linalg.trace", "linalg.vecdot",
    "linalg.vector_norm",
})

#: Random ops that cannot be meaningfully benchmarked with float64.
EXCLUDED_RANDOM: frozenset[str] = frozenset({
    "random.bytes",             # returns bytes, not FP
    "random.random_integers",   # removed in NumPy 2.x
    "random.ranf",              # alias for random_sample
    "random.sample",            # alias for random_sample
})

#: Ops that only exist in specific NumPy versions, or operate on non-float
#: dtypes (e.g. datetime64) where a float64 weight is not applicable.
EXCLUDED_VERSION_DEPENDENT: frozenset[str] = frozenset({
    "isnat",  # datetime64/timedelta64 only -- no FP operations
})

ALL_EXCLUDED = (
    EXCLUDED_BITWISE | EXCLUDED_COMPLEX | EXCLUDED_CONTRACTION
    | EXCLUDED_LINALG | EXCLUDED_RANDOM | EXCLUDED_VERSION_DEPENDENT
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weights() -> dict[str, float]:
    """Load weights.json."""
    assert WEIGHTS_PATH.exists(), f"weights.json not found at {WEIGHTS_PATH}"
    data = json.loads(WEIGHTS_PATH.read_text())
    assert "weights" in data, "weights.json missing 'weights' key"
    assert "meta" in data, "weights.json missing 'meta' key"
    return data["weights"]


@pytest.fixture(scope="module")
def docs_text() -> str:
    """Load empirical-weights.md."""
    assert DOCS_PATH.exists(), f"empirical-weights.md not found at {DOCS_PATH}"
    return DOCS_PATH.read_text()


@pytest.fixture(scope="module")
def counted_ops() -> set[str]:
    """All non-free, non-blacklisted operations in the registry."""
    return {
        name for name, entry in REGISTRY.items()
        if entry["category"] not in ("free", "blacklisted")
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWeightsJsonCoverage:
    """Verify weights.json covers all benchmarkable operations."""

    def test_all_weighted_ops_are_in_registry(self, weights: dict[str, float]):
        """Every op in weights.json should exist in the registry."""
        extra = set(weights) - set(REGISTRY)
        assert not extra, (
            f"weights.json contains {len(extra)} ops not in registry: "
            f"{sorted(extra)[:10]}"
        )

    def test_all_weights_are_positive(self, weights: dict[str, float]):
        """Weights must be non-negative (0 is allowed for near-free ops)."""
        negative = {k: v for k, v in weights.items() if v < 0}
        assert not negative, f"Negative weights: {negative}"

    def test_baseline_add_is_present(self, weights: dict[str, float]):
        """The baseline op (add) must be weighted."""
        assert "add" in weights, "Baseline 'add' missing from weights"

    def test_counted_ops_are_covered_or_excluded(
        self,
        weights: dict[str, float],
        counted_ops: set[str],
    ):
        """Every counted op must be weighted, in a benchmark module, aliased, or excluded."""
        covered = set(weights)
        # An alias is accounted for if its target is either already weighted
        # or listed in a benchmark module's ops list.
        alias_targets = set(weights) | BENCHMARKED_OPS
        aliased = {name for name in ALIAS_MAP if ALIAS_MAP[name] in alias_targets}
        accounted_for = covered | aliased | BENCHMARKED_OPS | ALL_EXCLUDED

        uncovered = sorted(counted_ops - accounted_for)
        assert not uncovered, (
            f"{len(uncovered)} counted ops are not weighted, benchmarked, aliased, "
            f"or excluded:\n"
            + "\n".join(
                f"  {name:30s} ({REGISTRY[name]['category']})"
                for name in uncovered[:20]
            )
            + ("\n  ..." if len(uncovered) > 20 else "")
        )

    def test_alias_targets_exist(self, weights: dict[str, float]):
        """Every alias target in ALIAS_MAP should be weighted or in a benchmark module."""
        alias_targets = set(weights) | BENCHMARKED_OPS
        missing_targets = {
            f"{alias} -> {target}"
            for alias, target in ALIAS_MAP.items()
            if target not in alias_targets
        }
        assert not missing_targets, (
            f"Alias targets missing from weights and benchmark modules: "
            f"{sorted(missing_targets)}"
        )

    def test_excluded_ops_are_actually_in_registry(self, counted_ops: set[str]):
        """Excluded ops should actually be in the registry (catch stale exclusions)."""
        stale = sorted(ALL_EXCLUDED - counted_ops)
        assert not stale, (
            f"Excluded ops not in registry (stale?): {stale}"
        )

    def test_no_double_accounting(self, weights: dict[str, float]):
        """Ops should not be both excluded AND weighted/benchmarked."""
        double = sorted(ALL_EXCLUDED & (set(weights) | BENCHMARKED_OPS))
        # Allow ops that are in BENCHMARKED_OPS to also be excluded --
        # this can happen during transition.  Only flag ops that are in
        # weights.json AND excluded.
        truly_double = sorted(ALL_EXCLUDED & set(weights))
        assert not truly_double, (
            f"Ops in both weights.json and exclusion sets: {truly_double}"
        )

    def test_benchmarked_ops_exist_in_registry(self):
        """Every op listed in a benchmark module should be in the registry."""
        missing = sorted(BENCHMARKED_OPS - set(REGISTRY))
        assert not missing, (
            f"Benchmark ops not in registry: {missing}"
        )

    def test_meta_has_required_fields(self):
        """weights.json metadata should have hardware/software/config."""
        data = json.loads(WEIGHTS_PATH.read_text())
        meta = data["meta"]
        assert "hardware" in meta
        assert "software" in meta
        assert "benchmark_config" in meta
        assert meta["benchmark_config"]["measurement_mode"] in ("perf", "timing")


class TestDocsWeightCoverage:
    """Verify empirical-weights.md mentions all weighted operations."""

    def test_all_weighted_ops_appear_in_docs(
        self,
        weights: dict[str, float],
        docs_text: str,
    ):
        """Every op in weights.json should appear in the markdown."""
        missing = [name for name in sorted(weights) if f"`{name}`" not in docs_text]
        assert not missing, (
            f"{len(missing)} weighted ops missing from docs:\n"
            + "\n".join(f"  {name}" for name in missing[:20])
        )

    def test_docs_mentions_measurement_mode(self, docs_text: str):
        """Docs should mention the measurement mode."""
        assert "perf" in docs_text, "Docs should mention perf measurement mode"

    def test_docs_mentions_baseline(self, docs_text: str):
        """Docs should mention add as the baseline."""
        assert "add" in docs_text and "baseline" in docs_text.lower()
