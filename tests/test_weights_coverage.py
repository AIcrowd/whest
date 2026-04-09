"""Validate that weights.json and the empirical-weights docs cover all counted operations.

This test ensures that every non-free, non-blacklisted operation in the
registry either:
  1. Has a direct weight in weights.json, OR
  2. Is a known alias of a weighted operation, OR
  3. Falls into a documented exclusion category (bitwise, complex, etc.)

It also validates that the docs/reference/empirical-weights.md mentions every
operation that has a weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechestim._registry import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "src" / "mechestim" / "data" / "weights.json"
DOCS_PATH = ROOT / "docs" / "reference" / "empirical-weights.md"

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
}

# ---------------------------------------------------------------------------
# Exclusion categories: operations that don't produce meaningful float64
# perf weights because they operate on integers, complex types, or have
# shape-dependent cost that can't be captured by a single scalar weight.
# ---------------------------------------------------------------------------

#: Bitwise / integer-only ops — no fp_arith_inst_retired events.
EXCLUDED_BITWISE: frozenset[str] = frozenset({
    "bitwise_and", "bitwise_count", "bitwise_invert", "bitwise_left_shift",
    "bitwise_not", "bitwise_or", "bitwise_right_shift", "bitwise_xor",
    "invert", "left_shift", "right_shift", "gcd", "lcm",
})

#: Complex-number ops — benchmarked with float64 but internally branch
#: on dtype; a single real-dtype weight isn't representative.
EXCLUDED_COMPLEX: frozenset[str] = frozenset({
    "angle", "conj", "conjugate", "imag", "real", "real_if_close",
    "iscomplex", "iscomplexobj", "isreal", "isrealobj", "sort_complex",
})

#: BLAS contraction ops — cost is shape-dependent (M×N×K), a single
#: scalar weight doesn't apply. These need a correction-factor approach.
EXCLUDED_CONTRACTION: frozenset[str] = frozenset({
    "dot", "einsum", "einsum_path", "inner", "kron", "matmul",
    "outer", "tensordot", "vdot", "vecdot",
})

#: linalg ops not in the decomposition benchmark suite — delegates to
#: contraction ops or uses decompositions internally.
EXCLUDED_LINALG: frozenset[str] = frozenset({
    "linalg.cond", "linalg.cross", "linalg.matmul", "linalg.matrix_norm",
    "linalg.matrix_power", "linalg.matrix_rank", "linalg.multi_dot",
    "linalg.norm", "linalg.outer", "linalg.tensordot", "linalg.tensorinv",
    "linalg.tensorsolve", "linalg.trace", "linalg.vecdot",
    "linalg.vector_norm",
})

#: Random distributions not yet in the benchmark suite — only 10
#: representative distributions are benchmarked; these share the same
#: underlying RNG + transform pattern.
EXCLUDED_RANDOM: frozenset[str] = frozenset({
    name for name in REGISTRY
    if name.startswith("random.") and name not in {
        "random.binomial", "random.permutation", "random.poisson",
        "random.shuffle", "random.standard_cauchy",
        "random.standard_exponential", "random.standard_gamma",
        "random.standard_normal", "random.standard_t", "random.uniform",
    }
    and REGISTRY[name]["category"] != "free"
})

#: Window functions — cost is trivially n (one trig eval per sample);
#: not benchmarked separately as they reduce to sin/cos weights.
EXCLUDED_WINDOW: frozenset[str] = frozenset({
    "bartlett", "blackman", "hamming", "hanning", "kaiser",
})

#: Set operations — sort-dominated, weight inheritable from sort.
EXCLUDED_SET: frozenset[str] = frozenset({
    "in1d", "intersect1d", "isin", "setdiff1d", "setxor1d", "union1d",
    "unique_all", "unique_counts", "unique_inverse", "unique_values",
})

#: Misc element-wise ops that are either trivially cheap (comparison/test
#: ops like isclose, nan_to_num) or have custom formulas where the
#: per-element weight is ~1.0 (diff, clip, allclose, etc.).
#: These should be benchmarked in a future calibration pass.
EXCLUDED_PENDING_BENCHMARK: frozenset[str] = frozenset({
    # Simple element-wise tests / conversions (weight ≈ comparison ops)
    "allclose", "array_equal", "array_equiv", "clip", "heaviside",
    "isclose", "isnat", "isneginf", "isposinf", "nan_to_num",
    "frexp", "modf", "spacing", "sinc", "i0",
    # Differencing / gradient (element-wise subtraction)
    "diff", "ediff1d", "gradient",
    # Generation ops (cost = num elements)
    "geomspace", "logspace", "vander",
    # Aggregation / statistical (delegate to weighted ops internally)
    "bincount", "corrcoef", "correlate", "convolve", "cov", "cross",
    "digitize", "histogram", "histogram2d", "histogram_bin_edges",
    "histogramdd", "interp", "trace", "trapezoid", "trapz", "unwrap",
    # Pure planning op (no FP work)
    "einsum_path",
    # Unary that are essentially identity / real extraction
    "real_if_close", "sort_complex",
})

ALL_EXCLUDED = (
    EXCLUDED_BITWISE | EXCLUDED_COMPLEX | EXCLUDED_CONTRACTION
    | EXCLUDED_LINALG | EXCLUDED_RANDOM | EXCLUDED_WINDOW
    | EXCLUDED_SET | EXCLUDED_PENDING_BENCHMARK
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
        """Every counted op must be weighted, aliased, or in an exclusion list."""
        covered = set(weights)
        aliased = {name for name in ALIAS_MAP if ALIAS_MAP[name] in weights}
        accounted_for = covered | aliased | ALL_EXCLUDED

        uncovered = sorted(counted_ops - accounted_for)
        assert not uncovered, (
            f"{len(uncovered)} counted ops are not weighted, aliased, "
            f"or excluded:\n"
            + "\n".join(
                f"  {name:30s} ({REGISTRY[name]['category']})"
                for name in uncovered[:20]
            )
            + ("\n  ..." if len(uncovered) > 20 else "")
        )

    def test_alias_targets_are_weighted(self, weights: dict[str, float]):
        """Every alias target in ALIAS_MAP should have a weight."""
        missing_targets = {
            f"{alias} -> {target}"
            for alias, target in ALIAS_MAP.items()
            if target not in weights
        }
        assert not missing_targets, (
            f"Alias targets missing from weights: {sorted(missing_targets)}"
        )

    def test_excluded_ops_are_actually_in_registry(self, counted_ops: set[str]):
        """Excluded ops should actually be in the registry (catch stale exclusions)."""
        stale = sorted(ALL_EXCLUDED - counted_ops)
        assert not stale, (
            f"Excluded ops not in registry (stale?): {stale}"
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
