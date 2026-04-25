"""Verify benchmark_size descriptions match bench_code variable names.

Every array variable used in bench_code (e.g., a, b, x, A, B) must appear
in benchmark_size with a shape descriptor.  Scalar parameters and output
buffers (_out) are exempt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "src" / "flopscope" / "data" / "weights.json"

# Tokens that appear as bench_code arguments but are NOT array variables
# whose shapes need documenting in benchmark_size.
_EXEMPT_TOKENS = frozenset(
    {
        "_out",  # pre-allocated output buffer
        "out",  # same
        "None",  # keyword default
        "rcond",  # keyword arg
        "mode",  # keyword arg
        "0.5",  # scalar constant (heaviside h)
    }
)


def _extract_bench_vars(bench_code: str) -> list[str]:
    """Extract array variable names from a benchmark code string.

    Parses ``np.func(a, b, out=_out)`` and returns ``['a', 'b']``.
    Skips numeric literals, keyword arguments, and exempt tokens.
    Strips leading underscores from variable names (``_pool`` -> ``pool``)
    so they match benchmark_size which uses clean names.
    """
    # Match the argument list inside np.something(...)
    m = re.search(r"np\.[\w.]+\((.+)\)", bench_code)
    if not m:
        return []
    args_str = m.group(1)

    # Flatten nested parens: ((x, y)) -> x, y
    # Remove all parens and brackets, then split on commas
    flat = args_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")

    variables = []
    for arg in flat.split(","):
        arg = arg.strip()
        # Skip keyword arguments (key=value)
        if "=" in arg:
            continue
        # Skip numeric literals
        if re.match(r"^-?[\d.]+$", arg):
            continue
        # Skip string literals
        if arg.startswith(("'", '"')):
            continue
        # Skip exempt tokens
        if arg in _EXEMPT_TOKENS:
            continue
        # Must be a variable name
        if re.match(r"^[a-zA-Z_]\w*$", arg):
            # Strip leading underscores to match benchmark_size convention
            # _pool -> pool, _mean -> mean, _cov -> cov
            clean = (
                arg.lstrip("_")
                if arg.startswith("_") and arg not in _EXEMPT_TOKENS
                else arg
            )
            if clean:
                variables.append(clean)

    return variables


def _extract_size_vars(benchmark_size: str) -> list[str]:
    """Extract variable names documented in benchmark_size.

    Parses ``a: (10000000,), b: (10000000,)`` and returns ``['a', 'b']``.
    Also handles ``output: (10000000,)`` and ``n=10000000``.
    """
    # Match "varname:" pattern
    vars_with_colon = re.findall(r"([a-zA-Z_]\w*)\s*:", benchmark_size)
    # Match "varname=" pattern (e.g., kth=5000000, degree=100)
    vars_with_eq = re.findall(r"([a-zA-Z_]\w*)\s*=", benchmark_size)
    return vars_with_colon + vars_with_eq


@pytest.fixture(scope="module")
def per_op_details() -> dict:
    assert WEIGHTS_PATH.exists(), f"weights.json not found: {WEIGHTS_PATH}"
    data = json.loads(WEIGHTS_PATH.read_text())
    return data["meta"]["per_op_details"]


class TestBenchmarkSizeConsistency:
    """Every array variable in bench_code must appear in benchmark_size."""

    def test_all_bench_vars_documented_in_size(self, per_op_details: dict):
        """Each array variable from bench_code must have a shape in benchmark_size."""
        missing = []
        for op, d in sorted(per_op_details.items()):
            bench = d.get("bench_code", "")
            size = d.get("benchmark_size", "")
            if not bench or not size:
                continue

            bench_vars = _extract_bench_vars(bench)
            size_vars = _extract_size_vars(size)
            size_var_set = set(size_vars)

            for var in bench_vars:
                if var not in size_var_set:
                    missing.append(
                        f"{op}: variable '{var}' in bench_code "
                        f"'{bench}' not documented in "
                        f"benchmark_size '{size}'"
                    )

        assert not missing, (
            f"{len(missing)} bench_code variables missing from benchmark_size:\n"
            + "\n".join(f"  {m}" for m in missing[:20])
        )

    def test_benchmark_size_is_nonempty(self, per_op_details: dict):
        """Every benchmarked op should have a non-empty benchmark_size."""
        empty = [
            op
            for op, d in per_op_details.items()
            if d.get("bench_code", "").strip()
            and not d.get("benchmark_size", "").strip()
        ]
        assert not empty, f"Ops with empty benchmark_size: {empty}"

    def test_benchmark_size_uses_explicit_shapes(self, per_op_details: dict):
        """benchmark_size should use explicit shape format, not bare n=..."""
        bare_n = []
        for op, d in sorted(per_op_details.items()):
            size = d.get("benchmark_size", "")
            # "n=10000000" without any ":" is the old ambiguous format
            if size and ":" not in size and size.startswith("n="):
                bench = d.get("bench_code", "")
                # Exception: permutation takes a scalar n, not an array
                if "permutation" in bench:
                    continue
                bare_n.append(f"{op}: '{size}' — should use 'x: (n,)' format")
        assert not bare_n, (
            f"{len(bare_n)} ops still use bare 'n=...' format:\n"
            + "\n".join(f"  {m}" for m in bare_n[:20])
        )


class TestExtractFunctions:
    """Unit tests for the parsing helpers."""

    def test_extract_bench_vars_binary(self):
        assert _extract_bench_vars("np.add(a, b, out=_out)") == ["a", "b"]

    def test_extract_bench_vars_unary(self):
        assert _extract_bench_vars("np.sin(x, out=_out)") == ["x"]

    def test_extract_bench_vars_with_literal(self):
        assert _extract_bench_vars("np.partition(x, 5000000)") == ["x"]

    def test_extract_bench_vars_lexsort_tuple(self):
        # Nested parens are flattened: ((x, y)) -> x, y
        assert _extract_bench_vars("np.lexsort((x, y))") == ["x", "y"]

    def test_extract_bench_vars_strips_underscore_prefix(self):
        assert _extract_bench_vars("np.random.choice(_pool, 10000000)") == ["pool"]

    def test_extract_bench_vars_multiple_underscore_prefix(self):
        result = _extract_bench_vars(
            "np.random.multivariate_normal(_mean, _cov, 10000000)"
        )
        assert result == ["mean", "cov"]

    def test_extract_bench_vars_window(self):
        # np.bartlett(10000000) — no variable names, just a literal
        assert _extract_bench_vars("np.bartlett(10000000)") == []

    def test_extract_bench_vars_matmul(self):
        assert _extract_bench_vars("np.matmul(A, B)") == ["A", "B"]

    def test_extract_size_vars_binary(self):
        assert _extract_size_vars("a: (10000000,), b: (10000000,)") == ["a", "b"]

    def test_extract_size_vars_with_kwarg(self):
        result = _extract_size_vars("x: (10000000,), kth=5000000")
        assert "x" in result
        assert "kth" in result

    def test_extract_size_vars_output(self):
        assert _extract_size_vars("output: (10000000,)") == ["output"]
