"""Tests for sorting benchmark module."""

import math
from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._sorting import (
    SORTING_OPS,
    _analytical_cost,
    benchmark_sorting,
)


class TestOpsLists:
    def test_sorting_ops_non_empty(self):
        assert len(SORTING_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("sort", "argsort", "lexsort", "searchsorted", "unique"):
            assert op in SORTING_OPS, f"{op} missing from SORTING_OPS"

    def test_contains_set_ops(self):
        for op in ("in1d", "isin", "intersect1d", "setdiff1d", "setxor1d", "union1d"):
            assert op in SORTING_OPS, f"{op} missing from SORTING_OPS"

    def test_contains_unique_variants(self):
        for op in ("unique_all", "unique_counts", "unique_inverse", "unique_values"):
            assert op in SORTING_OPS, f"{op} missing from SORTING_OPS"


class TestAnalyticalCost:
    def test_sort_nlogn(self):
        n = 1024
        expected = n * math.ceil(math.log2(n))
        assert _analytical_cost("sort", n) == expected

    def test_argsort_nlogn(self):
        n = 1024
        expected = n * math.ceil(math.log2(n))
        assert _analytical_cost("argsort", n) == expected

    def test_unique_nlogn(self):
        n = 1024
        expected = n * math.ceil(math.log2(n))
        assert _analytical_cost("unique", n) == expected

    def test_unique_variants_nlogn(self):
        n = 1024
        expected = n * math.ceil(math.log2(n))
        for op in ("unique_all", "unique_counts", "unique_inverse", "unique_values"):
            assert _analytical_cost(op, n) == expected, f"{op} cost mismatch"

    def test_lexsort_k_nlogn(self):
        n = 1024
        k = 2
        expected = k * n * math.ceil(math.log2(n))
        assert _analytical_cost("lexsort", n, k=k) == expected

    def test_lexsort_default_k(self):
        n = 1024
        # Default k=2
        expected = 2 * n * math.ceil(math.log2(n))
        assert _analytical_cost("lexsort", n) == expected

    def test_searchsorted_mlogn(self):
        n = 1024
        m = 512
        expected = m * math.ceil(math.log2(n))
        assert _analytical_cost("searchsorted", n, m=m) == expected

    def test_searchsorted_default_m(self):
        n = 1024
        # Default m=n
        expected = n * math.ceil(math.log2(n))
        assert _analytical_cost("searchsorted", n) == expected

    def test_partition_linear(self):
        n = 1024
        assert _analytical_cost("partition", n) == n

    def test_argpartition_linear(self):
        n = 1024
        assert _analytical_cost("argpartition", n) == n

    def test_set_ops_nm_logn_m(self):
        n = 1024
        m = 512
        total = n + m
        expected = total * math.ceil(math.log2(total))
        for op in ("in1d", "isin", "intersect1d", "setdiff1d", "setxor1d", "union1d"):
            assert _analytical_cost(op, n, m=m) == expected, f"{op} cost mismatch"

    def test_set_ops_default_m(self):
        n = 1024
        total = n + n
        expected = total * math.ceil(math.log2(total))
        assert _analytical_cost("in1d", n) == expected

    def test_small_n_edge_case(self):
        # n=1 should not cause log2(0) error
        assert _analytical_cost("sort", 1) == 1
        assert _analytical_cost("searchsorted", 1) == 1
        assert _analytical_cost("in1d", 1, m=1) == 2 * math.ceil(math.log2(2))


class TestBenchmarkSorting:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._sorting.measure_flops", return_value=mock_result):
            result = benchmark_sorting(n=1_000, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        # All original ops must be present; NumPy 2.x unique variants may
        # fail with RuntimeError if unavailable, so we check the core set.
        core_ops = {
            "sort",
            "argsort",
            "lexsort",
            "partition",
            "argpartition",
            "searchsorted",
            "unique",
            "in1d",
            "isin",
            "intersect1d",
            "setdiff1d",
            "setxor1d",
            "union1d",
            "unique_all",
            "unique_counts",
            "unique_inverse",
            "unique_values",
        }
        assert core_ops.issubset(set(result.keys()))

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._sorting.measure_flops", return_value=mock_result):
            result = benchmark_sorting(n=1_000, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_sort_uses_analytical_denominator(self):
        """Verify sort normalizes by n*ceil(log2(n)), not by n."""
        n = 1_000
        repeats = 2
        total_flops = 10_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._sorting.measure_flops", return_value=mock_result):
            result = benchmark_sorting(n=n, dtype="float64", repeats=repeats)

        analytical = _analytical_cost("sort", n)
        expected = total_flops / (analytical * repeats)
        assert result["sort"] == pytest.approx(expected)

    def test_partition_uses_linear_denominator(self):
        """Verify partition normalizes by n (linear analytical cost)."""
        n = 1_000
        repeats = 2
        total_flops = 10_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._sorting.measure_flops", return_value=mock_result):
            result = benchmark_sorting(n=n, dtype="float64", repeats=repeats)

        expected = total_flops / (n * repeats)
        assert result["partition"] == pytest.approx(expected)

    def test_lexsort_uses_k_nlogn_denominator(self):
        """Verify lexsort normalizes by k*n*ceil(log2(n))."""
        n = 1_000
        repeats = 1
        total_flops = 20_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._sorting.measure_flops", return_value=mock_result):
            result = benchmark_sorting(n=n, dtype="float64", repeats=repeats)

        analytical = _analytical_cost("lexsort", n, k=2)
        expected = total_flops / (analytical * repeats)
        assert result["lexsort"] == pytest.approx(expected)

    def test_set_op_uses_nm_logn_denominator(self):
        """Verify set ops normalize by (n+m)*ceil(log2(n+m))."""
        n = 1_000
        repeats = 1
        total_flops = 50_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._sorting.measure_flops", return_value=mock_result):
            result = benchmark_sorting(n=n, dtype="float64", repeats=repeats)

        # Set ops use m=n in the benchmark
        analytical = _analytical_cost("intersect1d", n, m=n)
        expected = total_flops / (analytical * repeats)
        assert result["intersect1d"] == pytest.approx(expected)
