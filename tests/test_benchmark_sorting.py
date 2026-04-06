"""Tests for sorting benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._sorting import SORTING_OPS, benchmark_sorting


class TestOpsLists:
    def test_sorting_ops_non_empty(self):
        assert len(SORTING_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("sort", "argsort", "lexsort", "searchsorted", "unique"):
            assert op in SORTING_OPS, f"{op} missing from SORTING_OPS"


class TestBenchmarkSorting:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._sorting.measure_flops", return_value=mock_result
        ):
            result = benchmark_sorting(n=1_000, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(SORTING_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._sorting.measure_flops", return_value=mock_result
        ):
            result = benchmark_sorting(n=1_000, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_flops_per_element_calculation(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=1_000,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._sorting.measure_flops", return_value=mock_result
        ):
            result = benchmark_sorting(n=1_000, dtype="float64", repeats=2)

        # total_flops = 1000 * 4 = 4000, per_element = 4000 / (1000 * 2) = 2.0
        for val in result.values():
            assert val == pytest.approx(2.0)
