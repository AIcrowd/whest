"""Tests for reductions benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._reductions import REDUCTION_OPS, benchmark_reductions


class TestOpsLists:
    def test_reduction_ops_non_empty(self):
        assert len(REDUCTION_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("sum", "mean", "std", "argmax", "nansum", "median", "percentile", "count_nonzero"):
            assert op in REDUCTION_OPS, f"{op} missing from REDUCTION_OPS"


class TestBenchmarkReductions:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._reductions.measure_flops", return_value=mock_result
        ):
            result = benchmark_reductions(n=1_000_000, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(REDUCTION_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._reductions.measure_flops", return_value=mock_result
        ):
            result = benchmark_reductions(n=1_000_000, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_flops_per_element_calculation(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=500_000,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._reductions.measure_flops", return_value=mock_result
        ):
            result = benchmark_reductions(n=1_000_000, dtype="float64", repeats=2)

        # total_flops = 500K * 4 = 2M, per_element = 2M / (1M * 2) = 1.0
        for val in result.values():
            assert val == pytest.approx(1.0)
