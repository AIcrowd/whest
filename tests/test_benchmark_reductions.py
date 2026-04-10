"""Tests for reductions benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._reductions import REDUCTION_OPS, benchmark_reductions


class TestOpsLists:
    def test_reduction_ops_non_empty(self):
        assert len(REDUCTION_OPS) > 0

    def test_contains_expected_ops(self):
        for op in (
            "sum",
            "mean",
            "std",
            "argmax",
            "nansum",
            "median",
            "percentile",
            "count_nonzero",
        ):
            assert op in REDUCTION_OPS, f"{op} missing from REDUCTION_OPS"


class TestBenchmarkReductions:
    def test_returns_tuple_with_alphas_and_details(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._reductions.measure_flops", return_value=mock_result):
            result, details = benchmark_reductions(
                n=1_000_000, dtype="float64", repeats=1
            )

        assert isinstance(result, dict)
        assert isinstance(details, dict)
        assert set(result.keys()) == set(REDUCTION_OPS)
        assert set(details.keys()) == set(REDUCTION_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._reductions.measure_flops", return_value=mock_result):
            result, _details = benchmark_reductions(
                n=1_000_000, dtype="float64", repeats=1
            )

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_flops_per_element_calculation(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=500_000,
            packed_512_double=0,
        )
        with patch("benchmarks._reductions.measure_flops", return_value=mock_result):
            result, _details = benchmark_reductions(
                n=1_000_000, dtype="float64", repeats=2
            )

        # total_flops = 500K * 4 = 2M, per_element = 2M / (1M * 2) = 1.0
        for val in result.values():
            assert val == pytest.approx(1.0)

    def test_details_schema(self):
        """Verify details dict has the expected keys and types for each op."""
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        n = 1_000_000
        with patch("benchmarks._reductions.measure_flops", return_value=mock_result):
            _result, details = benchmark_reductions(
                n=n, dtype="float64", repeats=5
            )

        expected_detail_keys = {
            "category",
            "analytical_formula",
            "analytical_flops",
            "benchmark_size",
            "bench_code",
            "repeats",
            "perf_instructions_total",
            "distribution_alphas",
        }

        # Check a standard reduction op
        sum_detail = details["sum"]
        assert set(sum_detail.keys()) == expected_detail_keys
        assert sum_detail["category"] == "counted_reduction"
        assert sum_detail["analytical_formula"] == "numel(input)"
        assert sum_detail["analytical_flops"] == n
        assert sum_detail["benchmark_size"] == f"x: ({n},)"
        assert isinstance(sum_detail["bench_code"], str)
        assert sum_detail["repeats"] == 5
        assert isinstance(sum_detail["perf_instructions_total"], list)
        assert len(sum_detail["perf_instructions_total"]) == 3  # 3 distributions
        assert isinstance(sum_detail["distribution_alphas"], list)
        assert len(sum_detail["distribution_alphas"]) == 3
