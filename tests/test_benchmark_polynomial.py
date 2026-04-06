"""Tests for polynomial benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._polynomial import POLYNOMIAL_OPS, benchmark_polynomial


class TestOpsLists:
    def test_polynomial_ops_non_empty(self):
        assert len(POLYNOMIAL_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("polyval", "polyfit", "polyadd", "polymul", "roots"):
            assert op in POLYNOMIAL_OPS, f"{op} missing from POLYNOMIAL_OPS"


class TestBenchmarkPolynomial:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._polynomial.measure_flops", return_value=mock_result
        ):
            result = benchmark_polynomial(
                n=1_000, dtype="float64", repeats=1, degree=5
            )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(POLYNOMIAL_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._polynomial.measure_flops", return_value=mock_result
        ):
            result = benchmark_polynomial(
                n=1_000, dtype="float64", repeats=1, degree=5
            )

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_polyval_normalizes_by_n(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=500,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._polynomial.measure_flops", return_value=mock_result
        ):
            result = benchmark_polynomial(
                n=1_000, dtype="float64", repeats=1, degree=5
            )

        # polyval: total_flops = 500*4 = 2000, normalized = 2000 / (1000 * 1) = 2.0
        assert result["polyval"] == pytest.approx(2.0)

    def test_polyadd_normalizes_by_degree(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=50,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._polynomial.measure_flops", return_value=mock_result
        ):
            result = benchmark_polynomial(
                n=1_000, dtype="float64", repeats=1, degree=10
            )

        # polyadd: total_flops = 50*4 = 200, normalized = 200 / (10 * 1) = 20.0
        assert result["polyadd"] == pytest.approx(20.0)
