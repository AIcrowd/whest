"""Tests for polynomial benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._polynomial import POLYNOMIAL_OPS, _analytical_cost, benchmark_polynomial


class TestOpsLists:
    def test_polynomial_ops_non_empty(self):
        assert len(POLYNOMIAL_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("polyval", "polyfit", "polyadd", "polymul", "roots"):
            assert op in POLYNOMIAL_OPS, f"{op} missing from POLYNOMIAL_OPS"


class TestAnalyticalCost:
    def test_polyval(self):
        assert _analytical_cost("polyval", 1000, 5) == 2 * 1000 * 5

    def test_polyfit(self):
        assert _analytical_cost("polyfit", 1000, 5) == 2 * 1000 * 6**2

    def test_roots(self):
        assert _analytical_cost("roots", 100, 10) == 10 * 10**3

    def test_polymul(self):
        assert _analytical_cost("polymul", 100, 10) == 11**2

    def test_polydiv(self):
        assert _analytical_cost("polydiv", 100, 10) == 11**2

    def test_polyadd(self):
        assert _analytical_cost("polyadd", 100, 10) == 11

    def test_polysub(self):
        assert _analytical_cost("polysub", 100, 10) == 11

    def test_polyder(self):
        assert _analytical_cost("polyder", 100, 10) == 10

    def test_polyint(self):
        assert _analytical_cost("polyint", 100, 10) == 10

    def test_poly(self):
        assert _analytical_cost("poly", 100, 10) == 100

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="Unknown polynomial op"):
            _analytical_cost("bogus", 100, 10)

    def test_all_ops_covered(self):
        """Every op in POLYNOMIAL_OPS has an analytical cost entry."""
        for op in POLYNOMIAL_OPS:
            cost = _analytical_cost(op, 1000, 10)
            assert cost > 0, f"{op} returned non-positive cost"


class TestBenchmarkPolynomial:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._polynomial.measure_flops", return_value=mock_result):
            result = benchmark_polynomial(n=1_000, dtype="float64", repeats=1, degree=5)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(POLYNOMIAL_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._polynomial.measure_flops", return_value=mock_result):
            result = benchmark_polynomial(n=1_000, dtype="float64", repeats=1, degree=5)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_polyval_normalizes_by_analytical_cost(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=500,
            packed_512_double=0,
        )
        n, degree = 1_000, 5
        with patch("benchmarks._polynomial.measure_flops", return_value=mock_result):
            result = benchmark_polynomial(n=n, dtype="float64", repeats=1, degree=degree)

        # polyval: total_flops = 500*4 = 2000
        # analytical = 2 * 1000 * 5 = 10000
        # normalized = 2000 / 10000 = 0.2
        expected = 2000.0 / _analytical_cost("polyval", n, degree)
        assert result["polyval"] == pytest.approx(expected)

    def test_polyadd_normalizes_by_analytical_cost(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=50,
            packed_512_double=0,
        )
        n, degree = 1_000, 10
        with patch("benchmarks._polynomial.measure_flops", return_value=mock_result):
            result = benchmark_polynomial(
                n=n, dtype="float64", repeats=1, degree=degree
            )

        # polyadd: total_flops = 50*4 = 200
        # analytical = degree + 1 = 11
        # normalized = 200 / 11 ≈ 18.18
        expected = 200.0 / _analytical_cost("polyadd", n, degree)
        assert result["polyadd"] == pytest.approx(expected)
