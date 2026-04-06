"""Tests for linalg benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._linalg import LINALG_OPS, _analytical_cost, benchmark_linalg
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_linalg_ops_non_empty(self):
        assert len(LINALG_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("linalg.cholesky", "linalg.qr", "linalg.svd", "linalg.solve"):
            assert op in LINALG_OPS, f"{op} missing from LINALG_OPS"


class TestAnalyticalCost:
    def test_cholesky_cost(self):
        assert _analytical_cost("linalg.cholesky", 100) == 100**3 // 3

    def test_qr_cost(self):
        n = 100
        expected = 2 * n * n**2 - 2 * n**3 // 3
        assert _analytical_cost("linalg.qr", n) == expected

    def test_solve_cost(self):
        n = 100
        expected = 2 * n**3 // 3 + 2 * n**2
        assert _analytical_cost("linalg.solve", n) == expected

    def test_det_cost(self):
        n = 100
        expected = 2 * n**3 // 3
        assert _analytical_cost("linalg.det", n) == expected

    def test_inv_cost(self):
        n = 100
        assert _analytical_cost("linalg.inv", n) == n**3


class TestBenchmarkLinalg:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._linalg.measure_flops", return_value=mock_result
        ):
            result = benchmark_linalg(n=64, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(LINALG_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch(
            "benchmarks._linalg.measure_flops", return_value=mock_result
        ):
            result = benchmark_linalg(n=64, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"
