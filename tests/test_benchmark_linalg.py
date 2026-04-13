"""Tests for linalg benchmark module."""

from unittest.mock import patch

from benchmarks._linalg import (
    _FORMULA_STRINGS,
    LINALG_OPS,
    _analytical_cost,
    benchmark_linalg,
)
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_linalg_ops_non_empty(self):
        assert len(LINALG_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("linalg.cholesky", "linalg.qr", "linalg.svd", "linalg.solve"):
            assert op in LINALG_OPS, f"{op} missing from LINALG_OPS"


class TestAnalyticalCost:
    def test_cholesky_cost(self):
        assert _analytical_cost("linalg.cholesky", 100) == 100**3

    def test_qr_cost(self):
        n = 100
        expected = n * n * min(n, n)
        assert _analytical_cost("linalg.qr", n) == expected

    def test_solve_cost(self):
        n = 100
        expected = n**3
        assert _analytical_cost("linalg.solve", n) == expected

    def test_det_cost(self):
        n = 100
        expected = n**3
        assert _analytical_cost("linalg.det", n) == expected

    def test_inv_cost(self):
        n = 100
        assert _analytical_cost("linalg.inv", n) == n**3


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in LINALG_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"
            assert isinstance(_FORMULA_STRINGS[op], str)
            assert len(_FORMULA_STRINGS[op]) > 0


class TestBenchmarkLinalg:
    def test_returns_tuple_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._linalg.measure_flops", return_value=mock_result):
            result, details = benchmark_linalg(n=64, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert isinstance(details, dict)
        assert set(result.keys()) == set(LINALG_OPS)
        assert set(details.keys()) == set(LINALG_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._linalg.measure_flops", return_value=mock_result):
            result, _details = benchmark_linalg(n=64, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_details_have_required_keys(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._linalg.measure_flops", return_value=mock_result):
            _result, details = benchmark_linalg(n=64, dtype="float64", repeats=1)

        required_keys = {
            "category",
            "measurement_mode",
            "analytical_formula",
            "analytical_flops",
            "benchmark_size",
            "bench_code",
            "repeats",
            "perf_instructions_total",
            "distribution_alphas",
        }
        for op, d in details.items():
            assert required_keys.issubset(d.keys()), (
                f"{op} missing keys: {required_keys - set(d.keys())}"
            )
            assert d["category"] == "counted_custom"
            assert isinstance(d["analytical_flops"], int)
            assert isinstance(d["bench_code"], str)
            assert isinstance(d["distribution_alphas"], list)
            assert len(d["distribution_alphas"]) > 0
            assert "(64,64)" in d["benchmark_size"]
