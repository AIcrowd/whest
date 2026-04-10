"""Tests for contraction benchmark module."""

from unittest.mock import patch

from benchmarks._contractions import (
    CONTRACTION_OPS,
    _BENCHMARK_SIZE_STRINGS,
    _FORMULA_STRINGS,
    _analytical_cost,
    benchmark_contractions,
)
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_contraction_ops_non_empty(self):
        assert len(CONTRACTION_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("dot", "matmul", "inner", "vdot", "outer", "einsum"):
            assert op in CONTRACTION_OPS, f"{op} missing from CONTRACTION_OPS"

    def test_expected_count(self):
        assert len(CONTRACTION_OPS) == 9


class TestAnalyticalCost:
    def test_dot_cost(self):
        assert _analytical_cost("dot") == 2 * 512 * 512 * 512

    def test_matmul_cost(self):
        assert _analytical_cost("matmul") == 2 * 512 * 512 * 512

    def test_inner_cost(self):
        assert _analytical_cost("inner") == 1_000_000  # a.size, no factor of 2

    def test_vdot_cost(self):
        assert _analytical_cost("vdot") == 1_000_000  # a.size, no factor of 2

    def test_vecdot_cost(self):
        assert _analytical_cost("vecdot") == 1000 * 512  # batch * contracted_axis

    def test_outer_cost(self):
        assert _analytical_cost("outer") == 5000 * 5000

    def test_tensordot_cost(self):
        assert _analytical_cost("tensordot") == 2 * 64**5

    def test_kron_cost(self):
        assert _analytical_cost("kron") == 64**4

    def test_einsum_cost(self):
        assert _analytical_cost("einsum") == 2 * 512 * 512 * 512

    def test_all_ops_have_cost(self):
        """Every op in CONTRACTION_OPS has an analytical cost entry."""
        for op in CONTRACTION_OPS:
            cost = _analytical_cost(op)
            assert isinstance(cost, int), f"{op} cost is not int"
            assert cost > 0, f"{op} cost must be positive"


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in CONTRACTION_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"
            assert isinstance(_FORMULA_STRINGS[op], str)
            assert len(_FORMULA_STRINGS[op]) > 0

    def test_all_ops_have_benchmark_size(self):
        for op in CONTRACTION_OPS:
            assert op in _BENCHMARK_SIZE_STRINGS, (
                f"{op} missing from _BENCHMARK_SIZE_STRINGS"
            )
            assert isinstance(_BENCHMARK_SIZE_STRINGS[op], str)
            assert len(_BENCHMARK_SIZE_STRINGS[op]) > 0


class TestBenchmarkContractions:
    def test_returns_tuple(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._contractions.measure_flops", return_value=mock_result):
            result, details = benchmark_contractions(dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert isinstance(details, dict)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._contractions.measure_flops", return_value=mock_result):
            result, _details = benchmark_contractions(dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_all_non_vecdot_ops_present(self):
        """All ops except possibly vecdot (NumPy 2.x) are present."""
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._contractions.measure_flops", return_value=mock_result):
            result, _details = benchmark_contractions(dtype="float64", repeats=1)

        expected = set(CONTRACTION_OPS) - {"vecdot"}
        assert expected.issubset(set(result.keys()))

    def test_details_have_required_keys(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._contractions.measure_flops", return_value=mock_result):
            _result, details = benchmark_contractions(dtype="float64", repeats=1)

        required_keys = {
            "category",
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
