"""Tests for contraction benchmark module."""

from unittest.mock import patch

from benchmarks._contractions import (
    CONTRACTION_OPS,
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
        assert _analytical_cost("inner") == 2 * 10_000

    def test_vdot_cost(self):
        assert _analytical_cost("vdot") == 2 * 10_000

    def test_vecdot_cost(self):
        assert _analytical_cost("vecdot") == 2 * 512 * 1000

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


class TestBenchmarkContractions:
    def test_returns_dict(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._contractions.measure_flops", return_value=mock_result):
            result = benchmark_contractions(dtype="float64", repeats=1)

        assert isinstance(result, dict)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._contractions.measure_flops", return_value=mock_result):
            result = benchmark_contractions(dtype="float64", repeats=1)

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
            result = benchmark_contractions(dtype="float64", repeats=1)

        expected = set(CONTRACTION_OPS) - {"vecdot"}
        assert expected.issubset(set(result.keys()))
