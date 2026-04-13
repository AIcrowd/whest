"""Tests for bitwise/integer benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._bitwise import (
    _FORMULA_STRINGS,
    BINARY_OPS,
    BITWISE_OPS,
    SHIFT_OPS,
    SPECIAL_OPS,
    UNARY_OPS,
    _analytical_cost,
    benchmark_bitwise,
)
from benchmarks._perf import InstructionsResult, TimingResult


class TestOpsLists:
    def test_unary_ops_contents(self):
        expected = {"bitwise_not", "bitwise_invert", "bitwise_count", "invert"}
        assert set(UNARY_OPS) == expected

    def test_binary_ops_contents(self):
        expected = {"bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "lcm"}
        assert set(BINARY_OPS) == expected

    def test_shift_ops_contents(self):
        expected = {
            "bitwise_left_shift",
            "bitwise_right_shift",
            "left_shift",
            "right_shift",
        }
        assert set(SHIFT_OPS) == expected

    def test_special_ops_contents(self):
        assert "isnat" in SPECIAL_OPS

    def test_bitwise_ops_is_union(self):
        assert set(BITWISE_OPS) == (
            set(UNARY_OPS) | set(BINARY_OPS) | set(SHIFT_OPS) | set(SPECIAL_OPS)
        )

    def test_total_op_count(self):
        # 13 bitwise/integer ops + isnat = 14
        assert len(BITWISE_OPS) == 14


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in BITWISE_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"

    def test_all_formulas_are_n(self):
        for op, formula in _FORMULA_STRINGS.items():
            assert formula == "n", f"{op} has unexpected formula: {formula}"


class TestAnalyticalCost:
    def test_cost_equals_n(self):
        for op in BITWISE_OPS:
            assert _analytical_cost(op, 1000) == 1000
            assert _analytical_cost(op, 10_000_000) == 10_000_000


class TestBenchmarkBitwise:
    def test_returns_tuple_with_alphas_and_details(self):
        mock_result = InstructionsResult(instructions=5_000_000)
        with patch("benchmarks._bitwise.measure_instructions", return_value=mock_result):
            alphas, details = benchmark_bitwise(n=1_000, repeats=1)

        assert isinstance(alphas, dict)
        assert isinstance(details, dict)
        assert set(alphas.keys()) == set(BITWISE_OPS)
        assert set(details.keys()) == set(BITWISE_OPS)

    def test_values_are_floats(self):
        mock_result = InstructionsResult(instructions=2_000_000)
        with patch("benchmarks._bitwise.measure_instructions", return_value=mock_result):
            alphas, _details = benchmark_bitwise(n=1_000, repeats=1)

        for key, val in alphas.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_alpha_calculation(self):
        # elapsed_ns=10_000_000, n=1000, repeats=5
        # alpha = 10_000_000 / (1000 * 5) = 2000.0
        mock_result = InstructionsResult(instructions=10_000_000)
        with patch("benchmarks._bitwise.measure_instructions", return_value=mock_result):
            alphas, _details = benchmark_bitwise(n=1_000, repeats=5)

        for val in alphas.values():
            assert val == pytest.approx(2000.0)

    def test_details_schema(self):
        mock_result = InstructionsResult(instructions=1_000_000)
        n = 1_000
        with patch("benchmarks._bitwise.measure_instructions", return_value=mock_result):
            _alphas, details = benchmark_bitwise(n=n, repeats=5)

        expected_keys = {
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

        # Check a unary op
        d = details["bitwise_not"]
        assert set(d.keys()) == expected_keys
        assert d["category"] == "instructions_unary"
        assert d["measurement_mode"] == "instructions"
        assert d["analytical_formula"] == "n"
        assert d["analytical_flops"] == n
        assert d["benchmark_size"] == f"x: ({n},)"
        assert d["repeats"] == 5
        assert len(d["distribution_alphas"]) == 3

        # Check a binary op
        d = details["bitwise_and"]
        assert d["category"] == "instructions_binary"
        assert d["benchmark_size"] == f"a: ({n},), b: ({n},)"

        # Check a shift op
        d = details["left_shift"]
        assert d["category"] == "instructions_shift"
        assert d["benchmark_size"] == f"a: ({n},), b: ({n},) (values 0-10)"

        # Check isnat
        d = details["isnat"]
        assert d["category"] == "instructions_special"
        assert d["measurement_mode"] == "instructions"
        assert "datetime64" in d["benchmark_size"]

    def test_uses_instructions_mode(self):
        """Verify measure_instructions is called (not measure_flops)."""
        call_count = [0]

        def fake_measure(setup, bench, repeats=10):
            call_count[0] += 1
            return InstructionsResult(instructions=1_000)

        with patch("benchmarks._bitwise.measure_instructions", side_effect=fake_measure):
            benchmark_bitwise(n=100, repeats=1)

        # measure_instructions should have been called for each op × distribution
        assert call_count[0] > 0

    def test_handles_runtime_error(self):
        """Verify RuntimeError in measurement is caught gracefully."""
        with patch(
            "benchmarks._bitwise.measure_instructions",
            side_effect=RuntimeError("boom"),
        ):
            alphas, details = benchmark_bitwise(n=100, repeats=1)
        # RuntimeError is caught inside the loop, so we get empty results
        assert len(alphas) == 0

    def test_dtype_param_ignored(self):
        """dtype param exists for interface consistency but ops always use int64."""
        mock_result = InstructionsResult(instructions=1_000_000)
        with patch("benchmarks._bitwise.measure_instructions", return_value=mock_result):
            a1, _ = benchmark_bitwise(n=100, dtype="int64", repeats=1)
            a2, _ = benchmark_bitwise(n=100, dtype="float64", repeats=1)
        # Should produce identical results since dtype is ignored
        assert set(a1.keys()) == set(a2.keys())
