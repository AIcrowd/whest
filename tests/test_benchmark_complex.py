"""Tests for complex-number benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._complex import (
    _FORMULA_STRINGS,
    _INSTRUCTIONS_OPS,
    COMPLEX_OPS,
    benchmark_complex,
)
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_complex_ops_non_empty(self):
        assert len(COMPLEX_OPS) > 0

    def test_contains_expected_perf_ops(self):
        for op in (
            "angle",
            "conj",
            "conjugate",
            "imag",
            "real",
            "real_if_close",
            "iscomplex",
            "isreal",
            "sort_complex",
        ):
            assert op in COMPLEX_OPS, f"{op} missing from COMPLEX_OPS"

    def test_contains_timing_only_ops(self):
        for op in ("iscomplexobj", "isrealobj"):
            assert op in COMPLEX_OPS, f"{op} missing from COMPLEX_OPS"
            assert op in _INSTRUCTIONS_OPS, f"{op} missing from _INSTRUCTIONS_OPS"

    def test_timing_only_ops_is_subset(self):
        assert _INSTRUCTIONS_OPS.issubset(set(COMPLEX_OPS))

    def test_exactly_11_ops(self):
        assert len(COMPLEX_OPS) == 11


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in COMPLEX_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"

    def test_formulas_are_strings(self):
        for op, formula in _FORMULA_STRINGS.items():
            assert isinstance(formula, str), f"{op} formula is not a string"
            assert len(formula) > 0, f"{op} formula is empty"

    def test_all_formulas_are_numel(self):
        for op, formula in _FORMULA_STRINGS.items():
            assert formula == "numel(output)", (
                f"{op} formula should be 'numel(output)', got '{formula}'"
            )


class TestBenchmarkComplex:
    def test_returns_tuple(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            rv = benchmark_complex(n=1_000, repeats=1)

        assert isinstance(rv, tuple)
        assert len(rv) == 2

    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            result, details = benchmark_complex(n=1_000, repeats=1)

        assert isinstance(result, dict)
        assert set(COMPLEX_OPS) == set(result.keys())

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            result, _details = benchmark_complex(n=1_000, repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_sort_complex_uses_smaller_n(self):
        """Verify sort_complex uses n=1_000_000, not 10_000_000."""
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            _result, details = benchmark_complex(n=10_000, repeats=1)

        # sort_complex should use op_n = 1_000_000 (not the n=10_000 we passed)
        # ... but we passed n=10_000 which is < 1_000_000. Actually the code
        # uses min of the override. Let's check analytical_flops instead.
        # When n=10_000, op_n for sort_complex = 1_000_000.
        assert details["sort_complex"]["analytical_flops"] == 1_000_000

    def test_analytical_flops_equal_n(self):
        """All complex ops use numel(output) = op_n as analytical cost."""
        n = 5_000
        mock_result = PerfResult(
            scalar_double=100_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            _result, details = benchmark_complex(n=n, repeats=1)

        for op, d in details.items():
            expected_n = 1_000_000 if op == "sort_complex" else n
            assert d["analytical_flops"] == expected_n, (
                f"{op}: expected analytical_flops={expected_n}, got {d['analytical_flops']}"
            )

    def test_alpha_normalization(self):
        """Verify alpha = total_flops / (analytical * repeats)."""
        n = 2_000
        repeats = 3
        total_flops = 60_000
        mock_result = PerfResult(
            scalar_double=total_flops,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            result, _details = benchmark_complex(n=n, repeats=repeats)

        # angle uses op_n = n = 2_000 as analytical cost
        expected = total_flops / (n * repeats)
        assert result["angle"] == pytest.approx(expected)

    def test_timing_only_ops_benchmark_size(self):
        """Type-check ops should have '(type check)' in benchmark_size."""
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        from benchmarks._perf import InstructionsResult
        mock_instr = InstructionsResult(instructions=1_000_000)
        with patch("benchmarks._complex.measure_flops", return_value=mock_result), \
             patch("benchmarks._complex.measure_instructions", return_value=mock_instr):
            _result, details = benchmark_complex(n=5_000, repeats=1)

        for op in _INSTRUCTIONS_OPS:
            assert "(instructions counter)" in details[op]["benchmark_size"], (
                f"{op} benchmark_size missing '(instructions counter)'"
            )

    def test_perf_ops_benchmark_size_no_instructions_tag(self):
        """Non-instructions ops should NOT have '(instructions counter)' in benchmark_size."""
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        from benchmarks._perf import InstructionsResult
        mock_instr = InstructionsResult(instructions=1_000_000)
        with patch("benchmarks._complex.measure_flops", return_value=mock_result), \
             patch("benchmarks._complex.measure_instructions", return_value=mock_instr):
            _result, details = benchmark_complex(n=5_000, repeats=1)

        for op in COMPLEX_OPS:
            if op not in _INSTRUCTIONS_OPS:
                assert "(instructions counter)" not in details[op]["benchmark_size"], (
                    f"{op} benchmark_size should not have '(instructions counter)'"
                )

    def test_details_keys_match_results(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            result, details = benchmark_complex(n=1_000, repeats=1)

        assert set(result.keys()) == set(details.keys())

    def test_details_schema(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._complex.measure_flops", return_value=mock_result):
            _result, details = benchmark_complex(n=1_000, repeats=1)

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
        for op, d in details.items():
            assert set(d.keys()) == expected_keys, f"{op} details keys mismatch"
            assert d["category"] == "counted_complex"
            assert isinstance(d["analytical_formula"], str)
            assert isinstance(d["analytical_flops"], int)
            assert "complex128" in d["benchmark_size"]
            assert isinstance(d["bench_code"], str)
            assert d["repeats"] == 1
            assert isinstance(d["perf_instructions_total"], list)
            assert isinstance(d["distribution_alphas"], list)
            assert len(d["distribution_alphas"]) == 3  # 3 distributions
