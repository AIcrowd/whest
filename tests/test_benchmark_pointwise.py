"""Tests for pointwise benchmark module."""

from unittest.mock import patch

import numpy as np
import pytest

from benchmarks._perf import PerfResult
from benchmarks._pointwise import (
    BINARY_OPS,
    SPECIAL_OPS,
    UNARY_OPS,
    _make_inputs_binary,
    _make_inputs_unary,
    benchmark_pointwise,
)


class TestOpsLists:
    def test_unary_ops_non_empty(self):
        assert len(UNARY_OPS) > 0

    def test_binary_ops_non_empty(self):
        assert len(BINARY_OPS) > 0

    def test_unary_ops_contains_expected(self):
        for op in ("abs", "exp", "sin", "sqrt", "logical_not"):
            assert op in UNARY_OPS, f"{op} missing from UNARY_OPS"

    def test_unary_ops_contains_new_ops(self):
        for op in (
            "frexp",
            "modf",
            "sinc",
            "i0",
            "spacing",
            "nan_to_num",
            "isneginf",
            "isposinf",
        ):
            assert op in UNARY_OPS, f"{op} missing from UNARY_OPS"

    def test_binary_ops_contains_expected(self):
        for op in ("add", "subtract", "multiply", "divide", "maximum", "logaddexp"):
            assert op in BINARY_OPS, f"{op} missing from BINARY_OPS"

    def test_special_ops_contains_expected(self):
        for op in ("isclose", "heaviside", "clip"):
            assert op in SPECIAL_OPS, f"{op} missing from SPECIAL_OPS"


class TestMakeInputs:
    def test_make_inputs_unary_returns_3_arrays(self):
        inputs = _make_inputs_unary(100, "float64")
        assert len(inputs) == 3
        for arr in inputs:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (100,)
            assert arr.dtype == np.float64

    def test_make_inputs_binary_returns_3_tuples(self):
        inputs = _make_inputs_binary(100, "float64")
        assert len(inputs) == 3
        for a, b in inputs:
            assert isinstance(a, np.ndarray)
            assert isinstance(b, np.ndarray)
            assert a.shape == (100,)
            assert b.shape == (100,)

    def test_make_inputs_unary_respects_dtype(self):
        inputs = _make_inputs_unary(50, "float32")
        for arr in inputs:
            assert arr.dtype == np.float32


class TestBenchmarkPointwise:
    def test_returns_tuple_with_alphas_and_details(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._pointwise.measure_flops", return_value=mock_result):
            result, details = benchmark_pointwise(
                n=1_000_000, dtype="float64", repeats=1, distributions=1
            )

        assert isinstance(result, dict)
        assert isinstance(details, dict)
        expected_keys = set(UNARY_OPS) | set(BINARY_OPS) | set(SPECIAL_OPS)
        assert set(result.keys()) == expected_keys
        assert set(details.keys()) == expected_keys

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._pointwise.measure_flops", return_value=mock_result):
            result, _details = benchmark_pointwise(
                n=1_000_000, dtype="float64", repeats=1, distributions=1
            )

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_flops_per_element_calculation(self):
        # 2M packed_256 = 8M flops; n=1M, repeats=2 => 4.0 per element
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=2_000_000,
            packed_512_double=0,
        )
        with patch("benchmarks._pointwise.measure_flops", return_value=mock_result):
            result, _details = benchmark_pointwise(
                n=1_000_000, dtype="float64", repeats=2, distributions=1
            )

        # total_flops = 2M * 4 = 8M, per_element = 8M / (1M * 2) = 4.0
        for val in result.values():
            assert val == pytest.approx(4.0)

    def test_details_schema(self):
        """Verify details dict has the expected keys and types for each op."""
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        n = 1_000_000
        with patch("benchmarks._pointwise.measure_flops", return_value=mock_result):
            _result, details = benchmark_pointwise(
                n=n, dtype="float64", repeats=5, distributions=2
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

        # Check a unary op
        sin_detail = details["sin"]
        assert set(sin_detail.keys()) == expected_detail_keys
        assert sin_detail["category"] == "counted_unary"
        assert sin_detail["analytical_formula"] == "numel(output)"
        assert sin_detail["analytical_flops"] == n
        assert sin_detail["benchmark_size"] == f"n={n}"
        assert isinstance(sin_detail["bench_code"], str)
        assert sin_detail["repeats"] == 5
        assert isinstance(sin_detail["perf_instructions_total"], list)
        assert len(sin_detail["perf_instructions_total"]) == 2  # distributions=2
        assert isinstance(sin_detail["distribution_alphas"], list)
        assert len(sin_detail["distribution_alphas"]) == 2

        # Check a binary op
        add_detail = details["add"]
        assert add_detail["category"] == "counted_binary"

        # Check a special op
        clip_detail = details["clip"]
        assert clip_detail["category"] == "counted_unary"
        isclose_detail = details["isclose"]
        assert isclose_detail["category"] == "counted_binary"
