"""Tests for FFT benchmark module."""

from unittest.mock import patch

from benchmarks._fft import FFT_OPS, _FORMULA_STRINGS, _analytical_cost, benchmark_fft
from benchmarks._perf import PerfResult


class TestOpsLists:
    def test_fft_ops_non_empty(self):
        assert len(FFT_OPS) > 0

    def test_contains_expected_ops(self):
        for op in ("fft.fft", "fft.ifft", "fft.rfft", "fft.fft2", "fft.fftn"):
            assert op in FFT_OPS, f"{op} missing from FFT_OPS"


class TestAnalyticalCost:
    def test_fft_cost(self):
        n = 1024
        # 5 * n * ceil(log2(n)) = 5 * 1024 * 10 = 51200
        assert _analytical_cost("fft.fft", n) == 5 * 1024 * 10

    def test_rfft_cost(self):
        n = 1024
        # 5 * (n//2) * ceil(log2(n)) = 5 * 512 * 10 = 25600
        assert _analytical_cost("fft.rfft", n) == 5 * 512 * 10

    def test_ifft_cost(self):
        n = 1024
        # Not an rfft variant => 5 * n * ceil(log2(n))
        assert _analytical_cost("fft.ifft", n) == 5 * 1024 * 10


class TestFormulaStrings:
    def test_all_ops_have_formula(self):
        for op in FFT_OPS:
            assert op in _FORMULA_STRINGS, f"{op} missing from _FORMULA_STRINGS"
            assert isinstance(_FORMULA_STRINGS[op], str)
            assert len(_FORMULA_STRINGS[op]) > 0


class TestBenchmarkFFT:
    def test_returns_tuple_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            result, details = benchmark_fft(n=1024, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert isinstance(details, dict)
        assert set(result.keys()) == set(FFT_OPS)
        assert set(details.keys()) == set(FFT_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            result, _details = benchmark_fft(n=1024, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_details_have_required_keys(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            _result, details = benchmark_fft(n=1024, dtype="float64", repeats=1)

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

    def test_1d_ops_have_n_size(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            _result, details = benchmark_fft(n=1024, dtype="float64", repeats=1)

        # 1D ops should have explicit input shape
        assert details["fft.fft"]["benchmark_size"] == "x: (1024,)"

    def test_2d_ops_have_side_size(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            _result, details = benchmark_fft(n=1024, dtype="float64", repeats=1)

        # 2D ops should have explicit 2D input shape (isqrt(1024) = 32)
        assert details["fft.fft2"]["benchmark_size"] == "x: (32,32)"
