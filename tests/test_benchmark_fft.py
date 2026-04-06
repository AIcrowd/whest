"""Tests for FFT benchmark module."""

from unittest.mock import patch

from benchmarks._fft import FFT_OPS, _analytical_cost, benchmark_fft
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


class TestBenchmarkFFT:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            result = benchmark_fft(n=1024, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(FFT_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._fft.measure_flops", return_value=mock_result):
            result = benchmark_fft(n=1024, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"
