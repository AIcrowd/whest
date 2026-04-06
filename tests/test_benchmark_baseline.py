"""Tests for baseline (np.add) measurement."""

from unittest.mock import patch

import pytest

from benchmarks._baseline import measure_baseline
from benchmarks._perf import PerfResult


def test_measure_baseline_returns_flops_per_element():
    mock_result = PerfResult(
        scalar_double=0,
        packed_128_double=0,
        packed_256_double=5_000_000,  # 5M 256-bit packed = 20M flops
        packed_512_double=0,
    )
    with patch("benchmarks._baseline.measure_flops", return_value=mock_result):
        fpe = measure_baseline(n=10_000_000, dtype="float64", repeats=10)
        # total_flops = 5M * 4 = 20M, per_element = 20M / (10M * 10) = 0.2
        assert fpe == pytest.approx(0.2)


def test_measure_baseline_different_sizes():
    mock_result = PerfResult(
        scalar_double=1_000_000,
        packed_128_double=0,
        packed_256_double=0,
        packed_512_double=0,
    )
    with patch("benchmarks._baseline.measure_flops", return_value=mock_result):
        fpe = measure_baseline(n=1_000_000, dtype="float64", repeats=1)
        assert fpe == pytest.approx(1.0)
