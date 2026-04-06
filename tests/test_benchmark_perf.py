"""Tests for the perf stat wrapper and wall-clock fallback."""

from unittest.mock import patch

import pytest

from benchmarks._perf import (
    PerfResult,
    TimingResult,
    _parse_perf_csv,
    has_perf,
    measure_flops,
    measurement_mode,
)

SAMPLE_PERF_CSV = """\
# Started on ...
123456,,fp_arith_inst_retired.scalar_double,1000000,100.00,,
789012,,fp_arith_inst_retired.128b_packed_double,1000000,100.00,,
345678,,fp_arith_inst_retired.256b_packed_double,1000000,100.00,,
0,,fp_arith_inst_retired.512b_packed_double,1000000,100.00,,
"""


def test_parse_perf_csv():
    result = _parse_perf_csv(SAMPLE_PERF_CSV)
    assert isinstance(result, PerfResult)
    assert result.scalar_double == 123456
    assert result.packed_128_double == 789012
    assert result.packed_256_double == 345678
    assert result.packed_512_double == 0
    # total = 123456*1 + 789012*2 + 345678*4 + 0*8
    assert result.total_flops == 123456 + 789012 * 2 + 345678 * 4


def test_parse_perf_csv_with_not_supported():
    csv_with_unsupported = """\
123456,,fp_arith_inst_retired.scalar_double,1000000,100.00,,
<not supported>,,fp_arith_inst_retired.128b_packed_double,0,0.00,,
0,,fp_arith_inst_retired.256b_packed_double,0,0.00,,
0,,fp_arith_inst_retired.512b_packed_double,0,0.00,,
"""
    result = _parse_perf_csv(csv_with_unsupported)
    assert result.packed_128_double == 0
    assert result.total_flops == 123456


def test_timing_result_total_flops():
    result = TimingResult(elapsed_ns=5_000_000)
    assert result.total_flops == 5_000_000


def test_measurement_mode():
    mode = measurement_mode()
    assert mode in ("perf", "timing")


def test_measure_flops_timing_fallback():
    """When perf is unavailable, measure_flops uses wall-clock timing."""
    with patch("benchmarks._perf.has_perf", return_value=False):
        result = measure_flops(
            "x = np.ones(1000); y = np.ones(1000)",
            "np.add(x, y)",
            repeats=5,
        )
        assert isinstance(result, TimingResult)
        assert result.total_flops > 0


@pytest.mark.skipif(not has_perf(), reason="perf not available")
def test_measure_flops_perf_integration():
    """Smoke test: measure np.add on a small array with perf."""
    result = measure_flops(
        "x = np.ones(1000); y = np.ones(1000)",
        "np.add(x, y)",
        repeats=5,
    )
    assert isinstance(result, PerfResult)
    assert result.total_flops >= 1000
