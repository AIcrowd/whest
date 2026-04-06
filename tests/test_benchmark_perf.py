"""Tests for the perf stat wrapper."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from benchmarks._perf import (
    PerfResult,
    _parse_perf_csv,
    has_perf,
    measure_flops,
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


@pytest.mark.skipif(not has_perf(), reason="perf not available")
def test_measure_flops_integration():
    """Smoke test: measure np.add on a small array."""
    import numpy as np

    x = np.ones(1000, dtype=np.float64)
    y = np.ones(1000, dtype=np.float64)
    result = measure_flops(
        "import numpy as np; x = np.ones(1000); y = np.ones(1000)",
        "np.add(x, y)",
        repeats=5,
    )
    # np.add on 1000 elements should produce at least 1000 FP ops total
    assert result.total_flops >= 1000
