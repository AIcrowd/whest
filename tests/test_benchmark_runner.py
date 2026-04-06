"""Tests for the benchmark runner."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

from benchmarks.runner import run_benchmarks, normalize_weights, _BENCHMARK_FUNCS


def test_normalize_weights():
    raw = {"exp": 18.3, "add": 1.0, "log": 12.7}
    baseline_fpe = 1.0
    normalized = normalize_weights(raw, baseline_fpe)
    assert normalized["exp"] == 18.3
    assert normalized["add"] == 1.0
    assert normalized["log"] == 12.7


def test_normalize_weights_with_baseline():
    raw = {"exp": 36.6, "add": 2.0, "log": 25.4}
    baseline_fpe = 2.0
    normalized = normalize_weights(raw, baseline_fpe)
    assert normalized["exp"] == 36.6 / 2.0
    assert normalized["add"] == 1.0
    assert normalized["log"] == 25.4 / 2.0


def test_run_benchmarks_writes_json():
    mock_meta = {
        "timestamp": "2026-04-06",
        "hardware": {},
        "software": {},
        "benchmark_config": {"dtype": "float64", "repeats": 10, "distributions": 3},
    }

    mock_funcs = {
        "pointwise": lambda **kw: {"add": 1.0, "exp": 18.3},
        "reductions": lambda **kw: {"sum": 1.0},
        "linalg": lambda **kw: {"linalg.cholesky": 1.15},
        "fft": lambda **kw: {"fft.fft": 1.05},
        "sorting": lambda **kw: {"sort": 1.0},
        "random": lambda **kw: {"random.standard_normal": 4.2},
        "polynomial": lambda **kw: {"polyval": 1.0},
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        with (
            patch("benchmarks.runner.collect_metadata", return_value=mock_meta),
            patch("benchmarks.runner.measure_baseline", return_value=1.0),
            patch.dict(_BENCHMARK_FUNCS, mock_funcs),
            patch("benchmarks.runner.render_terminal", return_value="summary"),
        ):
            run_benchmarks(
                dtype="float64",
                output=output_path,
                categories=["all"],
                repeats=10,
            )

        with open(output_path) as f:
            data = json.load(f)
        assert "meta" in data
        assert "weights" in data
        assert data["weights"]["exp"] == 18.3
        assert data["weights"]["linalg.cholesky"] == 1.15
    finally:
        os.unlink(output_path)
