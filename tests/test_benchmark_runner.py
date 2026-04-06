"""Tests for the benchmark runner."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

from benchmarks.runner import run_benchmarks, normalize_weights


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
    mock_weights = {"add": 1.0, "exp": 18.3}
    mock_meta = {
        "timestamp": "2026-04-06",
        "hardware": {},
        "software": {},
        "benchmark_config": {"dtype": "float64", "repeats": 10, "distributions": 3},
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        with (
            patch("benchmarks.runner.collect_metadata", return_value=mock_meta),
            patch("benchmarks.runner.measure_baseline", return_value=1.0),
            patch("benchmarks.runner.benchmark_pointwise", return_value={"add": 1.0, "exp": 18.3}),
            patch("benchmarks.runner.benchmark_reductions", return_value={"sum": 1.0}),
            patch("benchmarks.runner.benchmark_linalg", return_value={"linalg.cholesky": 1.15}),
            patch("benchmarks.runner.benchmark_fft", return_value={"fft.fft": 1.05}),
            patch("benchmarks.runner.benchmark_sorting", return_value={"sort": 1.0}),
            patch("benchmarks.runner.benchmark_random", return_value={"random.standard_normal": 4.2}),
            patch("benchmarks.runner.benchmark_polynomial", return_value={"polyval": 1.0}),
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
