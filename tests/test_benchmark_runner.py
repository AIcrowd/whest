"""Tests for the benchmark runner."""

import json
import os
import tempfile
from unittest.mock import patch

from benchmarks._baseline import BaselineResult
from benchmarks.runner import (
    _BENCHMARK_FUNCS,
    _unpack_benchmark_result,
    normalize_weights,
    run_benchmarks,
)


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
        "pointwise": lambda **kw: ({"add": 1.0, "exp": 18.3}, {}),
        "reductions": lambda **kw: ({"sum": 1.0}, {}),
        "linalg": lambda **kw: ({"linalg.cholesky": 1.15}, {}),
        "fft": lambda **kw: ({"fft.fft": 1.05}, {}),
        "sorting": lambda **kw: ({"sort": 1.0}, {}),
        "random": lambda **kw: ({"random.standard_normal": 4.2}, {}),
        "polynomial": lambda **kw: ({"polyval": 1.0}, {}),
        "contractions": lambda **kw: ({"einsum": 1.0}, {}),
        "misc": lambda **kw: ({"where": 1.0}, {}),
        "window": lambda **kw: ({"bartlett": 1.0}, {}),
    }

    # With zero overhead, normalize_weights_v2 just clamps to max(alpha, 1.0)
    mock_baselines = BaselineResult(alpha_add=1.0, alpha_abs=0.0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        with (
            patch("benchmarks.runner.collect_metadata", return_value=mock_meta),
            patch("benchmarks.runner.measure_baselines", return_value=mock_baselines),
            patch.dict(_BENCHMARK_FUNCS, mock_funcs, clear=True),
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
        # With zero overhead: weight = max(alpha - 0, 1.0)
        assert data["weights"]["exp"] == 18.3
        assert data["weights"]["linalg.cholesky"] == 1.15
    finally:
        os.unlink(output_path)


def test_unpack_benchmark_result_tuple():
    """Tuple return is unpacked correctly."""
    alphas = {"sin": 1.5}
    details = {"sin": {"bench_code": "np.sin(x, out=_out)"}}
    a, d = _unpack_benchmark_result((alphas, details))
    assert a is alphas
    assert d is details


def test_unpack_benchmark_result_dict():
    """Legacy dict-only return produces empty details."""
    raw = {"sin": 1.5}
    a, d = _unpack_benchmark_result(raw)
    assert a is raw
    assert d == {}


def test_run_benchmarks_per_op_details():
    """per_op_details is populated when benchmark functions return details."""
    mock_meta = {
        "timestamp": "2026-04-06",
        "hardware": {},
        "software": {},
        "benchmark_config": {"dtype": "float64", "repeats": 10, "distributions": 3},
    }

    sample_details = {
        "exp": {
            "category": "counted_unary",
            "measurement_mode": "ufunc_unary",
            "analytical_formula": "numel(output)",
            "analytical_flops": 10_000_000,
            "benchmark_size": "n=10000000",
            "bench_code": "np.exp(x, out=_out)",
            "repeats": 10,
            "perf_instructions_total": [180_000_000],
            "distribution_alphas": [18.0, 18.3, 18.6],
        },
    }

    mock_funcs = {
        "pointwise": lambda **kw: ({"exp": 18.3}, sample_details),
    }

    with (
        patch("benchmarks.runner.collect_metadata", return_value=mock_meta),
        patch(
            "benchmarks.runner.measure_baselines",
            return_value=BaselineResult(alpha_add=1.0, alpha_abs=0.0),
        ),
        patch.dict(_BENCHMARK_FUNCS, mock_funcs, clear=True),
        patch("benchmarks.runner.render_terminal", return_value="summary"),
    ):
        result = run_benchmarks(
            dtype="float64",
            categories=["pointwise"],
            repeats=10,
        )

    meta = result["meta"]
    assert "per_op_details" in meta
    details = meta["per_op_details"]
    assert "exp" in details
    d = details["exp"]

    # Check enriched fields
    assert "perf_weight" in d
    assert d["perf_weight"] == result["weights"]["exp"]
    assert "absolute_alpha" in d
    assert "baseline_alpha" in d
    assert "baseline_analytical_flops" in d
    assert "baseline_bench_code" in d
    assert "baseline_perf_instructions_total" in d
    assert "notes" in d

    # Original fields preserved
    assert d["bench_code"] == "np.exp(x, out=_out)"
    assert d["analytical_flops"] == 10_000_000


def test_run_benchmarks_backward_compat_dict_return():
    """Runner handles legacy dict-only benchmark functions gracefully."""
    mock_meta = {
        "timestamp": "2026-04-06",
        "hardware": {},
        "software": {},
        "benchmark_config": {"dtype": "float64", "repeats": 10, "distributions": 3},
    }

    # Legacy dict return (no details)
    mock_funcs = {
        "pointwise": lambda **kw: {"add": 1.0, "exp": 18.3},
    }

    with (
        patch("benchmarks.runner.collect_metadata", return_value=mock_meta),
        patch(
            "benchmarks.runner.measure_baselines",
            return_value=BaselineResult(alpha_add=1.0, alpha_abs=0.0),
        ),
        patch.dict(_BENCHMARK_FUNCS, mock_funcs, clear=True),
        patch("benchmarks.runner.render_terminal", return_value="summary"),
    ):
        result = run_benchmarks(
            dtype="float64",
            categories=["pointwise"],
            repeats=10,
        )

    # Should not crash, per_op_details should be empty dict
    assert "per_op_details" in result["meta"]
    assert result["meta"]["per_op_details"] == {}
    assert result["weights"]["exp"] == 18.3
