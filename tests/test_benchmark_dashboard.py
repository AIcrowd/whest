"""Tests for benchmark dashboard."""

from benchmarks.dashboard import render_terminal, render_html


def test_render_terminal_produces_string():
    meta = {
        "timestamp": "2026-04-06T00:00:00Z",
        "hardware": {"cpu_model": "Test CPU", "cpu_cores": 4, "arch": "x86_64", "ram_gb": 16},
        "software": {"os": "Linux 5.15", "python": "3.11.7", "numpy": "2.1.3", "blas": "OpenBLAS"},
        "benchmark_config": {"dtype": "float64", "repeats": 10, "distributions": 3},
    }
    weights = {"add": 1.0, "exp": 18.3, "linalg.cholesky": 1.15}
    baseline_fpe = 1.0
    output = render_terminal(meta, weights, baseline_fpe, total_ops=3, duration_seconds=60)
    assert isinstance(output, str)
    assert "exp" in output
    assert "18.3" in output or "18.30" in output


def test_render_html_produces_valid_html():
    meta = {
        "timestamp": "2026-04-06T00:00:00Z",
        "hardware": {"cpu_model": "Test CPU", "cpu_cores": 4, "arch": "x86_64", "ram_gb": 16},
        "software": {"os": "Linux 5.15", "python": "3.11.7", "numpy": "2.1.3", "blas": "OpenBLAS"},
        "benchmark_config": {"dtype": "float64", "repeats": 10, "distributions": 3},
    }
    weights = {"add": 1.0, "exp": 18.3}
    baseline_fpe = 1.0
    html = render_html(meta, weights, baseline_fpe, total_ops=2, duration_seconds=60)
    assert "<html" in html.lower()
    assert "exp" in html
    assert "18.3" in html or "18.30" in html
