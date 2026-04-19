import gc

from benchmarks.overhead.timing import (
    calibrate_iterations,
    measure_samples,
    summarize_samples,
)


def test_summarize_samples_computes_median_and_best():
    summary = summarize_samples([9, 5, 7])

    assert summary.best_ns == 5
    assert summary.median_ns == 7
    assert summary.sample_count == 3


def test_calibrate_iterations_respects_minimum_elapsed_ns():
    calls = []

    def fast_callable():
        calls.append(1)

    iterations, elapsed_ns = calibrate_iterations(
        fast_callable,
        minimum_elapsed_ns=50_000,
        max_iterations=1024,
    )

    assert iterations >= 1
    assert elapsed_ns >= 0
    assert calls


def test_measure_samples_disables_and_restores_gc(monkeypatch):
    observed = []
    state = {"enabled": True}

    def measured():
        observed.append(gc.isenabled())

    def disable():
        observed.append("disabled")
        state["enabled"] = False

    def enable():
        observed.append("enabled")
        state["enabled"] = True

    monkeypatch.setattr(gc, "disable", disable)
    monkeypatch.setattr(gc, "enable", enable)
    monkeypatch.setattr(gc, "isenabled", lambda: state["enabled"])

    samples, iterations = measure_samples(
        measured,
        warmup_samples=2,
        measured_samples=3,
        minimum_elapsed_ns=1,
    )

    assert iterations >= 1
    assert len(samples) == 3
    assert observed[0] == "disabled"
    assert observed[-1] == "enabled"
    assert all(value is False for value in observed if isinstance(value, bool))
