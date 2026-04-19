import gc

from benchmarks.overhead import timing as timing_mod
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


def test_summarize_samples_normalizes_per_iteration():
    numpy_summary = summarize_samples([100, 120, 110], iterations=10)
    whest_summary = summarize_samples([50, 60, 55], iterations=5)

    assert numpy_summary.best_ns == 10
    assert numpy_summary.median_ns == 11
    assert whest_summary.best_ns == 10
    assert whest_summary.median_ns == 11


def test_calibrate_iterations_doubles_until_threshold(monkeypatch):
    calls = []
    elapsed_by_iterations = {1: 1_000, 2: 2_000, 4: 60_000}

    def fast_callable():
        calls.append("call")

    def fake_time_iterations(func, iterations):
        calls.append(iterations)
        func()
        return elapsed_by_iterations[iterations]

    monkeypatch.setattr(timing_mod, "_time_iterations", fake_time_iterations)

    iterations, elapsed_ns = calibrate_iterations(
        fast_callable,
        minimum_elapsed_ns=50_000,
        max_iterations=8,
    )

    assert iterations == 4
    assert elapsed_ns == 60_000
    assert calls == [1, "call", 2, "call", 4, "call"]


def test_calibrate_iterations_stops_at_max_iterations(monkeypatch):
    calls = []

    def fast_callable():
        calls.append("call")

    def fake_time_iterations(func, iterations):
        calls.append(iterations)
        func()
        return 1

    monkeypatch.setattr(timing_mod, "_time_iterations", fake_time_iterations)

    iterations, elapsed_ns = calibrate_iterations(
        fast_callable,
        minimum_elapsed_ns=50_000,
        max_iterations=4,
    )

    assert iterations == 4
    assert elapsed_ns == 1
    assert calls == [1, "call", 2, "call", 4, "call"]


def test_measure_samples_disables_and_restores_gc(monkeypatch):
    events = []
    state = {"enabled": True}
    phase = {"value": "warmup"}

    def measured():
        events.append(phase["value"])

    def disable():
        events.append("disabled")
        state["enabled"] = False

    def enable():
        events.append("enabled")
        state["enabled"] = True

    def fake_calibrate_iterations(func, minimum_elapsed_ns, max_iterations=1 << 20):
        phase["value"] = "calibrate"
        events.append(("calibrate", minimum_elapsed_ns, max_iterations))
        return 8, 123

    def fake_time_iterations(func, iterations):
        phase["value"] = "batch"
        events.append(("batch", iterations))
        for _ in range(iterations):
            func()
        return 456

    monkeypatch.setattr(gc, "disable", disable)
    monkeypatch.setattr(gc, "enable", enable)
    monkeypatch.setattr(gc, "isenabled", lambda: state["enabled"])
    monkeypatch.setattr(timing_mod, "calibrate_iterations", fake_calibrate_iterations)
    monkeypatch.setattr(timing_mod, "_time_iterations", fake_time_iterations)

    samples, iterations = measure_samples(
        measured,
        warmup_samples=2,
        measured_samples=3,
        minimum_elapsed_ns=1,
    )

    assert iterations == 8
    assert len(samples) == 3
    assert samples == [456, 456, 456]
    assert events[:3] == ["disabled", "warmup", "warmup"]
    assert events[3] == ("calibrate", 1, 1 << 20)
    batch_events = [event for event in events if isinstance(event, tuple) and event[0] == "batch"]
    measured_events = [event for event in events if event == "batch"]

    assert batch_events == [("batch", 8), ("batch", 8), ("batch", 8)]
    assert len(measured_events) == 24
    assert events[-1] == "enabled"
