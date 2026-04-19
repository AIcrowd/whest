import json

import numpy as np
import pytest

from benchmarks.overhead.specs import BenchmarkCase, seed_cases


def _case(
    *,
    numpy_factory,
    whest_factory,
    operand_shapes=((2,), (2,)),
    dtype="float64",
):
    return BenchmarkCase(
        case_id="case-add-api-tiny",
        op_name="add",
        qualified_name="whest.add",
        family="pointwise",
        surface="api",
        dtype=dtype,
        size_name="tiny",
        startup_mode="warmup",
        source_file="src/whest/_pointwise.py",
        operand_shapes=operand_shapes,
        numpy_factory=numpy_factory,
        whest_factory=whest_factory,
    )


def test_run_case_returns_top_level_steady_state_fields(monkeypatch):
    from benchmarks.overhead import execution as execution_mod

    summarize_calls = []
    sample_calls = []

    case = _case(
        numpy_factory=lambda a, b: np.add(a, b),
        whest_factory=lambda a, b: np.add(a, b),
    )

    def fake_measure_samples(
        func, warmup_samples, measured_samples, minimum_elapsed_ns
    ):
        result = func()
        sample_calls.append(
            {
                "warmup": warmup_samples,
                "measured": measured_samples,
                "minimum_elapsed_ns": minimum_elapsed_ns,
                "shape": tuple(result.shape),
            }
        )
        if len(sample_calls) == 1:
            return [100, 120, 110], 10
        return [200, 220, 210], 20

    def fake_summarize_samples(samples, *, iterations):
        summarize_calls.append((tuple(samples), iterations))
        return execution_mod.SampleSummary(
            best_ns=min(samples) / iterations,
            median_ns=sorted(samples)[len(samples) // 2] / iterations,
            sample_count=len(samples),
        )

    monkeypatch.setattr(execution_mod, "measure_samples", fake_measure_samples)
    monkeypatch.setattr(execution_mod, "summarize_samples", fake_summarize_samples)
    monkeypatch.setattr(
        execution_mod,
        "_measure_whest_details",
        lambda whest_callable: (_ for _ in ()).throw(
            AssertionError("steady-state details must come from the measured session")
        ),
    )
    monkeypatch.setattr(
        execution_mod,
        "_startup_result",
        lambda startup_case: {
            "numpy": {"elapsed_ns": 25},
            "whest": {"elapsed_ns": 50},
            "ratio": 2.0,
            "whest_details": {
                "flops_used": 2,
                "op_count": 1,
                "tracked_time_s": 0.01,
                "operations": {"add": {"calls": 1}},
            },
        },
    )

    result = execution_mod.run_case(case, mode="timing")

    assert summarize_calls == [
        ((100, 120, 110), 10),
        ((200, 220, 210), 20),
    ]
    assert len(sample_calls) == 2
    assert result["case_id"] == case.case_id
    assert result["op_name"] == case.op_name
    assert result["family"] == case.family
    assert result["surface"] == case.surface
    assert result["size_name"] == case.size_name
    assert result["dtype"] == case.dtype
    assert result["source_file"] == case.source_file
    assert result["mode"] == "timing"
    assert result["numpy"]["median_ns"] == 11
    assert result["whest"]["median_ns"] == 10.5
    assert result["ratio"] == 10.5 / 11
    assert result["whest_details"]["flops_used"] >= 0
    assert result["whest_details"]["op_count"] >= 0
    assert "tracked_time_s" in result["whest_details"]
    assert "operations" in result["whest_details"]
    assert result["startup"] == {
        "numpy": {"elapsed_ns": 25},
        "whest": {"elapsed_ns": 50},
        "ratio": 2.0,
        "whest_details": {
            "flops_used": 2,
            "op_count": 1,
            "tracked_time_s": 0.01,
            "operations": {"add": {"calls": 1}},
        },
    }

    assert sample_calls == [
        {
            "warmup": 3,
            "measured": 15,
            "minimum_elapsed_ns": 200_000,
            "shape": (2,),
        },
        {
            "warmup": 3,
            "measured": 15,
            "minimum_elapsed_ns": 200_000,
            "shape": (2,),
        },
    ]


@pytest.mark.parametrize(
    ("mode", "expected_measured_samples"),
    [("ci", 7), ("timing", 15)],
)
def test_run_case_uses_mode_to_choose_measured_sample_count(
    monkeypatch, mode, expected_measured_samples
):
    from benchmarks.overhead import execution as execution_mod

    measured_samples_seen = []
    case = _case(
        numpy_factory=lambda a, b: np.add(a, b),
        whest_factory=lambda a, b: np.add(a, b),
    )

    def fake_measure_samples(
        func, warmup_samples, measured_samples, minimum_elapsed_ns
    ):
        func()
        measured_samples_seen.append(measured_samples)
        return [100, 110, 120], 10

    monkeypatch.setattr(execution_mod, "measure_samples", fake_measure_samples)
    monkeypatch.setattr(
        execution_mod,
        "_measure_whest_details",
        lambda whest_callable: (_ for _ in ()).throw(
            AssertionError("steady-state details must come from the measured session")
        ),
    )
    monkeypatch.setattr(
        execution_mod,
        "_startup_result",
        lambda startup_case: {
            "numpy": {"elapsed_ns": 1},
            "whest": {"elapsed_ns": 2},
            "ratio": 2.0,
            "whest_details": {
                "flops_used": 0,
                "op_count": 0,
                "tracked_time_s": 0.0,
                "operations": {},
            },
        },
    )

    execution_mod.run_case(case, mode=mode)

    assert measured_samples_seen == [
        expected_measured_samples,
        expected_measured_samples,
    ]


def test_startup_result_reports_numpy_and_whest_timings():
    from benchmarks.overhead.execution import _startup_result

    case = seed_cases()[0]

    result = _startup_result(case)

    assert set(result) >= {"numpy", "whest", "ratio"}
    assert result["numpy"]["elapsed_ns"] > 0
    assert result["whest"]["elapsed_ns"] > 0
    assert result["ratio"] == result["whest"]["elapsed_ns"] / result["numpy"]["elapsed_ns"]
    assert set(result["whest_details"]) == {
        "flops_used",
        "op_count",
        "tracked_time_s",
        "operations",
    }
    json.dumps(result)
