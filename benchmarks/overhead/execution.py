"""Execution helpers for overhead benchmark cases."""

from __future__ import annotations

import gc
import importlib
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np

from benchmarks.overhead.profiles import materialize_case_inputs
from benchmarks.overhead.specs import BenchmarkCase
from benchmarks.overhead.timing import (
    SampleSummary,
    calibrate_iterations,
    measure_samples,
    summarize_samples,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STARTUP_BUDGET = int(1e15)
_STEADY_STATE_BUDGET = int(1e15)
_WARMUP_SAMPLES = 3
_CI_MEASURED_SAMPLES = 7
_FULL_MEASURED_SAMPLES = 5
_FOCUS_MEASURED_SAMPLES = 15
_MINIMUM_ELAPSED_NS = 200_000


def _materialize_operand(
    shape: tuple[int, ...], dtype: str, *, offset: int
) -> np.ndarray:
    size = int(np.prod(shape, dtype=int)) if shape else 1
    values = np.arange(size, dtype=np.float64)
    values = ((values + (offset * 7)) % 31 - 15) / 16.0
    return values.reshape(shape).astype(np.dtype(dtype), copy=False)


def _materialize_operands(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        _materialize_operand(shape, case.dtype, offset=index * 17)
        for index, shape in enumerate(case.operand_shapes)
    )


def _resolve_factory(factory: Callable[..., object] | str) -> Callable[..., object]:
    if callable(factory):
        return factory
    module_name, _, qualname = factory.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"factory must be a callable or import path: {factory!r}")
    obj: Any = importlib.import_module(module_name)
    for segment in qualname.split("."):
        obj = getattr(obj, segment)
    if not callable(obj):
        raise TypeError(f"resolved factory is not callable: {factory!r}")
    return obj


def _case_inputs(case: BenchmarkCase) -> tuple[tuple[object, ...], dict[str, object]]:
    if case.profile_kind:
        return materialize_case_inputs(
            {
                "op_name": case.op_name,
                "size_name": case.size_name,
                "dtype": case.dtype,
                "profile_kind": case.profile_kind,
                "profile_params": case.profile_params,
            }
        )
    return _materialize_operands(case), {}


def _build_case_closures(
    case: BenchmarkCase,
    args_and_kwargs: tuple[tuple[object, ...], dict[str, object]] | tuple[object, ...] | None = None,
) -> tuple[Callable[[], object], Callable[[], object]]:
    if args_and_kwargs is None:
        args, kwargs = _case_inputs(case)
    elif (
        isinstance(args_and_kwargs, tuple)
        and len(args_and_kwargs) == 2
        and isinstance(args_and_kwargs[1], dict)
    ):
        args, kwargs = args_and_kwargs
    else:
        args = tuple(args_and_kwargs)
        kwargs = {}
    numpy_factory = _resolve_factory(case.numpy_factory)
    whest_factory = _resolve_factory(case.whest_factory)

    def numpy_callable() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            with np.errstate(all="ignore"):
                return numpy_factory(*args, **kwargs)

    def whest_callable() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            with np.errstate(all="ignore"):
                return whest_factory(*args, **kwargs)

    return numpy_callable, whest_callable


def _summary_to_dict(summary: SampleSummary, *, iterations: int) -> dict[str, float | int]:
    return {
        "best_ns": summary.best_ns,
        "median_ns": summary.median_ns,
        "sample_count": summary.sample_count,
        "iterations": iterations,
    }


def _summarize_budget(budget) -> dict[str, object]:
    return _summarize_operation_records(
        budget.op_log,
        flops_used=budget.flops_used,
        tracked_time_s=budget.total_tracked_time,
    )


def _summarize_operation_records(
    op_log: list[object], *, flops_used: int, tracked_time_s: float
) -> dict[str, object]:
    operations: dict[str, dict[str, float | int]] = {}
    for record in op_log:
        bucket = operations.setdefault(
            record.op_name,
            {"flop_cost": 0, "calls": 0, "duration": 0.0},
        )
        bucket["flop_cost"] += record.flop_cost
        bucket["calls"] += 1
        if record.duration is not None:
            bucket["duration"] += record.duration
    return {
        "flops_used": flops_used,
        "op_count": len(op_log),
        "tracked_time_s": tracked_time_s,
        "operations": operations,
    }


def _measured_samples_for_mode(mode: str) -> int:
    if mode == "ci":
        return _CI_MEASURED_SAMPLES
    if mode in {"focus", "timing"}:
        return _FOCUS_MEASURED_SAMPLES
    return _FULL_MEASURED_SAMPLES


def _time_iterations(func: Callable[[], object], iterations: int) -> int:
    started_ns = perf_counter_ns()
    for _ in range(iterations):
        func()
    return perf_counter_ns() - started_ns


def _measure_whest_samples(
    whest_callable: Callable[[], object], *, measured_samples: int
) -> tuple[list[int], int, dict[str, object]]:
    import whest as we

    we.budget_reset()
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for _ in range(_WARMUP_SAMPLES):
            whest_callable()

        iterations, _ = calibrate_iterations(
            whest_callable,
            minimum_elapsed_ns=_MINIMUM_ELAPSED_NS,
        )
        with we.BudgetContext(flop_budget=_STEADY_STATE_BUDGET, quiet=True) as budget:
            samples = [
                _time_iterations(whest_callable, iterations)
                for _ in range(measured_samples)
            ]
        return samples, iterations, _summarize_budget(budget)
    finally:
        if was_enabled:
            gc.enable()
        we.budget_reset()


def _steady_state_result(case: BenchmarkCase, *, mode: str) -> dict[str, object]:
    numpy_callable, whest_callable = _build_case_closures(case)
    measured_samples = _measured_samples_for_mode(mode)

    numpy_samples, numpy_iterations = measure_samples(
        numpy_callable,
        warmup_samples=_WARMUP_SAMPLES,
        measured_samples=measured_samples,
        minimum_elapsed_ns=_MINIMUM_ELAPSED_NS,
    )
    whest_samples, whest_iterations, whest_details = _measure_whest_samples(
        whest_callable,
        measured_samples=measured_samples,
    )

    numpy_summary = summarize_samples(numpy_samples, iterations=numpy_iterations)
    whest_summary = summarize_samples(whest_samples, iterations=whest_iterations)
    numpy_median = numpy_summary.median_ns
    ratio = float("inf") if numpy_median == 0 else whest_summary.median_ns / numpy_median

    return {
        "numpy": _summary_to_dict(numpy_summary, iterations=numpy_iterations),
        "whest": _summary_to_dict(whest_summary, iterations=whest_iterations),
        "ratio": ratio,
        "whest_details": whest_details,
    }


def _callable_payload(func: Callable[..., object] | str) -> dict[str, str]:
    if isinstance(func, str):
        module, _, qualname = func.rpartition(".")
        if not module or not qualname:
            raise ValueError(f"factory path must be importable: {func!r}")
        return {"module": module, "qualname": qualname}
    qualname = getattr(func, "__qualname__", "")
    module = getattr(func, "__module__", "")
    if not module or not qualname or "<locals>" in qualname:
        raise ValueError(f"factory must be importable: {func!r}")
    return {"module": module, "qualname": qualname}


def _numpy_startup_payload(func: Callable[..., object]) -> dict[str, str]:
    payload = _callable_payload(func)
    if (
        payload["module"] == "benchmarks.overhead.specs"
        and payload["qualname"].startswith("_numpy_")
    ):
        payload["module"] = "benchmarks.overhead.startup_numpy"
    return payload


def _case_payload(case: BenchmarkCase) -> dict[str, object]:
    return {
        "op_name": case.op_name,
        "size_name": case.size_name,
        "dtype": case.dtype,
        "operand_shapes": [list(shape) for shape in case.operand_shapes],
        "profile_kind": case.profile_kind,
        "profile_params": case.profile_params,
        "numpy_factory": _numpy_startup_payload(case.numpy_factory),
        "whest_factory": _callable_payload(case.whest_factory),
    }


def _startup_script() -> str:
    return textwrap.dedent(
        """
        import importlib
        import json
        import sys
        import time

        from benchmarks.overhead.profiles import materialize_case_inputs

        def materialize(shape, dtype, offset):
            import numpy as np

            size = int(np.prod(shape, dtype=int)) if shape else 1
            values = np.arange(size, dtype=np.float64)
            values = ((values + (offset * 7)) % 31 - 15) / 16.0
            return values.reshape(tuple(shape)).astype(dtype, copy=False)


        def resolve_callable(spec):
            obj = importlib.import_module(spec["module"])
            for segment in spec["qualname"].split("."):
                obj = getattr(obj, segment)
            return obj


        def build_inputs(case):
            if case.get("profile_kind"):
                return materialize_case_inputs(case)
            a = materialize(case["operand_shapes"][0], case["dtype"], 0)
            b = materialize(case["operand_shapes"][1], case["dtype"], 17)
            return (a, b), {}


        def run_numpy(payload):
            import numpy as np

            args, kwargs = build_inputs(payload)
            numpy_factory = resolve_callable(payload["numpy_factory"])
            with np.errstate(all="ignore"):
                return numpy_factory(*args, **kwargs)


        def summarize_budget(budget):
            summary = budget.summary_dict()
            return {
                "flops_used": budget.flops_used,
                "op_count": len(budget.op_log),
                "tracked_time_s": budget.total_tracked_time,
                "operations": summary["operations"],
            }


        def run_whest(payload):
            import numpy as np
            import whest as we

            args, kwargs = build_inputs(payload)
            whest_factory = resolve_callable(payload["whest_factory"])
            with we.BudgetContext(flop_budget=int(1e15), quiet=True) as budget:
                with np.errstate(all="ignore"):
                    whest_factory(*args, **kwargs)
            return summarize_budget(budget)


        payload = json.loads(sys.stdin.read())
        engine = payload["engine"]
        case = payload["case"]

        started_ns = time.perf_counter_ns()
        if engine == "numpy":
            run_numpy(case)
            result = {"elapsed_ns": time.perf_counter_ns() - started_ns}
        elif engine == "whest":
            details = run_whest(case)
            result = {
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "whest_details": details,
            }
        else:
            raise ValueError(f"unsupported engine: {engine}")

        sys.stdout.write(json.dumps(result))
        """
    )


def _startup_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(_REPO_ROOT), str(_REPO_ROOT / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _run_startup_subprocess(case: BenchmarkCase, *, engine: str) -> dict[str, object]:
    payload = {"engine": engine, "case": _case_payload(case)}
    completed = subprocess.run(
        [sys.executable, "-c", _startup_script()],
        check=True,
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_startup_env(),
        input=json.dumps(payload),
        text=True,
    )
    return json.loads(completed.stdout)


def _startup_result(case: BenchmarkCase) -> dict[str, object]:
    numpy_result = _run_startup_subprocess(case, engine="numpy")
    whest_result = _run_startup_subprocess(case, engine="whest")
    numpy_elapsed = numpy_result["elapsed_ns"]
    ratio = float("inf") if numpy_elapsed == 0 else whest_result["elapsed_ns"] / numpy_elapsed
    return {
        "numpy": {"elapsed_ns": numpy_elapsed},
        "whest": {"elapsed_ns": whest_result["elapsed_ns"]},
        "ratio": ratio,
        "whest_details": whest_result["whest_details"],
    }


def run_case(case: BenchmarkCase, *, mode: str) -> dict[str, object]:
    steady_state = _steady_state_result(case, mode=mode)
    return {
        "case_id": case.case_id,
        "op_name": case.op_name,
        "slug": case.slug or case.op_name,
        "family": case.family,
        "surface": case.surface,
        "size_name": case.size_name,
        "dtype": case.dtype,
        "source_file": case.source_file,
        "category": case.category,
        "area": case.area,
        "mode": mode,
        "numpy": steady_state["numpy"],
        "whest": steady_state["whest"],
        "ratio": steady_state["ratio"],
        "whest_details": steady_state["whest_details"],
        "startup": _startup_result(case),
    }
