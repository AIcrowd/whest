"""Execution helpers for overhead benchmark cases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from benchmarks.overhead.specs import BenchmarkCase
from benchmarks.overhead.timing import (
    SampleSummary,
    measure_samples,
    summarize_samples,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STARTUP_BUDGET = int(1e15)
_STEADY_STATE_BUDGET = int(1e15)
_WARMUP_SAMPLES = 3
_CI_MEASURED_SAMPLES = 7
_DEFAULT_MEASURED_SAMPLES = 15
_MINIMUM_ELAPSED_NS = 200_000


def _materialize_operand(
    shape: tuple[int, ...], dtype: str, *, offset: int
) -> np.ndarray:
    size = int(np.prod(shape, dtype=int)) if shape else 1
    values = np.arange(size, dtype=np.float64).reshape(shape)
    return (values + offset + 1).astype(np.dtype(dtype), copy=False)


def _materialize_operands(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        _materialize_operand(shape, case.dtype, offset=index * 17)
        for index, shape in enumerate(case.operand_shapes)
    )


def _build_case_closures(
    case: BenchmarkCase, operands: tuple[object, object]
) -> tuple[callable, callable]:
    numpy_callable = lambda: case.numpy_factory(*operands)
    whest_callable = lambda: case.whest_factory(*operands)
    return numpy_callable, whest_callable


def _summary_to_dict(summary: SampleSummary, *, iterations: int) -> dict[str, float | int]:
    return {
        "best_ns": summary.best_ns,
        "median_ns": summary.median_ns,
        "sample_count": summary.sample_count,
        "iterations": iterations,
    }


def _summarize_budget(budget) -> dict[str, object]:
    summary = budget.summary_dict()
    return {
        "flops_used": budget.flops_used,
        "op_count": len(budget.op_log),
        "tracked_time_s": budget.total_tracked_time,
        "operations": summary["operations"],
    }


def _measure_whest_details(whest_callable) -> dict[str, object]:
    import whest as we

    with we.BudgetContext(flop_budget=_STEADY_STATE_BUDGET, quiet=True) as budget:
        whest_callable()
    return _summarize_budget(budget)


def _measured_samples_for_mode(mode: str) -> int:
    return _CI_MEASURED_SAMPLES if mode == "ci" else _DEFAULT_MEASURED_SAMPLES


def _steady_state_result(case: BenchmarkCase, *, mode: str) -> dict[str, object]:
    operands = _materialize_operands(case)
    numpy_callable, whest_callable = _build_case_closures(case, operands)
    measured_samples = _measured_samples_for_mode(mode)

    numpy_samples, numpy_iterations = measure_samples(
        numpy_callable,
        warmup_samples=_WARMUP_SAMPLES,
        measured_samples=measured_samples,
        minimum_elapsed_ns=_MINIMUM_ELAPSED_NS,
    )
    import whest as we

    with we.BudgetContext(flop_budget=_STEADY_STATE_BUDGET, quiet=True) as budget:
        whest_samples, whest_iterations = measure_samples(
            whest_callable,
            warmup_samples=_WARMUP_SAMPLES,
            measured_samples=measured_samples,
            minimum_elapsed_ns=_MINIMUM_ELAPSED_NS,
        )

    numpy_summary = summarize_samples(numpy_samples, iterations=numpy_iterations)
    whest_summary = summarize_samples(whest_samples, iterations=whest_iterations)
    numpy_median = numpy_summary.median_ns
    ratio = float("inf") if numpy_median == 0 else whest_summary.median_ns / numpy_median

    return {
        "numpy": _summary_to_dict(numpy_summary, iterations=numpy_iterations),
        "whest": _summary_to_dict(whest_summary, iterations=whest_iterations),
        "ratio": ratio,
        "whest_details": _summarize_budget(budget),
    }


def _case_payload(case: BenchmarkCase) -> dict[str, object]:
    return {
        "op_name": case.op_name,
        "surface": case.surface,
        "dtype": case.dtype,
        "operand_shapes": [list(shape) for shape in case.operand_shapes],
    }


def _startup_script() -> str:
    return textwrap.dedent(
        """
        import json
        import sys
        import time


        def materialize(shape, dtype, offset):
            import numpy as np

            size = int(np.prod(shape, dtype=int)) if shape else 1
            values = np.arange(size, dtype=np.float64).reshape(tuple(shape))
            return (values + offset + 1).astype(dtype, copy=False)


        def run_numpy(payload):
            import numpy as np

            a = materialize(payload["operand_shapes"][0], payload["dtype"], 0)
            b = materialize(payload["operand_shapes"][1], payload["dtype"], 17)
            op_name = payload["op_name"]
            surface = payload["surface"]
            if op_name == "add":
                return np.add(a, b) if surface == "api" else a + b
            if op_name == "matmul":
                return np.matmul(a, b) if surface == "api" else a @ b
            raise ValueError(f"unsupported startup case: {op_name}/{surface}")


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

            a = materialize(payload["operand_shapes"][0], payload["dtype"], 0)
            b = materialize(payload["operand_shapes"][1], payload["dtype"], 17)
            op_name = payload["op_name"]
            surface = payload["surface"]
            with we.BudgetContext(flop_budget=int(1e15), quiet=True) as budget:
                if op_name == "add":
                    if surface == "api":
                        we.add(a, b)
                    else:
                        we.array(a) + we.array(b)
                elif op_name == "matmul":
                    if surface == "api":
                        we.matmul(a, b)
                    else:
                        we.array(a) @ we.array(b)
                else:
                    raise ValueError(f"unsupported startup case: {op_name}/{surface}")
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
        "family": case.family,
        "surface": case.surface,
        "size_name": case.size_name,
        "dtype": case.dtype,
        "source_file": case.source_file,
        "mode": mode,
        "numpy": steady_state["numpy"],
        "whest": steady_state["whest"],
        "ratio": steady_state["ratio"],
        "whest_details": steady_state["whest_details"],
        "startup": _startup_result(case),
    }
