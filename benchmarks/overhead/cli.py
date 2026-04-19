"""CLI entrypoints for the overhead benchmark workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from benchmarks._metadata import collect_environment_metadata
from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.artifacts import _json_text, load_run, write_run_artifacts
from benchmarks.overhead.discovery import classify_public_operations
from benchmarks.overhead.execution import run_case
from benchmarks.overhead.policy import evaluate_case, load_policy, suggest_thresholds
from benchmarks.overhead.report import render_terminal_summary
from benchmarks.overhead.specs import BenchmarkCase, seed_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.overhead")
    parser.add_argument(
        "mode",
        choices=("ci", "full", "focus", "suggest-thresholds"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/overhead"),
    )
    parser.add_argument("--family")
    parser.add_argument("--surface")
    parser.add_argument("--case-id")
    parser.add_argument("--policy", type=Path)
    return parser


def _select_cases(
    cases: Iterable[BenchmarkCase],
    *,
    family: str | None,
    surface: str | None,
    case_id: str | None,
) -> list[BenchmarkCase]:
    selected = []
    for case in cases:
        if family and case.family != family:
            continue
        if surface and case.surface != surface:
            continue
        if case_id and case.case_id != case_id:
            continue
        selected.append(case)
    return selected


def _sample_rows(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for engine in ("numpy", "whest"):
        rows.append(
            {
                "case_id": case_result["case_id"],
                "phase": "steady_state",
                "engine": engine,
                "median_ns": case_result[engine]["median_ns"],
            }
        )
    for engine in ("numpy", "whest"):
        rows.append(
            {
                "case_id": case_result["case_id"],
                "phase": "startup",
                "engine": engine,
                "elapsed_ns": case_result["startup"][engine]["elapsed_ns"],
            }
        )
    return rows


def _whest_detail_rows(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for phase, details in (
        ("steady_state", case_result.get("whest_details", {})),
        ("startup", case_result.get("startup", {}).get("whest_details", {})),
    ):
        rows.append(
            {
                "case_id": case_result["case_id"],
                "phase": phase,
                "flops_used": details.get("flops_used", 0),
                "op_count": details.get("op_count", 0),
                "tracked_time_s": details.get("tracked_time_s", 0.0),
            }
        )
    return rows


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(case.get("ratio", 0.0)) for case in cases]
    failed = sum(1 for case in cases if not case.get("passed", False))
    return {
        "case_count": len(cases),
        "passed": len(cases) - failed,
        "failed": failed,
        "worst_ratio": max(ratios) if ratios else 0.0,
    }


def _write_suggested_policy(
    output_dir: Path, *, cases: list[dict[str, Any]], policy: dict[str, Any]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suggested_path = output_dir / "suggested_policy.json"
    suggested_path.write_text(
        _json_text(suggest_thresholds(cases, policy)),
        encoding="utf-8",
    )
    return suggested_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = load_policy(args.policy)

    if args.mode == "suggest-thresholds":
        prior_run = load_run(args.output_dir)
        suggested_path = _write_suggested_policy(
            args.output_dir,
            cases=list(prior_run.get("cases", [])),
            policy=policy,
        )
        print(f"Wrote suggested policy to {suggested_path}")
        return 0

    selected_cases = _select_cases(
        seed_cases(),
        family=args.family,
        surface=args.surface,
        case_id=args.case_id,
    )
    if not selected_cases:
        print("No benchmark cases matched the requested filters.")
        return 1

    discovered = classify_public_operations()
    evaluated_cases = [
        evaluate_case(run_case(case, mode=args.mode), policy) for case in selected_cases
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "selected_cases": [case.case_id for case in selected_cases],
        "unclassified_operations": [
            entry["qualified_name"] for entry in discovered.get("unclassified", [])
        ],
    }
    run = {
        "manifest": manifest,
        "environment": collect_environment_metadata(),
        "summary": _summary(evaluated_cases),
        "cases": evaluated_cases,
        "samples": [
            row for case_result in evaluated_cases for row in _sample_rows(case_result)
        ],
        "whest_details": [
            row
            for case_result in evaluated_cases
            for row in _whest_detail_rows(case_result)
        ],
    }
    write_run_artifacts(args.output_dir, run)
    print(render_terminal_summary(manifest, evaluated_cases))
    return 0 if all(case.get("passed", False) for case in evaluated_cases) else 1
