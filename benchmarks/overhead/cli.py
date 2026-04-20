"""CLI entrypoints for the overhead benchmark workflow."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks._metadata import collect_environment_metadata
from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.artifacts import (
    _json_text,
    compare_runs,
    load_run,
    write_run_artifacts,
)
from benchmarks.overhead.discovery import classify_public_operations
from benchmarks.overhead.execution import run_case
from benchmarks.overhead.policy import evaluate_case, load_policy, suggest_thresholds
from benchmarks.overhead.report import render_terminal_summary, write_browser_report
from benchmarks.overhead.specs import (
    BenchmarkCase,
    documented_operations,
    full_cases,
    seed_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.overhead")
    parser.add_argument(
        "mode",
        choices=("ci", "full", "focus", "suggest-thresholds", "report"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/overhead"),
    )
    parser.add_argument("--family")
    parser.add_argument("--surface")
    parser.add_argument("--case-id")
    parser.add_argument("--op-name")
    parser.add_argument("--slug")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    return parser


def _select_cases(
    cases: Iterable[BenchmarkCase],
    *,
    family: str | None,
    surface: str | None,
    case_id: str | None,
    op_name: str | None,
    slug: str | None,
) -> list[BenchmarkCase]:
    selected = []
    for case in cases:
        if family and case.family != family:
            continue
        if surface and case.surface != surface:
            continue
        if case_id and case.case_id != case_id:
            continue
        if op_name and case.op_name != op_name:
            continue
        if slug and (case.slug or case.op_name) != slug:
            continue
        selected.append(case)
    return selected


def _cases_for_mode(mode: str) -> tuple[BenchmarkCase, ...]:
    if mode == "ci":
        return seed_cases()
    if mode == "full":
        return full_cases()
    if mode == "focus":
        deduped: dict[str, BenchmarkCase] = {}
        for case in (*seed_cases(), *full_cases()):
            deduped.setdefault(case.case_id, case)
        return tuple(deduped.values())
    raise ValueError(f"unsupported benchmark mode: {mode}")


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


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _build_accountability(
    ops_catalog: list[dict[str, Any]],
    discovered: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    documented_qualified = {
        str(entry["qualified_name"])
        for entry in ops_catalog
        if entry.get("qualified_name")
    }
    discovered_qualified = {
        str(entry["qualified_name"])
        for entry in discovered.get("inventory", [])
        if entry.get("qualified_name")
    }
    discovered_missing_in_docs = sorted(discovered_qualified - documented_qualified)
    documented_missing_in_discovery = sorted(
        documented_qualified - discovered_qualified
    )
    return {
        "documented_total": len(ops_catalog),
        "discovered_total": len(discovered.get("inventory", [])),
        "documented_missing_in_discovery": documented_missing_in_discovery,
        "discovered_missing_in_docs": discovered_missing_in_docs,
        # Compatibility field used by older terminal/report views; now means
        # runtime callables present in discovery but missing from the docs
        # catalog rather than "not covered by seed cases".
        "unclassified_operations": discovered_missing_in_docs,
    }


def _aggregate_operations(
    ops_catalog: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    benchmark_errors: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    benchmark_errors = benchmark_errors or {}
    cases_by_slug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_slug[str(case.get("slug") or case.get("op_name"))].append(case)

    rows: list[dict[str, Any]] = []
    for entry in ops_catalog:
        slug = str(entry["slug"])
        measured_cases = cases_by_slug.get(slug, [])
        ratios = [
            float(case["ratio"])
            for case in measured_cases
            if case.get("ratio") is not None
        ]
        startup_ratios = [
            float(case.get("startup", {}).get("ratio"))
            for case in measured_cases
            if case.get("startup", {}).get("ratio") is not None
        ]
        pass_count = sum(1 for case in measured_cases if case.get("passed", False))
        fail_count = len(measured_cases) - pass_count

        if measured_cases and benchmark_errors.get(slug):
            coverage_status = "partial_error"
        elif measured_cases:
            coverage_status = "measured"
        elif benchmark_errors.get(slug):
            coverage_status = "benchmark_error"
        elif entry.get("generation_status") == "generated":
            coverage_status = "not_in_run"
        else:
            coverage_status = str(entry.get("generation_status", "profile_missing"))

        rows.append(
            {
                "slug": slug,
                "name": entry["name"],
                "module": entry.get("module"),
                "area": entry.get("area"),
                "family": entry.get("family"),
                "category": entry.get("category"),
                "display_type": entry.get("display_type"),
                "qualified_name": entry.get("qualified_name"),
                "numpy_ref": entry.get("numpy_ref"),
                "whest_ref": entry.get("whest_ref"),
                "summary": entry.get("summary"),
                "notes": entry.get("notes"),
                "source_file": entry.get("source_file"),
                "coverage_status": coverage_status,
                "generation_status": entry.get("generation_status"),
                "measured_case_count": len(measured_cases),
                "expected_case_count": len(entry.get("generated_case_ids", [])),
                "benchmark_error_count": len(benchmark_errors.get(slug, [])),
                "pass_count": pass_count,
                "fail_count": fail_count,
                "representative_ratio": _median(ratios),
                "worst_ratio": max(ratios) if ratios else None,
                "representative_startup_ratio": _median(startup_ratios),
                "case_ids": [case["case_id"] for case in measured_cases]
                or list(entry.get("generated_case_ids", [])),
                "error_messages": benchmark_errors.get(slug, []),
            }
        )
    return rows


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

    if args.mode == "report":
        run = load_run(args.output_dir)
        comparison = (
            compare_runs(args.baseline_run, args.output_dir)
            if args.baseline_run is not None
            else None
        )
        report_path = write_browser_report(
            args.output_dir,
            run=run,
            comparison=comparison,
        )
        print(f"Wrote browser report to {report_path}")
        return 0

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
        _cases_for_mode(args.mode),
        family=args.family,
        surface=args.surface,
        case_id=args.case_id,
        op_name=args.op_name,
        slug=args.slug,
    )
    if not selected_cases:
        print("No benchmark cases matched the requested filters.")
        return 1

    ops_catalog = list(documented_operations())
    discovered = classify_public_operations()
    evaluated_cases: list[dict[str, Any]] = []
    benchmark_errors: dict[str, list[str]] = defaultdict(list)
    for case in selected_cases:
        try:
            evaluated_cases.append(
                evaluate_case(run_case(case, mode=args.mode), policy)
            )
        except Exception as exc:  # pragma: no cover - exercised via operation rows
            benchmark_errors[str(case.slug or case.op_name)].append(
                f"{case.case_id}: {exc}"
            )
    accountability = _build_accountability(ops_catalog, discovered)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "selected_cases": [case.case_id for case in selected_cases],
        "unclassified_operations": list(
            accountability.get("unclassified_operations", [])
        ),
    }
    operation_rows = _aggregate_operations(
        ops_catalog,
        evaluated_cases,
        benchmark_errors=benchmark_errors,
    )
    run = {
        "manifest": manifest,
        "environment": collect_environment_metadata(),
        "summary": _summary(evaluated_cases),
        "accountability": accountability,
        "cases": evaluated_cases,
        "operations": operation_rows,
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
    return (
        0
        if all(case.get("passed", False) for case in evaluated_cases)
        and not benchmark_errors
        else 1
    )
