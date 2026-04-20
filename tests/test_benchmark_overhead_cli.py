import json
from pathlib import Path

from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.cli import (
    _aggregate_operations,
    _build_accountability,
    build_parser,
    main,
)
from benchmarks.overhead.specs import BenchmarkCase


def _case(case_id: str, *, family: str, surface: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        op_name=case_id.split("-")[0],
        qualified_name=f"whest.{case_id.split('-')[0]}" if surface == "api" else None,
        family=family,
        surface=surface,
        dtype="float64",
        size_name="tiny",
        startup_mode="warmup",
        source_file="src/whest/example.py",
        operand_shapes=((2,), (2,)),
        numpy_factory=lambda a, b: a,
        whest_factory=lambda a, b: b,
    )


def test_build_parser_accepts_required_modes_and_filters():
    parser = build_parser()

    args = parser.parse_args(
        [
            "focus",
            "--output-dir",
            "artifacts/run",
            "--family",
            "pointwise",
            "--surface",
            "api",
            "--case-id",
            "add-api-tiny",
            "--op-name",
            "add",
            "--slug",
            "add",
            "--policy",
            "policy.json",
        ]
    )

    assert args.mode == "focus"
    assert args.output_dir == Path("artifacts/run")
    assert args.family == "pointwise"
    assert args.surface == "api"
    assert args.case_id == "add-api-tiny"
    assert args.op_name == "add"
    assert args.slug == "add"
    assert args.policy == Path("policy.json")


def test_aggregate_operations_marks_partial_errors_explicitly():
    ops_catalog = [
        {
            "slug": "add",
            "name": "add",
            "family": "pointwise",
            "category": "counted_binary",
            "generation_status": "generated",
            "generated_case_ids": ["add-api-tiny", "add-api-medium"],
        }
    ]
    cases = [
        {
            "case_id": "add-api-tiny",
            "slug": "add",
            "ratio": 1.2,
            "passed": True,
            "startup": {"ratio": 1.1},
        }
    ]

    rows = _aggregate_operations(
        ops_catalog,
        cases,
        benchmark_errors={"add": ["add-api-medium: simulated failure"]},
    )

    assert rows == [
        {
            "slug": "add",
            "name": "add",
            "module": None,
            "area": None,
            "family": "pointwise",
            "category": "counted_binary",
            "display_type": None,
            "qualified_name": None,
            "numpy_ref": None,
            "whest_ref": None,
            "summary": None,
            "notes": None,
            "source_file": None,
            "coverage_status": "partial_error",
            "generation_status": "generated",
            "measured_case_count": 1,
            "expected_case_count": 2,
            "benchmark_error_count": 1,
            "pass_count": 1,
            "fail_count": 0,
            "representative_ratio": 1.2,
            "worst_ratio": 1.2,
            "representative_startup_ratio": 1.1,
            "case_ids": ["add-api-tiny"],
            "error_messages": ["add-api-medium: simulated failure"],
        }
    ]


def test_build_accountability_compat_field_tracks_discovery_only_callables():
    accountability = _build_accountability(
        [{"qualified_name": "whest.add"}],
        {"inventory": [{"qualified_name": "whest.add"}, {"qualified_name": "whest.linalg.matmul"}]},
    )

    assert accountability["discovered_missing_in_docs"] == ["whest.linalg.matmul"]
    assert accountability["unclassified_operations"] == ["whest.linalg.matmul"]


def test_build_parser_accepts_report_mode_and_baseline_run():
    parser = build_parser()

    args = parser.parse_args(
        [
            "report",
            "--output-dir",
            "artifacts/run",
            "--baseline-run",
            "artifacts/base",
        ]
    )

    assert args.mode == "report"
    assert args.output_dir == Path("artifacts/run")
    assert args.baseline_run == Path("artifacts/base")


def test_main_runs_selected_cases_writes_artifacts_and_returns_failure_code(
    tmp_path: Path, monkeypatch, capsys
):
    cases = [
        _case("add-api-tiny", family="pointwise", surface="api"),
        _case("add-operator-tiny", family="pointwise", surface="operator"),
        _case("matmul-api-tiny", family="contractions", surface="api"),
    ]
    run_calls = []
    written = {}

    monkeypatch.setattr("benchmarks.overhead.cli.seed_cases", lambda: tuple(cases))
    monkeypatch.setattr("benchmarks.overhead.cli.full_cases", lambda: ())
    monkeypatch.setattr("benchmarks.overhead.cli.documented_operations", lambda: [])
    monkeypatch.setattr(
        "benchmarks.overhead.cli.classify_public_operations",
        lambda: {
            "unclassified": [{"qualified_name": "whest.linalg.matmul"}],
            "inventory": [],
            "benchmarked": [],
            "excluded": [],
            "unsupported": [],
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.collect_environment_metadata",
        lambda: {"software": {"python": "test"}},
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_policy",
        lambda path=None: {
            "schema_version": SCHEMA_VERSION,
            "policy_version": "test",
            "default": {"ratio_max": 6.0},
        },
    )

    def fake_run_case(case, *, mode):
        run_calls.append((case.case_id, mode))
        return {
            "case_id": case.case_id,
            "op_name": case.op_name,
            "family": case.family,
            "surface": case.surface,
            "size_name": case.size_name,
            "dtype": case.dtype,
            "source_file": case.source_file,
            "mode": mode,
            "numpy": {"median_ns": 10, "best_ns": 9, "sample_count": 3, "iterations": 1},
            "whest": {"median_ns": 12, "best_ns": 11, "sample_count": 3, "iterations": 1},
            "ratio": 1.2 if case.case_id == "add-api-tiny" else 7.5,
            "whest_details": {"flops_used": 2, "op_count": 1, "tracked_time_s": 0.001},
            "startup": {
                "numpy": {"elapsed_ns": 100},
                "whest": {"elapsed_ns": 200},
                "ratio": 2.0,
                "whest_details": {"flops_used": 1, "op_count": 1, "tracked_time_s": 0.0001},
            },
        }

    monkeypatch.setattr("benchmarks.overhead.cli.run_case", fake_run_case)
    monkeypatch.setattr(
        "benchmarks.overhead.cli.evaluate_case",
        lambda case_result, policy: {
            **case_result,
            "threshold": 6.0,
            "policy_source": "default",
            "passed": case_result["ratio"] <= 6.0,
        },
    )

    def fake_write_run_artifacts(output_dir, run):
        written["output_dir"] = output_dir
        written["run"] = run
        return output_dir

    monkeypatch.setattr("benchmarks.overhead.cli.write_run_artifacts", fake_write_run_artifacts)
    monkeypatch.setattr(
        "benchmarks.overhead.cli.render_terminal_summary",
        lambda manifest, rendered_cases: "terminal summary",
    )

    exit_code = main(
        [
            "focus",
            "--output-dir",
            str(tmp_path),
            "--family",
            "pointwise",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert run_calls == [("add-api-tiny", "focus"), ("add-operator-tiny", "focus")]
    assert written["output_dir"] == tmp_path
    assert written["run"]["manifest"] == {
        "schema_version": SCHEMA_VERSION,
        "mode": "focus",
        "selected_cases": ["add-api-tiny", "add-operator-tiny"],
        "unclassified_operations": [],
    }
    assert written["run"]["environment"] == {"software": {"python": "test"}}
    assert written["run"]["summary"] == {
        "case_count": 2,
        "passed": 1,
        "failed": 1,
        "worst_ratio": 7.5,
    }
    assert written["run"]["accountability"]["documented_total"] == 0
    assert written["run"]["cases"][0]["passed"] is True
    assert written["run"]["cases"][1]["passed"] is False
    assert written["run"]["operations"] == []
    assert written["run"]["samples"] == [
        {"case_id": "add-api-tiny", "phase": "steady_state", "engine": "numpy", "median_ns": 10},
        {"case_id": "add-api-tiny", "phase": "steady_state", "engine": "whest", "median_ns": 12},
        {"case_id": "add-api-tiny", "phase": "startup", "engine": "numpy", "elapsed_ns": 100},
        {"case_id": "add-api-tiny", "phase": "startup", "engine": "whest", "elapsed_ns": 200},
        {"case_id": "add-operator-tiny", "phase": "steady_state", "engine": "numpy", "median_ns": 10},
        {"case_id": "add-operator-tiny", "phase": "steady_state", "engine": "whest", "median_ns": 12},
        {"case_id": "add-operator-tiny", "phase": "startup", "engine": "numpy", "elapsed_ns": 100},
        {"case_id": "add-operator-tiny", "phase": "startup", "engine": "whest", "elapsed_ns": 200},
    ]
    assert written["run"]["whest_details"] == [
        {
            "case_id": "add-api-tiny",
            "phase": "steady_state",
            "flops_used": 2,
            "op_count": 1,
            "tracked_time_s": 0.001,
        },
        {
            "case_id": "add-api-tiny",
            "phase": "startup",
            "flops_used": 1,
            "op_count": 1,
            "tracked_time_s": 0.0001,
        },
        {
            "case_id": "add-operator-tiny",
            "phase": "steady_state",
            "flops_used": 2,
            "op_count": 1,
            "tracked_time_s": 0.001,
        },
        {
            "case_id": "add-operator-tiny",
            "phase": "startup",
            "flops_used": 1,
            "op_count": 1,
            "tracked_time_s": 0.0001,
        },
    ]
    assert "terminal summary" in captured.out


def test_focus_mode_can_select_all_cases_for_one_operation(tmp_path: Path, monkeypatch, capsys):
    cases = [
        _case("add-api-tiny", family="pointwise", surface="api"),
        _case("add-api-medium", family="pointwise", surface="api"),
        _case("mean-api-tiny", family="reductions", surface="api"),
    ]

    monkeypatch.setattr("benchmarks.overhead.cli.seed_cases", lambda: ())
    monkeypatch.setattr("benchmarks.overhead.cli.full_cases", lambda: tuple(cases))
    monkeypatch.setattr(
        "benchmarks.overhead.cli.documented_operations",
        lambda: [
            {"slug": "add", "name": "add", "generation_status": "generated"},
            {"slug": "mean", "name": "mean", "generation_status": "generated"},
        ],
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.classify_public_operations",
        lambda: {
            "unclassified": [],
            "inventory": [],
            "benchmarked": [],
            "excluded": [],
            "unsupported": [],
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.collect_environment_metadata",
        lambda: {"software": {"python": "test"}},
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.run_case",
        lambda case, *, mode: {
            "case_id": case.case_id,
            "op_name": case.op_name,
            "slug": getattr(case, "slug", case.op_name),
            "family": case.family,
            "surface": case.surface,
            "size_name": case.size_name,
            "dtype": case.dtype,
            "source_file": case.source_file,
            "mode": mode,
            "numpy": {"median_ns": 10, "best_ns": 9, "sample_count": 3, "iterations": 1},
            "whest": {"median_ns": 12, "best_ns": 11, "sample_count": 3, "iterations": 1},
            "ratio": 1.2,
            "whest_details": {"flops_used": 2, "op_count": 1, "tracked_time_s": 0.001},
            "startup": {
                "numpy": {"elapsed_ns": 100},
                "whest": {"elapsed_ns": 200},
                "ratio": 2.0,
                "whest_details": {"flops_used": 1, "op_count": 1, "tracked_time_s": 0.0001},
            },
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.evaluate_case",
        lambda case_result, policy: {
            **case_result,
            "threshold": 6.0,
            "policy_source": "default",
            "passed": True,
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_policy",
        lambda path=None: {"schema_version": SCHEMA_VERSION, "default": {"ratio_max": 6.0}},
    )

    output_dir = tmp_path / "focus-op"
    exit_code = main(["focus", "--output-dir", str(output_dir), "--op-name", "add"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Mode: focus" in captured.out
    run = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run["selected_cases"] == ["add-api-tiny", "add-api-medium"]


def test_main_supports_suggest_thresholds_mode(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_policy",
        lambda path=None: {
            "schema_version": SCHEMA_VERSION,
            "policy_version": "baseline",
            "default": {"ratio_max": 6.0},
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_run",
        lambda output_dir: {
            "manifest": {"mode": "full"},
            "cases": [
                {"case_id": "add-api-tiny", "family": "pointwise", "surface": "api", "ratio": 1.5}
            ],
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.suggest_thresholds",
        lambda cases, policy: {
            "schema_version": SCHEMA_VERSION,
            "policy_version": "suggested",
            "default": {"ratio_max": 1.5},
        },
    )

    exit_code = main(["suggest-thresholds", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    suggested_path = tmp_path / "suggested_policy.json"

    assert exit_code == 0
    assert suggested_path.exists()
    assert "suggested_policy.json" in captured.out


def test_main_writes_json_safe_suggested_policy(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_policy",
        lambda path=None: {
            "schema_version": SCHEMA_VERSION,
            "policy_version": "baseline",
            "default": {"ratio_max": 6.0},
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_run",
        lambda output_dir: {
            "manifest": {"mode": "full"},
            "cases": [
                {"case_id": "add-api-tiny", "family": "pointwise", "surface": "api", "ratio": float("inf")}
            ],
        },
    )

    exit_code = main(["suggest-thresholds", "--output-dir", str(tmp_path)])
    suggested_path = tmp_path / "suggested_policy.json"
    suggested_text = suggested_path.read_text(encoding="utf-8")
    suggested = json.loads(suggested_text)

    assert exit_code == 0
    assert suggested_path.exists()
    assert "Infinity" not in suggested_text
    assert suggested["default"]["ratio_max"] == {"__whest_float__": "inf"}


def test_main_report_mode_loads_run_writes_browser_report_and_prints_path(
    tmp_path: Path, monkeypatch, capsys
):
    written = {}

    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_run",
        lambda output_dir: {
            "manifest": {"schema_version": SCHEMA_VERSION, "mode": "full", "selected_cases": []},
            "environment": {"software": {"python": "test"}},
            "summary": {"case_count": 0, "passed": 0, "failed": 0, "worst_ratio": 0.0},
            "cases": [],
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.write_browser_report",
        lambda output_dir, *, run, comparison=None: written.setdefault(
            "path", output_dir / "report.html"
        ),
    )

    exit_code = main(["report", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert written["path"] == tmp_path / "report.html"
    assert "report.html" in captured.out


def test_main_report_mode_builds_comparison_when_baseline_run_is_provided(
    tmp_path: Path, monkeypatch
):
    baseline_path = tmp_path / "baseline"
    observed = {}

    monkeypatch.setattr(
        "benchmarks.overhead.cli.load_run",
        lambda output_dir: {
            "manifest": {"schema_version": SCHEMA_VERSION, "mode": "full", "selected_cases": []},
            "environment": {},
            "summary": {"case_count": 0, "passed": 0, "failed": 0, "worst_ratio": 0.0},
            "cases": [],
        },
    )
    monkeypatch.setattr(
        "benchmarks.overhead.cli.compare_runs",
        lambda base_path, candidate_path: {
            "base": {"manifest": {"mode": "full"}},
            "candidate": {"manifest": {"mode": "full"}},
            "regressions": [{"case_id": "add-api-tiny", "ratio_delta": 1.0}],
            "improvements": [],
            "shared_case_count": 1,
        },
    )

    def fake_write_browser_report(output_dir, *, run, comparison=None):
        observed["comparison"] = comparison
        return output_dir / "report.html"

    monkeypatch.setattr("benchmarks.overhead.cli.write_browser_report", fake_write_browser_report)

    exit_code = main(
        [
            "report",
            "--output-dir",
            str(tmp_path),
            "--baseline-run",
            str(baseline_path),
        ]
    )

    assert exit_code == 0
    assert observed["comparison"]["shared_case_count"] == 1
