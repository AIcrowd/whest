from pathlib import Path

from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.cli import build_parser, main
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
            "--policy",
            "policy.json",
        ]
    )

    assert args.mode == "focus"
    assert args.output_dir == Path("artifacts/run")
    assert args.family == "pointwise"
    assert args.surface == "api"
    assert args.case_id == "add-api-tiny"
    assert args.policy == Path("policy.json")


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
        "unclassified_operations": ["whest.linalg.matmul"],
    }
    assert written["run"]["environment"] == {"software": {"python": "test"}}
    assert written["run"]["summary"] == {
        "case_count": 2,
        "passed": 1,
        "failed": 1,
        "worst_ratio": 7.5,
    }
    assert written["run"]["cases"][0]["passed"] is True
    assert written["run"]["cases"][1]["passed"] is False
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
