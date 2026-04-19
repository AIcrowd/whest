import json
import runpy
import sys
from pathlib import Path

import pytest

from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.artifacts import load_run
from benchmarks.overhead.cli import main
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


def test_cli_full_then_suggest_thresholds_round_trip(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "benchmarks.overhead.cli.seed_cases",
        lambda: (_case("add-api-tiny", family="pointwise", surface="api"),),
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
                "whest": {"elapsed_ns": 150},
                "ratio": 1.5,
                "whest_details": {"flops_used": 1, "op_count": 1, "tracked_time_s": 0.0001},
            },
        },
    )

    output_dir = tmp_path / "run"

    full_exit = main(["full", "--output-dir", str(output_dir)])
    suggest_exit = main(["suggest-thresholds", "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    run = load_run(output_dir)
    suggested = json.loads((output_dir / "suggested_policy.json").read_text(encoding="utf-8"))

    assert full_exit == 0
    assert suggest_exit == 0
    assert run["manifest"] == {
        "schema_version": SCHEMA_VERSION,
        "mode": "full",
        "selected_cases": ["add-api-tiny"],
        "unclassified_operations": [],
    }
    assert run["summary"]["case_count"] == 1
    assert run["cases"][0]["case_id"] == "add-api-tiny"
    assert suggested["schema_version"] == SCHEMA_VERSION
    assert suggested["policy_version"] == "suggested"
    assert "All cases passed." in captured.out


def test_focus_mode_writes_filtered_artifacts(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "benchmarks.overhead.cli.seed_cases",
        lambda: (
            _case("add-api-tiny", family="pointwise", surface="api"),
            _case("matmul-api-tiny", family="contractions", surface="api"),
        ),
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
                "whest": {"elapsed_ns": 150},
                "ratio": 1.5,
                "whest_details": {"flops_used": 1, "op_count": 1, "tracked_time_s": 0.0001},
            },
        },
    )

    output_dir = tmp_path / "focus-run"
    exit_code = main(
        [
            "focus",
            "--output-dir",
            str(output_dir),
            "--case-id",
            "matmul-api-tiny",
        ]
    )

    captured = capsys.readouterr()
    run = load_run(output_dir)

    assert exit_code == 0
    assert run["manifest"] == {
        "schema_version": SCHEMA_VERSION,
        "mode": "focus",
        "selected_cases": ["matmul-api-tiny"],
        "unclassified_operations": [],
    }
    assert run["summary"]["case_count"] == 1
    assert [case["case_id"] for case in run["cases"]] == ["matmul-api-tiny"]
    assert {row["case_id"] for row in run["samples"]} == {"matmul-api-tiny"}
    assert {row["case_id"] for row in run["whest_details"]} == {"matmul-api-tiny"}
    assert "Mode: focus" in captured.out


def test_focus_mode_returns_failure_when_filters_match_no_cases(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "benchmarks.overhead.cli.seed_cases",
        lambda: (_case("add-api-tiny", family="pointwise", surface="api"),),
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

    output_dir = tmp_path / "missing-focus"
    exit_code = main(
        [
            "focus",
            "--output-dir",
            str(output_dir),
            "--case-id",
            "missing-case",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No benchmark cases matched the requested filters." in captured.out
    assert not (output_dir / "manifest.json").exists()


def test_module_entrypoint_delegates_to_cli_main(monkeypatch):
    called = {}

    def fake_main(argv=None):
        called["argv"] = argv
        return 0

    monkeypatch.setattr("benchmarks.overhead.cli.main", fake_main)
    monkeypatch.setattr(sys, "argv", ["python", "-m", "benchmarks.overhead", "ci"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("benchmarks.overhead", run_name="__main__")

    assert excinfo.value.code == 0
    assert called["argv"] is None
