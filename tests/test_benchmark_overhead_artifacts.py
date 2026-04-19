from pathlib import Path

import pytest

from benchmarks.overhead.artifacts import compare_runs, load_run, write_run_artifacts
from benchmarks.overhead import SCHEMA_VERSION


def test_write_run_artifacts_creates_required_files(tmp_path: Path):
    run = {
        "manifest": {"schema_version": 1},
        "environment": {"software": {"numpy": "2.x"}},
        "summary": {"case_count": 1},
        "cases": [{"case_id": "pointwise:add:api:tiny:float64"}],
        "samples": [
            {"case_id": "pointwise:add:api:tiny:float64", "phase": "steady_state"}
        ],
        "whest_details": [
            {"case_id": "pointwise:add:api:tiny:float64", "flops_used": 16}
        ],
    }

    output_dir = write_run_artifacts(tmp_path, run)

    assert output_dir == tmp_path
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "environment.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "cases.jsonl").exists()
    assert (output_dir / "samples.jsonl").exists()
    assert (output_dir / "whest_details.jsonl").exists()


def test_load_run_and_compare_runs_round_trip(tmp_path: Path):
    base = {
        "manifest": {"schema_version": 1},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-a", "ratio": 2.0}],
        "samples": [],
        "whest_details": [],
    }
    candidate = {
        "manifest": {"schema_version": 1},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-a", "ratio": 3.0}],
        "samples": [],
        "whest_details": [],
    }

    base_dir = write_run_artifacts(tmp_path / "base", base)
    candidate_dir = write_run_artifacts(tmp_path / "candidate", candidate)

    loaded = load_run(base_dir)
    diff = compare_runs(base_dir, candidate_dir)

    assert loaded["cases"][0]["case_id"] == "case-a"
    assert loaded["cases"][0]["ratio"] == 2.0
    assert diff["regressions"][0]["case_id"] == "case-a"
    assert diff["regressions"][0]["base_ratio"] == 2.0
    assert diff["regressions"][0]["candidate_ratio"] == 3.0
    assert diff["regressions"][0]["ratio_delta"] == 1.0


def test_write_run_artifacts_injects_schema_version(tmp_path: Path):
    run = {
        "manifest": {},
        "environment": {},
        "summary": {},
        "cases": [],
        "samples": [],
        "whest_details": [],
    }

    output_dir = write_run_artifacts(tmp_path, run)

    assert load_run(output_dir)["manifest"]["schema_version"] == SCHEMA_VERSION


def test_write_run_artifacts_preserves_non_finite_ratios(tmp_path: Path):
    run = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-inf", "ratio": float("inf")}],
        "samples": [],
        "whest_details": [],
    }

    output_dir = write_run_artifacts(tmp_path, run)
    loaded = load_run(output_dir)

    assert loaded["cases"][0]["ratio"] == float("inf")
    assert "Infinity" not in (output_dir / "cases.jsonl").read_text(encoding="utf-8")


def test_write_run_artifacts_rejects_schema_version_mismatch(tmp_path: Path):
    run = {
        "manifest": {"schema_version": SCHEMA_VERSION + 1},
        "environment": {},
        "summary": {},
        "cases": [],
        "samples": [],
        "whest_details": [],
    }

    with pytest.raises(ValueError, match="schema_version"):
        write_run_artifacts(tmp_path, run)


def test_load_run_rejects_missing_required_artifact(tmp_path: Path):
    run = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [],
        "samples": [],
        "whest_details": [],
    }
    output_dir = write_run_artifacts(tmp_path, run)
    (output_dir / "samples.jsonl").unlink()

    with pytest.raises(FileNotFoundError):
        load_run(output_dir)


def test_load_run_rejects_schema_version_mismatch(tmp_path: Path):
    run = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [],
        "samples": [],
        "whest_details": [],
    }
    output_dir = write_run_artifacts(tmp_path, run)
    (output_dir / "manifest.json").write_text(
        '{"schema_version": %d}\n' % (SCHEMA_VERSION + 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_run(output_dir)


def test_compare_runs_rejects_schema_mismatch(tmp_path: Path):
    base = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-a", "ratio": 2.0}],
        "samples": [],
        "whest_details": [],
    }
    candidate = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-a", "ratio": 3.0}],
        "samples": [],
        "whest_details": [],
    }

    base_dir = write_run_artifacts(tmp_path / "base", base)
    candidate_dir = write_run_artifacts(tmp_path / "candidate", candidate)
    (candidate_dir / "manifest.json").write_text(
        '{"schema_version": %d}\n' % (SCHEMA_VERSION + 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        compare_runs(base_dir, candidate_dir)


def test_compare_runs_rejects_missing_schema_version(tmp_path: Path):
    base = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-a", "ratio": 2.0}],
        "samples": [],
        "whest_details": [],
    }
    candidate = {
        "manifest": {"schema_version": SCHEMA_VERSION},
        "environment": {},
        "summary": {},
        "cases": [{"case_id": "case-a", "ratio": 3.0}],
        "samples": [],
        "whest_details": [],
    }

    base_dir = write_run_artifacts(tmp_path / "base", base)
    candidate_dir = write_run_artifacts(tmp_path / "candidate", candidate)
    (candidate_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        compare_runs(base_dir, candidate_dir)
