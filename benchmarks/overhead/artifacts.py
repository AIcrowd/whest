"""Versioned JSON and JSONL artifacts for overhead benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_JSON_FILENAME_MAP = {
    "manifest": "manifest.json",
    "environment": "environment.json",
    "summary": "summary.json",
}

_JSONL_FILENAME_MAP = {
    "cases": "cases.jsonl",
    "samples": "samples.jsonl",
    "whest_details": "whest_details.jsonl",
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def write_run_artifacts(root: Path, run: dict[str, object]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for key, filename in _JSON_FILENAME_MAP.items():
        _write_json(root / filename, run.get(key, {}))

    for key, filename in _JSONL_FILENAME_MAP.items():
        rows = run.get(key, [])
        if not isinstance(rows, list):
            raise TypeError(f"{key} must be a list")
        _write_jsonl(root / filename, rows)

    return root


def load_run(path: Path) -> dict[str, object]:
    root = Path(path)
    run: dict[str, object] = {}

    for key, filename in _JSON_FILENAME_MAP.items():
        run[key] = _read_json(root / filename)

    for key, filename in _JSONL_FILENAME_MAP.items():
        run[key] = _read_jsonl(root / filename)

    return run


def compare_runs(base_path: Path, candidate_path: Path) -> dict[str, object]:
    base_run = load_run(base_path)
    candidate_run = load_run(candidate_path)

    base_cases = {row["case_id"]: row for row in base_run.get("cases", [])}
    candidate_cases = {
        row["case_id"]: row for row in candidate_run.get("cases", [])
    }

    regressions: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    shared_case_ids = sorted(base_cases.keys() & candidate_cases.keys())

    for case_id in shared_case_ids:
        base_ratio = base_cases[case_id].get("ratio")
        candidate_ratio = candidate_cases[case_id].get("ratio")
        if base_ratio is None or candidate_ratio is None:
            continue
        delta = candidate_ratio - base_ratio
        row = {
            "case_id": case_id,
            "base_ratio": base_ratio,
            "candidate_ratio": candidate_ratio,
            "ratio_delta": delta,
        }
        if delta > 0:
            regressions.append(row)
        elif delta < 0:
            improvements.append(row)

    return {
        "base": base_run,
        "candidate": candidate_run,
        "regressions": regressions,
        "improvements": improvements,
        "shared_case_count": len(shared_case_ids),
    }
