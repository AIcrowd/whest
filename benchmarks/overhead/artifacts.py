"""Versioned JSON and JSONL artifacts for overhead benchmark runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchmarks.overhead import SCHEMA_VERSION

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

_FLOAT_TAG = "__whest_float__"


def _encode_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if value != value:
            return {_FLOAT_TAG: "nan"}
        if value == float("inf"):
            return {_FLOAT_TAG: "inf"}
        if value == float("-inf"):
            return {_FLOAT_TAG: "-inf"}
    if isinstance(value, dict):
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    return value


def _decode_value(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            token = value[_FLOAT_TAG]
            if token == "nan":
                return float("nan")
            if token == "inf":
                return float("inf")
            if token == "-inf":
                return float("-inf")
            raise ValueError(f"unsupported encoded float token: {token!r}")
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _json_text(payload: object) -> str:
    return json.dumps(_encode_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_json_text(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    lines = [
        json.dumps(_encode_value(row), sort_keys=True, allow_nan=False)
        for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return _decode_value(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(_decode_value(json.loads(line)))
    return rows


def _manifest_schema_version(run: dict[str, object]) -> int:
    manifest = run.get("manifest")
    if manifest is None:
        return SCHEMA_VERSION
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a mapping")
    schema_version = manifest.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {schema_version}"
        )
    return SCHEMA_VERSION


def write_run_artifacts(root: Path, run: dict[str, object]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    raw_manifest = run.get("manifest", {})
    if raw_manifest is None:
        raw_manifest = {}
    if not isinstance(raw_manifest, dict):
        raise TypeError("manifest must be a mapping")
    manifest = dict(raw_manifest)
    manifest.setdefault("schema_version", SCHEMA_VERSION)
    _manifest_schema_version({"manifest": manifest})
    _write_json(root / _JSON_FILENAME_MAP["manifest"], manifest)

    for key, filename in _JSON_FILENAME_MAP.items():
        if key == "manifest":
            continue
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
    base_schema_version = _manifest_schema_version(base_run)
    candidate_schema_version = _manifest_schema_version(candidate_run)
    if base_schema_version != candidate_schema_version:
        raise ValueError(
            "schema_version mismatch: "
            f"base={base_schema_version}, candidate={candidate_schema_version}"
        )

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
