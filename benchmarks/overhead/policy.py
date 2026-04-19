"""Policy loading and threshold evaluation for overhead benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.overhead import SCHEMA_VERSION

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "overhead_thresholds.json"


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load a policy document from disk."""
    policy_path = _DEFAULT_POLICY_PATH if path is None else Path(path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    schema_version = policy.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {schema_version}"
        )
    policy["schema_version"] = SCHEMA_VERSION
    return policy


def _policy_bucket(
    case_result: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    surface = case_result.get("surface")
    family = case_result.get("family")
    surface_policies = policy.get("surface", {})
    family_policies = policy.get("family", {})

    if surface in surface_policies:
        return f"surface:{surface}", surface_policies[surface]
    if family in family_policies:
        return f"family:{family}", family_policies[family]
    return "default", policy.get("default", {})


def evaluate_case(
    case_result: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Annotate a case result with policy evaluation metadata."""
    policy_source, bucket = _policy_bucket(case_result, policy)
    threshold = float(bucket.get("ratio_max", float("inf")))
    ratio = float(case_result.get("ratio", float("inf")))
    return {
        **case_result,
        "threshold": threshold,
        "policy_source": policy_source,
        "passed": ratio <= threshold,
    }


def _max_ratio(cases: list[dict[str, Any]], *, current: float) -> float:
    ratios = [float(case["ratio"]) for case in cases if "ratio" in case]
    return max(ratios) if ratios else current


def suggest_thresholds(
    cases: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    """Suggest thresholds based on observed case ratios."""
    suggestion: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "suggested",
        "default": {
            "ratio_max": _max_ratio(
                cases, current=float(policy.get("default", {}).get("ratio_max", float("inf")))
            )
        },
    }

    family_policies = policy.get("family", {})
    if family_policies:
        suggestion["family"] = {}
        for family in sorted(family_policies):
            current = float(family_policies.get(family, {}).get("ratio_max", suggestion["default"]["ratio_max"]))
            matching = [case for case in cases if case.get("family") == family]
            suggestion["family"][family] = {"ratio_max": _max_ratio(matching, current=current)}

    surface_policies = policy.get("surface", {})
    if surface_policies:
        suggestion["surface"] = {}
        for surface in sorted(surface_policies):
            current = float(surface_policies.get(surface, {}).get("ratio_max", suggestion["default"]["ratio_max"]))
            matching = [case for case in cases if case.get("surface") == surface]
            suggestion["surface"][surface] = {"ratio_max": _max_ratio(matching, current=current)}

    return suggestion
