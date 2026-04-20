from pathlib import Path

from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.policy import evaluate_case, load_policy, suggest_thresholds


def test_load_policy_reads_json_file(tmp_path: Path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        """
        {
          "schema_version": 1,
          "policy_version": "custom",
          "default": {"ratio_max": 3.5}
        }
        """,
        encoding="utf-8",
    )

    policy = load_policy(policy_path)

    assert policy["schema_version"] == SCHEMA_VERSION
    assert policy["policy_version"] == "custom"
    assert policy["default"]["ratio_max"] == 3.5


def test_load_policy_rejects_missing_schema_version(tmp_path: Path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        """
        {
          "policy_version": "custom",
          "default": {"ratio_max": 3.5}
        }
        """,
        encoding="utf-8",
    )

    try:
        load_policy(policy_path)
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("expected load_policy() to reject missing schema_version")


def test_evaluate_case_prefers_surface_then_family_then_default():
    policy = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "test",
        "default": {"ratio_max": 6.0},
        "family": {"contractions": {"ratio_max": 4.0}},
        "surface": {"operator": {"ratio_max": 7.0}},
    }

    operator_case = {
        "case_id": "matmul-operator-medium",
        "family": "contractions",
        "surface": "operator",
        "ratio": 6.5,
    }
    api_case = {
        "case_id": "matmul-api-medium",
        "family": "contractions",
        "surface": "api",
        "ratio": 6.5,
    }
    fallback_case = {
        "case_id": "unknown-api-medium",
        "family": "unknown",
        "surface": "api",
        "ratio": 5.5,
    }

    operator_result = evaluate_case(operator_case, policy)
    api_result = evaluate_case(api_case, policy)
    fallback_result = evaluate_case(fallback_case, policy)

    assert operator_result["passed"] is True
    assert operator_result["threshold"] == 7.0
    assert operator_result["policy_source"] == "surface:operator"

    assert api_result["passed"] is False
    assert api_result["threshold"] == 4.0
    assert api_result["policy_source"] == "family:contractions"

    assert fallback_result["passed"] is True
    assert fallback_result["threshold"] == 6.0
    assert fallback_result["policy_source"] == "default"


def test_suggest_thresholds_preserves_policy_shape_and_uses_observed_ratios():
    policy = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "bootstrap",
        "default": {"ratio_max": 6.0},
        "family": {
            "pointwise": {"ratio_max": 6.0},
            "contractions": {"ratio_max": 4.0},
        },
        "surface": {"operator": {"ratio_max": 7.0}},
    }
    cases = [
        {
            "case_id": "add-api-medium",
            "family": "pointwise",
            "surface": "api",
            "ratio": 1.5,
        },
        {
            "case_id": "matmul-api-medium",
            "family": "contractions",
            "surface": "api",
            "ratio": 3.25,
        },
        {
            "case_id": "matmul-operator-medium",
            "family": "contractions",
            "surface": "operator",
            "ratio": 6.75,
        },
    ]

    suggestion = suggest_thresholds(cases, policy)

    assert suggestion == {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "suggested",
        "default": {"ratio_max": 6.0},
        "family": {
            "pointwise": {"ratio_max": 1.5},
            "contractions": {"ratio_max": 3.25},
        },
        "surface": {"operator": {"ratio_max": 6.75}},
    }
