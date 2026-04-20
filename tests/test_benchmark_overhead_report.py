import json

from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.report import (
    build_browser_report_payload,
    render_browser_report,
    render_terminal_summary,
    write_browser_report,
)


def test_render_terminal_summary_highlights_worst_regressions():
    manifest = {
        "mode": "full",
        "selected_cases": ["add-api-medium", "matmul-api-medium", "matmul-operator-medium"],
        "unclassified_operations": [{"qualified_name": "whest.linalg.matmul"}],
    }
    cases = [
        {
            "case_id": "add-api-medium",
            "ratio": 1.2,
            "threshold": 6.0,
            "passed": True,
            "policy_source": "family:pointwise",
        },
        {
            "case_id": "matmul-api-medium",
            "ratio": 5.1,
            "threshold": 4.0,
            "passed": False,
            "policy_source": "family:contractions",
        },
        {
            "case_id": "matmul-operator-medium",
            "ratio": 7.6,
            "threshold": 7.0,
            "passed": False,
            "policy_source": "surface:operator",
        },
    ]

    rendered = render_terminal_summary(manifest, cases)

    assert "Mode: full" in rendered
    assert "Cases: 3 total, 1 passed, 2 failed" in rendered
    assert "Worst regressions" in rendered
    assert "matmul-operator-medium" in rendered
    assert "7.60x > 7.00x" in rendered
    assert "matmul-api-medium" in rendered
    assert "Discovery-only operations: 1" in rendered


def test_render_terminal_summary_reports_all_pass_cleanly():
    manifest = {"mode": "ci", "selected_cases": ["add-api-tiny"], "unclassified_operations": []}
    cases = [
        {
            "case_id": "add-api-tiny",
            "ratio": 1.05,
            "threshold": 6.0,
            "passed": True,
            "policy_source": "default",
        }
    ]

    rendered = render_terminal_summary(manifest, cases)

    assert "Mode: ci" in rendered
    assert "Cases: 1 total, 1 passed, 0 failed" in rendered
    assert "All cases passed." in rendered


def test_build_browser_report_payload_flattens_cases_and_counts():
    run = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "mode": "full",
            "selected_cases": ["add-api-tiny", "matmul-api-medium"],
            "unclassified_operations": ["whest.array"],
        },
        "environment": {
            "hardware": {"arch": "arm64", "cpu_cores": 8},
            "software": {"python": "3.14.3", "numpy": "2.2.6"},
            "timestamp": "2026-04-20T00:00:00+00:00",
        },
        "summary": {
            "case_count": 2,
            "passed": 1,
            "failed": 1,
            "worst_ratio": 12.0,
        },
        "operations": [
            {
                "slug": "add",
                "name": "add",
                "category": "counted_binary",
                "coverage_status": "measured",
                "measured_case_count": 1,
                "representative_ratio": 12.0,
                "worst_ratio": 12.0,
                "case_ids": ["add-api-tiny"],
            },
            {
                "slug": "apply_along_axis",
                "name": "apply_along_axis",
                "category": "counted_custom",
                "coverage_status": "profile_missing",
                "measured_case_count": 0,
                "case_ids": [],
            },
        ],
        "cases": [
            {
                "case_id": "add-api-tiny",
                "op_name": "add",
                "family": "pointwise",
                "surface": "api",
                "size_name": "tiny",
                "dtype": "float64",
                "source_file": "src/whest/_pointwise.py",
                "ratio": 12.0,
                "threshold": 6.0,
                "passed": False,
                "policy_source": "default",
                "numpy": {"median_ns": 100.0},
                "whest": {"median_ns": 1200.0},
                "startup": {"ratio": 1.4},
                "whest_details": {
                    "flops_used": 8,
                    "op_count": 1,
                    "tracked_time_s": 0.0001,
                    "operations": {"add": {"calls": 1, "duration": 0.0001, "flop_cost": 8}},
                },
            },
            {
                "case_id": "matmul-api-medium",
                "op_name": "matmul",
                "family": "contractions",
                "surface": "api",
                "size_name": "medium",
                "dtype": "float64",
                "source_file": "src/whest/_pointwise.py",
                "ratio": 1.1,
                "threshold": 4.0,
                "passed": True,
                "policy_source": "family:contractions",
                "numpy": {"median_ns": 2_000_000.0},
                "whest": {"median_ns": 2_200_000.0},
                "startup": {"ratio": 1.2},
                "whest_details": {
                    "flops_used": 1024,
                    "op_count": 1,
                    "tracked_time_s": 0.02,
                    "operations": {
                        "matmul": {"calls": 1, "duration": 0.02, "flop_cost": 1024}
                    },
                },
            },
        ],
    }

    payload = build_browser_report_payload(run)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"]["case_count"] == 2
    assert payload["operations"][0]["slug"] == "add"
    assert payload["operations"][1]["coverage_status"] == "profile_missing"
    assert payload["aggregates"]["families"]["pointwise"]["failed"] == 1
    assert payload["aggregates"]["surfaces"]["api"]["count"] == 2
    assert payload["aggregates"]["sizes"]["tiny"]["count"] == 1
    assert payload["top_cases_by_ratio"][0]["case_id"] == "add-api-tiny"
    assert payload["cases"][0]["steady_state_delta_ns"] == 1100.0
    assert payload["cases"][0]["operation_names"] == ["add"]


def test_write_browser_report_emits_html_and_json(tmp_path):
    run = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "mode": "full",
            "selected_cases": ["add-api-tiny"],
            "unclassified_operations": [],
        },
        "environment": {"software": {"python": "3.14.3"}, "timestamp": "2026-04-20T00:00:00+00:00"},
        "summary": {"case_count": 1, "passed": 1, "failed": 0, "worst_ratio": 1.2},
        "operations": [
            {
                "slug": "add",
                "name": "add",
                "coverage_status": "partial_error",
                "measured_case_count": 1,
                "expected_case_count": 2,
                "benchmark_error_count": 1,
                "error_messages": ["add-api-medium: simulated failure"],
                "case_ids": ["add-api-tiny"],
            }
        ],
        "cases": [
            {
                "case_id": "add-api-tiny",
                "op_name": "add",
                "family": "pointwise",
                "surface": "api",
                "size_name": "tiny",
                "dtype": "float64",
                "source_file": "src/whest/_pointwise.py",
                "ratio": 1.2,
                "threshold": 6.0,
                "passed": True,
                "policy_source": "default",
                "numpy": {"median_ns": 100.0},
                "whest": {"median_ns": 120.0},
                "startup": {"ratio": 1.1},
                "whest_details": {
                    "flops_used": 8,
                    "op_count": 1,
                    "tracked_time_s": 0.0001,
                    "operations": {"add": {"calls": 1, "duration": 0.0001, "flop_cost": 8}},
                },
            }
        ],
    }

    report_path = write_browser_report(tmp_path, run=run)
    html = report_path.read_text(encoding="utf-8")
    data = json.loads((tmp_path / "report_data.json").read_text(encoding="utf-8"))

    assert report_path == tmp_path / "report.html"
    assert "Whest Overhead Benchmark Report" in html
    assert "add-api-tiny" in html
    assert "report-hero" in html
    assert "report-section-card" in html
    assert "metric-tile" in html
    assert "partial_error" in html
    assert "benchmark errors" in html
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["summary"]["case_count"] == 1


def test_render_browser_report_includes_filter_shell():
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest": {"mode": "full", "selected_cases": ["add-api-tiny"]},
        "summary": {"case_count": 1, "passed": 1, "failed": 0, "worst_ratio": 1.2},
        "operations": [
            {
                "slug": "add",
                "name": "add",
                "coverage_status": "partial_error",
                "measured_case_count": 1,
                "expected_case_count": 2,
                "benchmark_error_count": 1,
                "error_messages": ["add-api-medium: simulated failure"],
                "case_ids": ["add-api-tiny"],
                "representative_ratio": 1.2,
                "worst_ratio": 1.2,
            },
            {
                "slug": "apply_along_axis",
                "name": "apply_along_axis",
                "coverage_status": "profile_missing",
                "measured_case_count": 0,
                "case_ids": [],
            },
        ],
        "aggregates": {
            "families": {"pointwise": {"count": 1, "passed": 1, "failed": 0}},
            "surfaces": {"api": {"count": 1, "passed": 1, "failed": 0}},
            "sizes": {"tiny": {"count": 1, "passed": 1, "failed": 0}},
        },
        "environment": {"software": {"python": "3.14.3"}},
        "accountability": {"unclassified_operations": ["whest.linalg.matmul"]},
        "cases": [
            {
                "case_id": "add-api-tiny",
                "family": "pointwise",
                "surface": "api",
                "size_name": "tiny",
                "passed": True,
                "ratio": 1.2,
                "threshold": 6.0,
                "policy_source": "default",
                "startup_ratio": 1.1,
                "steady_state_delta_ns": 20.0,
                "startup_delta_ns": 40.0,
                "operation_names": ["add"],
                "source_file": "src/whest/_pointwise.py",
            }
        ],
        "top_cases_by_ratio": [{"case_id": "add-api-tiny", "ratio": 1.2}],
        "comparison": None,
    }

    html = render_browser_report(payload)

    assert "case-search" in html
    assert "case-table-body" in html
    assert "report-data" in html
    assert "operation-table-body" in html
    assert "apiReferenceShell" in html or "operation-inventory" in html
    assert "--coral:" in html
    assert "--gray-900:" in html
    assert "case-status-pill" in html
    assert "policy-pill" in html
    assert 'href="#case-drilldown"' in html
    assert "data-case-filter" in html
    assert "discovery-only callables" in html
    assert 'data-sort-table="operations"' in html
    assert 'data-sort-table="cases"' in html
    assert 'data-sort-key="worst_ratio"' in html
    assert 'data-sort-key="ratio"' in html
