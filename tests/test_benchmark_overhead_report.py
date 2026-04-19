from benchmarks.overhead.report import render_terminal_summary


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
    assert "Unclassified operations: 1" in rendered


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
