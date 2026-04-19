"""Terminal reporting for overhead benchmark runs."""

from __future__ import annotations


def render_terminal_summary(
    manifest: dict[str, object], cases: list[dict[str, object]]
) -> str:
    """Render a compact terminal summary."""
    total = len(cases)
    failed = [case for case in cases if not case.get("passed", False)]
    passed = total - len(failed)
    unclassified = manifest.get("unclassified_operations", [])

    lines = [
        f"Mode: {manifest.get('mode', 'unknown')}",
        f"Cases: {total} total, {passed} passed, {len(failed)} failed",
        f"Unclassified operations: {len(unclassified)}",
    ]

    if not failed:
        lines.append("All cases passed.")
        return "\n".join(lines)

    lines.append("Worst regressions:")
    ranked = sorted(
        failed,
        key=lambda case: (
            float(case.get("ratio", 0.0)) - float(case.get("threshold", 0.0)),
            float(case.get("ratio", 0.0)),
        ),
        reverse=True,
    )
    for case in ranked[:3]:
        lines.append(
            f"- {case['case_id']}: {float(case['ratio']):.2f}x > "
            f"{float(case['threshold']):.2f}x ({case['policy_source']})"
        )

    return "\n".join(lines)
