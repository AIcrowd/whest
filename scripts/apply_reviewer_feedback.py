#!/usr/bin/env python3
"""Download reviewer feedback from Google Sheet and apply weight tiers to weights.json.

Task 1: Downloads reviewer inputs (weight + notes) from the Google Sheet and
saves them to ``reviewer_feedback.json``.

Task 2: Applies the reviewer's 4-tier weight system (1/2/4/16) to
``src/mechestim/data/weights.json``, including delegation-based assignments
for ops marked with ``?``.

Usage::

    uv run python scripts/apply_reviewer_feedback.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "src" / "mechestim" / "data" / "weights.json"
FEEDBACK_PATH = REPO_ROOT / "reviewer_feedback.json"
SID = "1Jvs01W8jI4CkTNwNdNU9B8Nb102MnhpDTcE88-Y98BQ"


# ---------------------------------------------------------------------------
# gws CLI helpers (same pattern as upload_to_sheets.py)
# ---------------------------------------------------------------------------


def gws(*args: str, json_body: dict | None = None) -> dict:
    """Run a gws CLI command and return parsed JSON output."""
    cmd = ["gws"] + list(args)
    tmp_file = None
    if json_body is not None:
        body_str = json.dumps(json_body)
        if len(body_str) > 50_000:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            tmp_file.write(body_str)
            tmp_file.close()
            cmd += ["--json", f"@{tmp_file.name}"]
        else:
            cmd += ["--json", body_str]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if tmp_file:
            Path(tmp_file.name).unlink(missing_ok=True)

    for output in [result.stdout, result.stderr]:
        idx = output.find("{")
        if idx == -1:
            continue
        depth = 0
        for i, ch in enumerate(output[idx:], idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(output[idx : i + 1])
                        if "error" in parsed:
                            print(
                                f"  gws API error: {parsed['error'].get('message', '')[:200]}",
                                file=sys.stderr,
                            )
                        return parsed
                    except json.JSONDecodeError:
                        break

    if result.returncode != 0:
        print(f"gws error (exit {result.returncode}):", file=sys.stderr)
        print(f"  cmd: {' '.join(cmd[:6])}...", file=sys.stderr)
        print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
        sys.exit(1)
    return {}


# ---------------------------------------------------------------------------
# Task 1: Download reviewer data from Google Sheet
# ---------------------------------------------------------------------------


def download_reviewer_feedback() -> dict[str, dict[str, str]]:
    """Read the Google Sheet and extract reviewer inputs.

    Returns a dict: {op_name: {"weight": "...", "notes": "..."}, ...}
    """
    print("Downloading reviewer data from Google Sheet...")
    resp = gws(
        "sheets",
        "spreadsheets",
        "values",
        "get",
        "--params",
        json.dumps({
            "spreadsheetId": SID,
            "range": "'All Operations'!A1:ZZ",
        }),
    )
    all_rows = resp.get("values", [])
    if not all_rows:
        print("ERROR: No data found in sheet", file=sys.stderr)
        sys.exit(1)

    headers = all_rows[0]
    data = all_rows[1:]
    print(f"  Sheet: {len(headers)} columns, {len(data)} data rows")

    # Find key column indices.
    # The sheet has duplicate "Reviewer Weight" headers:
    #   - First occurrence (col F, index 5): numeric tier (0/1/2/4/16/?)
    #   - Second occurrence (col G, index 6): formula notes / descriptions
    # Use first-occurrence for weight, second for notes.
    op_idx = 0  # Column A is always Operation

    rev_weight_idx = None
    rev_notes_idx = None
    for i, h in enumerate(headers):
        if h == "Reviewer Weight":
            if rev_weight_idx is None:
                rev_weight_idx = i  # First "Reviewer Weight" = numeric tier
            else:
                rev_notes_idx = i  # Second "Reviewer Weight" = formula notes
                break

    # If there's only one "Reviewer Weight" column, that's the weight
    if rev_weight_idx is None:
        # Fallback search
        for i, h in enumerate(headers):
            if "reviewer" in h.lower() and "weight" in h.lower():
                rev_weight_idx = i
                break

    print(f"  Reviewer Weight column: {rev_weight_idx} ({headers[rev_weight_idx] if rev_weight_idx is not None else 'NOT FOUND'})")
    print(f"  Reviewer Notes column: {rev_notes_idx} ({headers[rev_notes_idx] if rev_notes_idx is not None else 'NOT FOUND'})")

    # Extract reviewer data
    reviewer_data: dict[str, dict[str, str]] = {}
    for row in data:
        if not row or len(row) <= op_idx:
            continue
        op = row[op_idx].strip()
        if not op:
            continue

        w = ""
        if rev_weight_idx is not None and rev_weight_idx < len(row):
            w = row[rev_weight_idx].strip()

        n = ""
        if rev_notes_idx is not None and rev_notes_idx < len(row):
            n = row[rev_notes_idx].strip()

        if w or n:
            reviewer_data[op] = {"weight": w, "notes": n}

    # Save locally
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(reviewer_data, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"  Saved {len(reviewer_data)} reviewer inputs to {FEEDBACK_PATH.name}")

    # Summary stats
    numeric = sum(1 for v in reviewer_data.values() if _is_numeric(v["weight"]))
    question = sum(1 for v in reviewer_data.values() if v["weight"] == "?")
    with_notes = sum(1 for v in reviewer_data.values() if v["notes"])
    print(f"  Breakdown: {numeric} numeric weights, {question} '?' weights, {with_notes} with notes")

    return reviewer_data


def _is_numeric(s: str) -> bool:
    """Check if a string represents a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Task 2: Apply reviewer weights to weights.json
# ---------------------------------------------------------------------------

# Delegation-based weight assignments for ops with "?" in reviewer weight.
# These are ops that delegate to other implementations:
#   - linalg delegates to contractions: weight=1
#   - linalg delegates to decompositions (cond, matrix_rank): weight=4
#   - linalg delegates to solve (tensorinv, tensorsolve): weight=1
#   - norms/trace: weight=1
#   - linalg.matrix_power: weight=4
#   - isnat: weight=1
DELEGATE_WEIGHTS: dict[str, float] = {
    # Linalg -> contraction (weight=1)
    "linalg.matmul": 1,
    "linalg.outer": 1,
    "linalg.tensordot": 1,
    "linalg.vecdot": 1,
    "linalg.cross": 1,
    "linalg.multi_dot": 1,
    # Linalg -> decomposition (weight=4)
    "linalg.cond": 4,
    "linalg.matrix_rank": 4,
    # Linalg -> solve (weight=1)
    "linalg.tensorinv": 1,
    "linalg.tensorsolve": 1,
    # Norms / trace (weight=1)
    "linalg.norm": 1,
    "linalg.vector_norm": 1,
    "linalg.matrix_norm": 1,
    "linalg.trace": 1,
    # Matrix power (iterative squaring -> decomposition-class, weight=4)
    "linalg.matrix_power": 4,
    # Other
    "isnat": 1,
}


def apply_weight_tiers() -> None:
    """Overwrite weights.json with reviewer's tier values.

    1. For each op with a numeric reviewer weight, overwrite the empirical value.
    2. For ops with '?' (delegation-based), assign from DELEGATE_WEIGHTS.
    3. Keep empirical values in meta.validation for reference.
    """
    # Load reviewer feedback
    with open(FEEDBACK_PATH) as f:
        reviewer = json.load(f)

    # Load current weights
    with open(WEIGHTS_PATH) as f:
        data = json.load(f)

    weights = data["weights"]

    # Store original empirical values for validation reference,
    # preserving existing validation data (perf_vs_timing etc.)
    if "meta" not in data:
        data["meta"] = {}
    validation = data["meta"].get("validation", {})
    validation["empirical_weights_before_review"] = dict(weights)
    validation["empirical_weights_note"] = (
        "Original perf-counter-derived weights before reviewer tier assignment"
    )
    data["meta"]["validation"] = validation

    updated = 0
    skipped = 0

    # Step 1: Apply numeric reviewer weights
    for op, feedback in reviewer.items():
        w = feedback.get("weight", "").strip()
        if not w or w == "?":
            continue
        try:
            new_weight = float(w)
        except ValueError:
            # Non-numeric (formula suggestion etc.) -- skip for weight tier
            skipped += 1
            continue

        if op in weights:
            weights[op] = new_weight
            updated += 1
        else:
            print(f"  WARNING: reviewer weight for '{op}' but op not in weights.json")

    # Step 2: Assign delegation-based weights for ops with '?'
    delegate_applied = 0
    for op, w in DELEGATE_WEIGHTS.items():
        if op in weights:
            weights[op] = w
            delegate_applied += 1

    # Write back
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Applied reviewer weights to {WEIGHTS_PATH.name}:")
    print(f"  {updated} ops updated from numeric reviewer weights")
    print(f"  {delegate_applied} ops updated from delegation assignments")
    print(f"  {skipped} ops skipped (non-numeric, non-'?' reviewer input)")


def verify_weights() -> bool:
    """Verify key weights match expected reviewer tier values."""
    with open(WEIGHTS_PATH) as f:
        data = json.load(f)
    w = data["weights"]

    checks = {
        "add": 1,
        "sin": 16,
        "std": 2,
        "linalg.svd": 4,
        "matmul": 1,
        "linalg.solve": 1,
    }

    print("\nVerification:")
    all_ok = True
    for op, expected in checks.items():
        actual = w.get(op, "MISSING")
        ok = actual == expected
        status = "OK" if ok else f"MISMATCH (got {actual})"
        print(f"  {op:25s} expected={expected:>3}  {status}")
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Task 1: Download reviewer feedback from Google Sheet")
    print("=" * 60)
    download_reviewer_feedback()

    print()
    print("=" * 60)
    print("Task 2: Apply reviewer weight tiers to weights.json")
    print("=" * 60)
    apply_weight_tiers()

    if not verify_weights():
        print("\nWARNING: Some weight verifications failed!", file=sys.stderr)
        sys.exit(1)

    print("\nDone. Next steps:")
    print("  uv run python scripts/generate_empirical_weights_docs.py")


if __name__ == "__main__":
    main()
