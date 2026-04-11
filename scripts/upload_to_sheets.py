#!/usr/bin/env python3
"""Upload weights.csv to Google Sheets with formatting, dropdowns, and color coding.

Requires: gws CLI (https://github.com/googleworkspace/cli) authenticated via `gws auth login`.

Usage::

    python scripts/upload_to_sheets.py
    python scripts/upload_to_sheets.py --csv src/mechestim/data/weights.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "src" / "mechestim" / "data" / "weights.csv"
TITLE = "mechestim FLOP Weight Calibration Review"


import tempfile


def gws(*args: str, json_body: dict | None = None) -> dict:
    """Run a gws CLI command and return parsed JSON output.

    For large JSON bodies, writes to a temp file to avoid CLI arg length limits.
    """
    cmd = ["gws"] + list(args)
    tmp_file = None
    if json_body is not None:
        body_str = json.dumps(json_body)
        # If body is large, write to temp file and use @file syntax
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

    # Try stdout first, then stderr
    for output in [result.stdout, result.stderr]:
        idx = output.find("{")
        if idx == -1:
            continue
        # Find the first complete JSON object
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
                            print(f"  gws API error: {parsed['error'].get('message', '')[:200]}",
                                  file=sys.stderr)
                        return parsed
                    except json.JSONDecodeError:
                        break

    if result.returncode != 0:
        print(f"gws error (exit {result.returncode}):", file=sys.stderr)
        print(f"  cmd: {' '.join(cmd[:6])}...", file=sys.stderr)
        print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
        sys.exit(1)
    return {}


def load_csv(path: Path) -> list[list[str]]:
    """Load CSV as a list of rows (each row is a list of strings)."""
    with open(path) as f:
        reader = csv.reader(f)
        return [row for row in reader]


def create_spreadsheet() -> str:
    """Create a new Google Sheets spreadsheet and return its ID."""
    print(f"Creating spreadsheet: {TITLE}")
    resp = gws(
        "sheets", "spreadsheets", "create",
        json_body={
            "properties": {"title": TITLE},
            "sheets": [
                {"properties": {"title": "All Operations", "sheetId": 0}},
                {"properties": {"title": "Review Summary", "sheetId": 1}},
            ],
        },
    )
    sid = resp.get("spreadsheetId", "")
    if not sid:
        print("ERROR: Could not create spreadsheet", file=sys.stderr)
        sys.exit(1)
    print(f"  Spreadsheet ID: {sid}")
    print(f"  URL: https://docs.google.com/spreadsheets/d/{sid}")
    return sid


def upload_data(sid: str, rows: list[list[str]], *, skip_col: int | None = 5) -> None:
    """Upload CSV data to Sheet 1.

    Parameters
    ----------
    skip_col : int or None
        Column index to skip (default: 5 = "Reviewer Weight" column F).
        When updating an existing sheet, this preserves reviewer edits.
        Set to None for fresh sheets (creates all columns).
    """
    print(f"Uploading {len(rows)} rows ({len(rows[0])} columns)...")

    CHUNK = 50
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start:start + CHUNK]
        row_start = start + 1

        if skip_col is not None and skip_col > 0:
            # Upload columns before skip_col (A-E)
            chunk_before = [row[:skip_col] for row in chunk]
            gws(
                "sheets", "spreadsheets", "values", "update",
                "--params", json.dumps({
                    "spreadsheetId": sid,
                    "range": f"'All Operations'!A{row_start}",
                    "valueInputOption": "USER_ENTERED",
                }),
                json_body={"values": chunk_before},
            )
            # Upload columns after skip_col (G onward)
            col_letter = chr(65 + skip_col + 1)  # F+1 = G
            chunk_after = [row[skip_col + 1:] for row in chunk]
            gws(
                "sheets", "spreadsheets", "values", "update",
                "--params", json.dumps({
                    "spreadsheetId": sid,
                    "range": f"'All Operations'!{col_letter}{row_start}",
                    "valueInputOption": "USER_ENTERED",
                }),
                json_body={"values": chunk_after},
            )
        else:
            # Upload all columns (fresh sheet)
            gws(
                "sheets", "spreadsheets", "values", "update",
                "--params", json.dumps({
                    "spreadsheetId": sid,
                    "range": f"'All Operations'!A{row_start}",
                    "valueInputOption": "USER_ENTERED",
                }),
                json_body={"values": chunk},
            )

    print("  Data uploaded.")


def _color(r: float, g: float, b: float) -> dict:
    """Create a color dict for the Sheets API (0-1 scale)."""
    return {"red": r, "green": g, "blue": b}


def _cond_rule(sheet_id: int, col: int, num_rows: int,
               condition_type: str, values: list, fmt: dict) -> dict:
    """Build a conditional format rule for a column."""
    rule = {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": num_rows,
                    "startColumnIndex": col,
                    "endColumnIndex": col + 1,
                }],
                "booleanRule": {
                    "condition": {
                        "type": condition_type,
                        "values": values,
                    },
                    "format": fmt,
                },
            },
            "index": 0,
        }
    }
    return rule


def _text_eq_rule(sheet_id: int, col: int, num_rows: int,
                  text: str, bg: dict, fg: dict | None = None) -> dict:
    """Conditional format: cell text equals a specific value."""
    fmt = {"backgroundColor": bg}
    if fg:
        fmt["textFormat"] = {"foregroundColor": fg}
    return _cond_rule(
        sheet_id, col, num_rows,
        "TEXT_EQ", [{"userEnteredValue": text}], fmt,
    )


def _number_between_rule(sheet_id: int, col: int, num_rows: int,
                         lo: str, hi: str, bg: dict) -> dict:
    """Conditional format: number between lo and hi."""
    return _cond_rule(
        sheet_id, col, num_rows,
        "NUMBER_BETWEEN",
        [{"userEnteredValue": lo}, {"userEnteredValue": hi}],
        {"backgroundColor": bg},
    )


def _number_rule(sheet_id: int, col: int, num_rows: int,
                 cond_type: str, value: str, bg: dict,
                 fg: dict | None = None) -> dict:
    """Conditional format: number comparison."""
    fmt = {"backgroundColor": bg}
    if fg:
        fmt["textFormat"] = {"foregroundColor": fg}
    return _cond_rule(
        sheet_id, col, num_rows,
        cond_type, [{"userEnteredValue": value}], fmt,
    )


def apply_formatting(sid: str, num_rows: int, num_cols: int) -> None:
    """Apply all formatting, dropdowns, and conditional rules."""
    print("Applying formatting...")
    sheet_id = 0
    requests = []

    # ---- Freeze header row + column A ----
    requests.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {
                    "frozenRowCount": 1,
                    "frozenColumnCount": 1,
                },
            },
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
        }
    })

    # ---- Header row formatting ----
    # Section A (cols 0-8): dark blue-gray bg, white text, bold
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0, "endRowIndex": 1,
                "startColumnIndex": 0, "endColumnIndex": 9,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": _color(0.2, 0.3, 0.4),
                    "textFormat": {
                        "foregroundColor": _color(1, 1, 1),
                        "bold": True,
                        "fontSize": 10,
                    },
                }
            },
            "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat",
        }
    })
    # Section B (cols 9-20): lighter gray bg
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0, "endRowIndex": 1,
                "startColumnIndex": 9, "endColumnIndex": num_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": _color(0.6, 0.6, 0.65),
                    "textFormat": {
                        "foregroundColor": _color(1, 1, 1),
                        "bold": True,
                        "fontSize": 10,
                    },
                }
            },
            "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat",
        }
    })

    # ---- Reviewer Weight column (F, index 5): light yellow bg ----
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 1, "endRowIndex": num_rows,
                "startColumnIndex": 5, "endColumnIndex": 6,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": _color(1.0, 0.98, 0.8),
                }
            },
            "fields": "userEnteredFormat.backgroundColor",
        }
    })

    # ---- Status dropdown (col B, index 1) ----
    requests.append({
        "setDataValidation": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 1, "endRowIndex": num_rows,
                "startColumnIndex": 1, "endColumnIndex": 2,
            },
            "rule": {
                "condition": {
                    "type": "ONE_OF_LIST",
                    "values": [
                        {"userEnteredValue": "benchmarked"},
                        {"userEnteredValue": "alias"},
                        {"userEnteredValue": "excluded"},
                        {"userEnteredValue": "free"},
                        {"userEnteredValue": "blacklisted"},
                        {"userEnteredValue": "blacklisted-by-reviewer"},
                        {"userEnteredValue": "keep"},
                    ],
                },
                "showCustomUi": True,
                "strict": False,
            },
        }
    })

    # ---- Conditional formatting: Status column (B, index 1) ----
    WHITE = _color(1, 1, 1)
    status_rules = [
        ("benchmarked",            _color(0.85, 0.93, 0.83), None),
        ("alias",                  _color(0.82, 0.88, 0.95), None),
        ("excluded",               _color(0.9, 0.9, 0.9),   None),
        ("free",                   _color(0.96, 0.96, 0.96), None),
        ("blacklisted",            _color(0.87, 0.36, 0.34), WHITE),
        ("blacklisted-by-reviewer", _color(0.6, 0.15, 0.15), WHITE),
        ("keep",                   _color(0.2, 0.55, 0.24),  WHITE),
    ]
    for text, bg, fg in status_rules:
        requests.append(_text_eq_rule(sheet_id, 1, num_rows, text, bg, fg))

    # ---- Conditional formatting: Weight column (E, index 4) ----
    # Order matters: more specific rules first (Sheets evaluates top-down, first match wins)
    weight_rules = [
        ("NUMBER_LESS", "0.001",  _color(1, 1, 1), _color(0.8, 0.1, 0.1)),  # zero FP: red text
        ("NUMBER_LESS", "0.5",    _color(0.85, 0.95, 0.85), None),           # negligible: light green
        ("NUMBER_LESS", "2.0",    _color(0.72, 0.88, 0.72), None),           # baseline: green
        ("NUMBER_LESS", "20.0",   _color(1.0, 0.95, 0.6),   None),           # moderate: yellow
        ("NUMBER_LESS", "100.0",  _color(1.0, 0.8, 0.4),    None),           # heavy: orange
        ("NUMBER_GREATER_THAN_EQ", "100.0", _color(0.92, 0.45, 0.4), None),  # extreme: red
    ]
    for cond_type, value, bg, fg in weight_rules:
        requests.append(_number_rule(sheet_id, 4, num_rows, cond_type, value, bg, fg))

    # ---- Conditional formatting: Confidence column (H, index 7) ----
    conf_rules = [
        ("high",   _color(0.72, 0.88, 0.72)),
        ("medium", _color(1.0, 0.95, 0.6)),
        ("low",    _color(0.95, 0.7, 0.65)),
    ]
    for text, bg in conf_rules:
        requests.append(_text_eq_rule(sheet_id, 7, num_rows, text, bg))

    # ---- Conditional formatting: Perf/Timing Agreement (M, index 12) ----
    # Green: 0.5-2.0, Yellow: 0.2-0.5 or 2.0-5.0, Red: <0.2 or >5.0
    requests.append(_number_between_rule(sheet_id, 12, num_rows, "0.5", "2.0",
                                          _color(0.72, 0.88, 0.72)))
    requests.append(_number_between_rule(sheet_id, 12, num_rows, "0.2", "0.5",
                                          _color(1.0, 0.95, 0.6)))
    requests.append(_number_between_rule(sheet_id, 12, num_rows, "2.0", "5.0",
                                          _color(1.0, 0.95, 0.6)))
    requests.append(_number_rule(sheet_id, 12, num_rows,
                                  "NUMBER_LESS", "0.2",
                                  _color(0.95, 0.7, 0.65)))
    requests.append(_number_rule(sheet_id, 12, num_rows,
                                  "NUMBER_GREATER", "5.0",
                                  _color(0.95, 0.7, 0.65)))

    # ---- Column widths ----
    col_widths = {
        0: 200,   # Operation
        1: 140,   # Status
        2: 140,   # Category
        3: 180,   # Cost Formula
        4: 90,    # Weight
        5: 120,   # Reviewer Weight
        6: 250,   # Effective Cost Example
        7: 100,   # Confidence
        8: 400,   # Notes
        9: 300,   # Exclusion Reason
        10: 100,  # HW FP Instructions
        11: 100,  # Timing Weight
        12: 100,  # Perf/Timing Agreement
        13: 100,  # CV
        14: 250,  # Benchmark Command
        15: 140,  # Benchmark Size
        16: 180,  # Total Perf Instructions
        17: 120,  # Total Timing
        18: 350,  # Implementation URL
        19: 100,  # Weight Tier
        20: 70,   # Repeats
    }
    for col_idx, width in col_widths.items():
        if col_idx < num_cols:
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": col_idx,
                        "endIndex": col_idx + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            })

    # ---- Wrap text on Notes column ----
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 1, "endRowIndex": num_rows,
                "startColumnIndex": 8, "endColumnIndex": 9,
            },
            "cell": {
                "userEnteredFormat": {"wrapStrategy": "WRAP"}
            },
            "fields": "userEnteredFormat.wrapStrategy",
        }
    })

    # ---- Send batch updates in chunks (avoid CLI arg length limits) ----
    CHUNK_SIZE = 10
    for i in range(0, len(requests), CHUNK_SIZE):
        chunk = requests[i : i + CHUNK_SIZE]
        print(f"  Sending batch {i // CHUNK_SIZE + 1} ({len(chunk)} requests)...")
        gws(
            "sheets", "spreadsheets", "batchUpdate",
            "--params", json.dumps({"spreadsheetId": sid}),
            json_body={"requests": chunk},
        )
    print(f"  Formatting applied ({len(requests)} requests total).")


def create_summary_sheet(sid: str, rows: list[list[str]]) -> None:
    """Populate the Review Summary sheet with formulas."""
    print("Creating summary sheet...")
    header = rows[0]
    data_rows = rows[1:]

    # Build summary data
    from collections import Counter
    statuses = Counter(r[1] for r in data_rows)  # col B
    categories = Counter(r[2] for r in data_rows if r[2])  # col C
    tiers = Counter(r[19] for r in data_rows if len(r) > 19 and r[19])  # col T (Weight Tier)
    confs = Counter(r[7] for r in data_rows if r[7])  # col H

    summary = [
        ["mechestim FLOP Weight Calibration — Review Summary", ""],
        ["", ""],
        ["Status", "Count"],
    ]
    for status in ["benchmarked", "alias", "excluded", "free", "blacklisted"]:
        summary.append([status, str(statuses.get(status, 0))])
    summary.append(["", ""])
    summary.append(["Category", "Count"])
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        summary.append([cat, str(count)])
    summary.append(["", ""])
    summary.append(["Weight Tier", "Count"])
    for tier in ["negligible", "baseline", "moderate", "heavy", "extreme"]:
        summary.append([tier, str(tiers.get(tier, 0))])
    summary.append(["", ""])
    summary.append(["Confidence", "Count"])
    for conf in ["high", "medium", "low"]:
        summary.append([conf, str(confs.get(conf, 0))])
    summary.append(["", ""])
    summary.append(["Total operations", str(len(data_rows))])
    summary.append(["", ""])
    summary.append(["Instructions:", ""])
    summary.append(["1. Review the 'Weight' column (E) in 'All Operations'", ""])
    summary.append(["2. Change Status dropdown to 'keep' or 'blacklisted-by-reviewer'", ""])
    summary.append(["3. Enter your preferred weight in 'Reviewer Weight' (F)", ""])
    summary.append(["4. Weight = 1.0 means same cost as np.add per analytical FLOP", ""])
    summary.append(["5. Weight < 1.0 means cheaper (e.g., matmul=0.46 due to FMA)", ""])
    summary.append(["6. Weight > 1.0 means more expensive (e.g., sin=18.39)", ""])

    gws(
        "sheets", "spreadsheets", "values", "update",
        "--params", json.dumps({
            "spreadsheetId": sid,
            "range": "'Review Summary'!A1",
            "valueInputOption": "RAW",
        }),
        json_body={"values": summary},
    )
    print("  Summary sheet created.")


def main():
    parser = argparse.ArgumentParser(description="Upload weights CSV to Google Sheets")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Path to weights CSV (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows, {len(rows[0])} columns from {args.csv}")

    sid = create_spreadsheet()
    upload_data(sid, rows)
    apply_formatting(sid, num_rows=len(rows), num_cols=len(rows[0]))
    create_summary_sheet(sid, rows)

    url = f"https://docs.google.com/spreadsheets/d/{sid}"
    print(f"\nDone! Spreadsheet URL:\n  {url}")


if __name__ == "__main__":
    main()
