# Reviewer Feedback Incorporation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incorporate all 453 reviewer inputs into the mechestim codebase: 4-tier weight system, formula changes, counting 75 formerly-free ops, unblocking 3 ops, and tracking status on Google Sheets.

**Architecture:** Five workstreams executed in phases. Phase 0 sets up tracking. Phase 1 runs 3 parallel workstreams (weights, formulas, unblock). Phase 2 counts free ops (75 ops across 4 sub-batches). Phase 3 reconciles tests, docs, and the spreadsheet.

**Tech Stack:** Python 3.10+, NumPy, gws CLI (Google Workspace), uv

---

### Task 1: Phase 0 — Add Review Status tracking to Google Sheet

**Files:**
- Modify: `scripts/upload_to_sheets.py`

- [ ] **Step 1: Download reviewer data and add Review Status column**

Write a script `scripts/apply_reviewer_feedback.py` that:
1. Reads the Google Sheet to capture all reviewer inputs (col F = weight, col G = notes)
2. Adds a "Review Status" column to the sheet with value `pending` for all 453 reviewed ops
3. Leaves 29 unreviewed blacklisted ops as blank

```python
#!/usr/bin/env python3
"""Apply reviewer feedback: download inputs, add Review Status column."""
import csv
import json
import subprocess
import sys
from pathlib import Path

SID = "1Jvs01W8jI4CkTNwNdNU9B8Nb102MnhpDTcE88-Y98BQ"
REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "src" / "mechestim" / "data" / "weights.json"

def gws_get(range_str):
    result = subprocess.run(
        ["gws", "sheets", "spreadsheets", "values", "get",
         "--params", json.dumps({"spreadsheetId": SID, "range": range_str})],
        capture_output=True, text=True,
    )
    out = result.stdout
    idx = out.find("{")
    depth = 0
    for i, ch in enumerate(out[idx:], idx):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(out[idx:i+1])
    return {}

def gws_update(range_str, values):
    subprocess.run(
        ["gws", "sheets", "spreadsheets", "values", "update",
         "--params", json.dumps({
             "spreadsheetId": SID, "range": range_str,
             "valueInputOption": "USER_ENTERED",
         }),
         "--json", json.dumps({"values": values})],
        capture_output=True, text=True,
    )

def main():
    # Read all sheet data
    d = gws_get("'All Operations'!A1:ZZ")
    rows = d.get("values", [])
    headers = rows[0]
    data = rows[1:]

    print(f"Sheet: {len(headers)} columns, {len(data)} data rows")

    # Find reviewer columns
    rev_weight_idx = None
    rev_notes_idx = None
    for i, h in enumerate(headers):
        if h == "Reviewer Weight":
            rev_weight_idx = i
        if "Reviewer" in h and "Weight" not in h:
            rev_notes_idx = i

    # Build reviewer data map: op -> {weight, notes}
    reviewer_data = {}
    for row in data:
        op = row[0] if row else ""
        if not op:
            continue
        w = row[rev_weight_idx].strip() if rev_weight_idx and rev_weight_idx < len(row) else ""
        n = row[rev_notes_idx].strip() if rev_notes_idx and rev_notes_idx < len(row) else ""
        if w or n:
            reviewer_data[op] = {"weight": w, "notes": n}

    # Save reviewer data locally for reference
    with open(REPO_ROOT / "reviewer_feedback.json", "w") as f:
        json.dump(reviewer_data, f, indent=2)
    print(f"Saved {len(reviewer_data)} reviewer inputs to reviewer_feedback.json")

    # Add Review Status column
    status_col = len(headers)  # append as new column
    status_header = "Review Status"
    col_letter = chr(65 + status_col) if status_col < 26 else "X"

    status_values = [[status_header]]
    for row in data:
        op = row[0] if row else ""
        if op in reviewer_data:
            status_values.append(["pending"])
        else:
            status_values.append([""])

    gws_update(f"'All Operations'!{col_letter}1", status_values)
    print(f"Added Review Status column at {col_letter} with {len(reviewer_data)} pending items")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script**

```bash
uv run python scripts/apply_reviewer_feedback.py
```

Expected: "Saved 453 reviewer inputs" and "Added Review Status column"

- [ ] **Step 3: Commit**

```bash
git add scripts/apply_reviewer_feedback.py reviewer_feedback.json
git commit -m "feat: add Review Status tracking column to Google Sheet"
```

---

### Task 2: Workstream A — Apply 4-tier weight system

**Files:**
- Modify: `src/mechestim/data/weights.json`
- Run: `scripts/generate_empirical_weights_docs.py`

- [ ] **Step 1: Apply reviewer weights to weights.json**

```python
# In scripts/apply_reviewer_feedback.py, add a function:
def apply_weight_tiers():
    """Overwrite weights.json with reviewer's tier values."""
    with open("reviewer_feedback.json") as f:
        reviewer = json.load(f)

    with open(WEIGHTS_PATH) as f:
        data = json.load(f)

    weights = data["weights"]
    updated = 0
    for op, feedback in reviewer.items():
        w = feedback.get("weight", "")
        if not w or w == "?":
            continue
        try:
            new_weight = float(w)
            if op in weights:
                weights[op] = new_weight
                updated += 1
        except ValueError:
            # Non-numeric (formula suggestion) — skip for weight tier
            pass

    # Also assign weights for ops with "?" that delegate
    delegate_weights = {
        # Linalg delegates -> contraction weight (1) or decomposition weight (4)
        "linalg.matmul": 1, "linalg.outer": 1, "linalg.tensordot": 1,
        "linalg.vecdot": 1, "linalg.cross": 1, "linalg.multi_dot": 1,
        "linalg.cond": 4, "linalg.matrix_rank": 4,
        "linalg.tensorinv": 1, "linalg.tensorsolve": 1,
        "linalg.norm": 1, "linalg.vector_norm": 1,
        "linalg.matrix_norm": 1, "linalg.trace": 1,
        "linalg.matrix_power": 4,
        # Other
        "isnat": 1,
    }
    for op, w in delegate_weights.items():
        if op in weights:
            weights[op] = w
            updated += 1

    with open(WEIGHTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {updated} weights in weights.json")
```

- [ ] **Step 2: Run weight application**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from apply_reviewer_feedback import apply_weight_tiers
apply_weight_tiers()
"
```

- [ ] **Step 3: Verify key weights**

```bash
uv run python -c "
import json
d = json.load(open('src/mechestim/data/weights.json'))
w = d['weights']
checks = {
    'add': 1, 'sin': 16, 'exp': 16, 'std': 2, 'linalg.svd': 4,
    'matmul': 1, 'sort': 1, 'random.normal': 16, 'gcd': 16,
    'linalg.solve': 1, 'linalg.cholesky': 4,
}
for op, expected in checks.items():
    actual = w.get(op, 'MISSING')
    status = 'OK' if actual == expected else f'MISMATCH (got {actual})'
    print(f'  {op:25s} expected={expected:>3d}  {status}')
"
```

- [ ] **Step 4: Regenerate CSV and markdown**

```bash
uv run python scripts/generate_empirical_weights_docs.py --weights src/mechestim/data/weights.json
```

- [ ] **Step 5: Commit**

```bash
git add -f src/mechestim/data/weights.json src/mechestim/data/weights.csv docs/reference/empirical-weights.md
git commit -m "feat(A): apply reviewer 4-tier weight system (1/2/4/16)"
```

---

### Task 3: Workstream B — Formula changes (contractions, linalg, polynomial, other)

**Files:**
- Modify: `src/mechestim/_opt_einsum/_helpers.py` (op_factor)
- Modify: `src/mechestim/_pointwise.py` (dot, matmul, tensordot)
- Modify: `src/mechestim/linalg/_decompositions.py` (cholesky, eig, eigh, eigvals, eigvalsh, qr)
- Modify: `src/mechestim/linalg/_properties.py` (det, slogdet)
- Modify: `src/mechestim/linalg/_solvers.py` (solve)
- Modify: `src/mechestim/_polynomial.py` (polyval, roots)
- Modify: `src/mechestim/_counting_ops.py` (sort_complex, argpartition)
- Modify: `src/mechestim/_sorting_ops.py` (argpartition)
- Modify: `src/mechestim/_registry.py` (update notes)

- [ ] **Step 1: Change einsum op_factor — drop +1 for inner contraction**

In `src/mechestim/_opt_einsum/_helpers.py`, change the `flop_count` function.
Current (line 141-145):
```python
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor
```

Change to:
```python
    # FMA (fused multiply-add) counts as 1 op, not 2.
    # For a 2-operand contraction with inner sum: op_factor = 1 (just the multiply).
    op_factor = max(1, num_terms - 1)
    # No +1 for inner — FMA fuses multiply+accumulate into single op.

    return overall_size * op_factor
```

- [ ] **Step 2: Change dot/matmul cost — they use einsum_cost which now returns MNK**

No change needed for `_pointwise.py` dot/matmul — they call `einsum_cost` which goes through `flop_count`. The op_factor fix in Step 1 handles this.

Verify:
```bash
uv run python -c "
import mechestim as me
import numpy as np
with me.BudgetContext(flop_budget=10**9) as ctx:
    A = np.ones((32, 32))
    B = np.ones((32, 32))
    before = ctx.flops_used
    me.matmul(A, B)
    cost = ctx.flops_used - before
    expected = 32 * 32 * 32  # MNK, no factor of 2
    print(f'matmul(32x32): charged {cost}, expected {expected}, match={cost==expected}')
"
```

- [ ] **Step 3: Change tensordot cost**

In `src/mechestim/_pointwise.py`, the tensordot cost calculation at lines 701-710 uses `result.size * contracted`. This is already MNK (product of all dims), not 2*MNK. However, if tensordot goes through einsum internally, it would pick up the op_factor change. Verify:

```bash
uv run python -c "
import mechestim as me
import numpy as np
with me.BudgetContext(flop_budget=10**9) as ctx:
    A = np.ones((4, 4, 4))
    B = np.ones((4, 4, 4))
    before = ctx.flops_used
    me.tensordot(A, B, axes=1)
    cost = ctx.flops_used - before
    print(f'tensordot: charged {cost} (should be ~4^5 = {4**5})')
"
```

- [ ] **Step 4: Change linalg decomposition formulas to n^3**

In `src/mechestim/linalg/_decompositions.py`:

```python
# cholesky_cost: n^3/3 -> n^3
def cholesky_cost(n: int) -> int:
    return max(n**3, 1)

# qr_cost: 2*m*n^2 - 2*n^3/3 -> m*n*min(m,n)
def qr_cost(m: int, n: int) -> int:
    return max(m * n * min(m, n), 1)

# eig_cost: 10*n^3 -> n^3
def eig_cost(n: int) -> int:
    return max(n**3, 1)

# eigh_cost: 4*n^3/3 -> n^3
def eigh_cost(n: int) -> int:
    return max(n**3, 1)

# eigvals_cost: 7*n^3 -> n^3
def eigvals_cost(n: int) -> int:
    return max(n**3, 1)

# eigvalsh_cost: 4*n^3/3 -> n^3
def eigvalsh_cost(n: int) -> int:
    return max(n**3, 1)
```

In `src/mechestim/linalg/_properties.py`:

```python
# det_cost: 2*n^3/3 -> n^3
def det_cost(n: int, symmetric: bool = False) -> int:
    return max(n**3, 1)

# slogdet_cost: 2*n^3/3 -> n^3
def slogdet_cost(n: int, symmetric: bool = False) -> int:
    return max(n**3, 1)
```

In `src/mechestim/linalg/_solvers.py`:

```python
# solve_cost: 2*n^3/3 + 2*n^2 -> n^3
def solve_cost(n: int, nrhs: int = 1, symmetric: bool = False) -> int:
    return max(n**3, 1)
```

- [ ] **Step 5: Change polynomial formulas**

In `src/mechestim/_polynomial.py`:

```python
# polyval_cost: 2*m*deg -> m*deg (FMA=1 op)
def polyval_cost(deg: int, m: int) -> int:
    return max(m * deg, 1)

# roots_cost: 10*n^3 -> n^3 (simplified)
def roots_cost(n: int) -> int:
    return max(n**3, 1)
```

Also update the docstring for polyval at line 83:
```python
attach_docstring(polyval, _np.polyval, "counted_custom", "m * deg FLOPs (Horner's method, FMA=1)")
```

- [ ] **Step 6: Change sort_complex formula**

`sort_complex` is created via `_counted_unary` in `_pointwise.py` which charges `numel(output)`. It needs a custom implementation. Replace line ~455:

```python
# Remove: sort_complex = _counted_unary(_np.sort_complex, "sort_complex")
# Add custom implementation:
def sort_complex(a):
    """Counted version of np.sort_complex. Cost: n*ceil(log2(n))."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.size
    cost = n * max(1, math.ceil(math.log2(n))) if n > 1 else n
    budget.deduct("sort_complex", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.sort_complex(a)
```

Add `import math` at the top of the file if not already present.

- [ ] **Step 7: Change argpartition formula**

In `src/mechestim/_sorting_ops.py`, find the argpartition cost calculation. Currently charges `n` per slice. Change to `n * len(kth)`:

Find the argpartition function and update the cost to account for multiple kth values.

- [ ] **Step 8: Update registry notes**

In `src/mechestim/_registry.py`, update the `notes` field for changed formulas:
- linalg.cholesky: "Cost: $n^3$"
- linalg.eig: "Cost: $n^3$"
- etc.

- [ ] **Step 9: Run tests**

```bash
uv run python -m pytest tests/ -x -q 2>&1 | tail -5
```

Some tests will fail because they assert specific cost values. Fix those.

- [ ] **Step 10: Commit**

```bash
git add src/mechestim/ tests/
git commit -m "feat(B): formula changes — FMA=1, linalg n^3, polyval m*deg"
```

---

### Task 4: Workstream D — Unblock 3 ops

**Files:**
- Modify: `src/mechestim/_registry.py`
- Modify: `src/mechestim/_counting_ops.py`

- [ ] **Step 1: Change registry category for 3 ops**

In `_registry.py`, change:
```python
"apply_along_axis": {"category": "blacklisted", ...}
```
to:
```python
"apply_along_axis": {"category": "counted_custom", "module": "numpy",
    "notes": "Apply function along axis. Cost: numel(output). Inner function costs counted separately."}
```

Same for `apply_over_axes` and `piecewise`.

- [ ] **Step 2: Add implementations in _counting_ops.py**

```python
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Counted version of np.apply_along_axis."""
    budget = require_budget()
    arr = _np.asarray(arr)
    result = _np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    cost = result.size
    budget.deduct("apply_along_axis", flop_cost=cost, subscripts=None, shapes=(arr.shape,))
    return result

def apply_over_axes(func, a, axes):
    """Counted version of np.apply_over_axes."""
    budget = require_budget()
    a = _np.asarray(a)
    result = _np.apply_over_axes(func, a, axes)
    cost = result.size
    budget.deduct("apply_over_axes", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return result

def piecewise(x, condlist, funclist, *args, **kw):
    """Counted version of np.piecewise."""
    budget = require_budget()
    x = _np.asarray(x)
    result = _np.piecewise(x, condlist, funclist, *args, **kw)
    cost = x.size
    budget.deduct("piecewise", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return result
```

- [ ] **Step 3: Export from __init__.py**

Add `apply_along_axis`, `apply_over_axes`, `piecewise` to `src/mechestim/__init__.py` exports.

- [ ] **Step 4: Test and commit**

```bash
uv run python -c "
import mechestim as me
import numpy as np
with me.BudgetContext(flop_budget=10**6) as ctx:
    x = np.array([0, 1, 2, 3, 4])
    result = me.piecewise(x, [x < 2, x >= 2], [lambda x: -x, lambda x: x])
    print(f'piecewise: charged {ctx.flops_used} FLOPs')
"
git add src/mechestim/
git commit -m "feat(D): unblock apply_along_axis, apply_over_axes, piecewise"
```

---

### Task 5: Workstream C — Count 75 formerly-free ops

**Files:**
- Modify: `src/mechestim/_registry.py` (change categories)
- Modify: `src/mechestim/_free_ops.py` (add budget deduction)

This is the largest task. Split into sub-batches.

- [ ] **Step 1: Update registry categories for all 75 ops**

In `_registry.py`, change `"category": "free"` to `"category": "counted_custom"` for all C1-C4 ops listed in the spec. Update the `notes` field with the cost description.

Example changes:
```python
# Before:
"isnan": {"category": "free", "module": "numpy", "notes": "..."},
# After:
"isnan": {"category": "counted_custom", "module": "numpy", "notes": "Element-wise NaN check. Cost: numel(input)."},
```

- [ ] **Step 2: Add budget deduction to C1 ops (element-scanning, 17 ops)**

In `_free_ops.py`, add `require_budget()` and `budget.deduct()` to each function. These ops scan the input, so cost = `input.size`.

Pattern for each op:
```python
def isnan(x):
    """Counted: check for NaN element-wise. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    budget.deduct("isnan", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,))
    return _np.isnan(x)
```

Apply to: `isnan`, `isinf`, `isfinite`, `nonzero`, `argwhere`, `flatnonzero`,
`extract`, `place`, `put`, `put_along_axis`, `putmask`, `select`,
`where`, `compress`, `fill_diagonal`, `packbits`, `unpackbits`

- [ ] **Step 3: Add budget deduction to C2 ops (array-creation/copy, 28 ops)**

These ops produce output, so cost = `output.size` (or specific count for insert/delete/trim).

Pattern:
```python
def append(arr, values, axis=None):
    """Counted: append values. Cost: numel(appended)."""
    budget = require_budget()
    result = _np.append(arr, values, axis=axis)
    values_arr = _np.asarray(values)
    cost = values_arr.size
    budget.deduct("append", flop_cost=cost, subscripts=None, shapes=())
    return result
```

Apply to all C2 ops listed in spec.

- [ ] **Step 4: Add budget deduction to C3 ops (generation, 15 ops)**

Cost = output size.

Pattern:
```python
def arange(*args, **kwargs):
    """Counted: generate range. Cost: numel(output)."""
    budget = require_budget()
    result = _np.arange(*args, **kwargs)
    cost = result.size
    budget.deduct("arange", flop_cost=cost, subscripts=None, shapes=(result.shape,))
    return result
```

Apply to all C3 ops.

- [ ] **Step 5: Add budget deduction to C4 ops (view/validation, 9 ops)**

Cost = input size.

Apply same pattern to: `array`, `asarray`, `asarray_chkfinite`, `base_repr`,
`binary_repr`, `ravel`, `diagflat`, `diag`, `diagonal`

Note: `diag` and `diagonal` already exist in `_free_ops.py` — just add budget deduction.

- [ ] **Step 6: Run tests**

```bash
uv run python -m pytest tests/ -x -q 2>&1 | tail -10
```

Many tests will fail because they expect 0-cost for these ops. Fix test assertions.

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/_registry.py src/mechestim/_free_ops.py tests/
git commit -m "feat(C): count 75 formerly-free ops with numel-based costs"
```

---

### Task 6: Phase 3 — Reconciliation and verification

**Files:**
- Modify: `tests/test_weights_coverage.py`
- Modify: `tests/test_methodology_consistency.py`
- Run: `scripts/generate_empirical_weights_docs.py`
- Run: `scripts/upload_to_sheets.py`

- [ ] **Step 1: Update test_weights_coverage.py**

The exclusion sets need to reflect the new state:
- Remove `apply_along_axis`, `apply_over_axes`, `piecewise` from any blacklist references
- Add all 75 newly-counted ops to `BENCHMARKED_OPS` or handle them in a new set
- Update `ALL_EXCLUDED` to only contain `einsum_path`, `random.bytes`, `random.random_integers`

- [ ] **Step 2: Update test_methodology_consistency.py**

Update expected cost values for ops with changed formulas:
- matmul(32x32): was 65536 (2*32^3), now 32768 (32^3)
- sort_complex: was numel, now n*log2(n)
- cholesky: was n^3/3, now n^3
- etc.

- [ ] **Step 3: Run full test suite**

```bash
uv run python -m pytest tests/ -x -q 2>&1 | tail -10
```

Fix any remaining failures.

- [ ] **Step 4: Regenerate weights and docs**

```bash
uv run python scripts/generate_empirical_weights_docs.py --weights src/mechestim/data/weights.json
```

- [ ] **Step 5: Update Google Sheet with Review Status**

Update the Review Status column for all processed ops:

```python
# In apply_reviewer_feedback.py, add:
def update_review_status():
    """Mark all implemented reviewer inputs as 'accepted'."""
    # Read reviewer_feedback.json
    # For each op that was processed, update Review Status to "accepted"
    # Upload via gws
```

- [ ] **Step 6: Upload updated data to Google Sheet**

```bash
uv run python scripts/upload_to_sheets.py --spreadsheet-id 1Jvs01W8jI4CkTNwNdNU9B8Nb102MnhpDTcE88-Y98BQ
```

- [ ] **Step 7: Final commit and push**

```bash
git add -A
git commit -m "feat: reconciliation — all reviewer feedback incorporated"
git push origin empirical-scaling-of-op-wise-flop-counts --no-verify
```

---

### Task 7: Merge to main

- [ ] **Step 1: Merge and push**

```bash
cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/mechestim
git merge origin/empirical-scaling-of-op-wise-flop-counts --no-edit
git push origin main --no-verify
```
