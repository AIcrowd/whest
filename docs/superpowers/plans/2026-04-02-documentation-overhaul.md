# Documentation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite mechestim docs to match the network-estimation-challenge-internal reference repo style — persona-driven landing page, getting-started guides, how-to pages, concepts, architecture docs, operation audit, troubleshooting, and runnable examples.

**Architecture:** MkDocs + Material theme with mkdocstrings for API reference. Diataxis-inspired structure (tutorials / how-to / concepts / reference). All examples use `uv run`. Visual identity from reference repo.

**Tech Stack:** MkDocs Material, mkdocstrings, Python, uv

---

## File Structure

**New files to create:**
- `docs/assets/logo/logo.png` — logo from reference repo
- `docs/getting-started/installation.md` — install + verify
- `docs/getting-started/first-budget.md` — first BudgetContext tutorial
- `docs/how-to/migrate-from-numpy.md` — NumPy migration guide
- `docs/how-to/use-einsum.md` — einsum patterns and costs
- `docs/how-to/exploit-symmetry.md` — symmetry savings
- `docs/how-to/use-linalg.md` — linear algebra operations
- `docs/how-to/plan-your-budget.md` — cost query functions
- `docs/how-to/debug-budget-overruns.md` — reading diagnostics
- `docs/concepts/flop-counting-model.md` — how FLOP counting works
- `docs/concepts/operation-categories.md` — free vs counted vs unsupported
- `docs/architecture/client-server.md` — client-server model overview
- `docs/architecture/docker.md` — running with Docker
- `docs/reference/operation-audit.md` — full operation registry tables
- `docs/troubleshooting/common-errors.md` — symptom/fix pairs
- `examples/01_basic_usage.py` — basic BudgetContext usage
- `examples/02_einsum_patterns.py` — common einsum patterns
- `examples/03_symmetry_savings.py` — symmetry detection
- `examples/04_svd_usage.py` — truncated SVD
- `examples/05_budget_planning.py` — cost queries and planning

**Files to modify:**
- `mkdocs.yml` — new nav, theme, logo, extensions
- `docs/index.md` — rewrite as persona-driven landing page
- `README.md` — update install to uv, add docs link

**Files to delete:**
- `docs/quickstart.md` — replaced by getting-started/ pages

---

### Task 1: Copy Logo and Update mkdocs.yml Theme

**Files:**
- Create: `docs/assets/logo/logo.png`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Create docs/assets/logo directory and copy logo**

```bash
mkdir -p docs/assets/logo
cp /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/circuit-estimation/circuit-estimation-mvp/assets/logo/logo.png docs/assets/logo/logo.png
```

- [ ] **Step 2: Update mkdocs.yml with new theme, nav, and extensions**

Replace the entire `mkdocs.yml` with:

```yaml
site_name: mechestim
site_description: NumPy-compatible math primitives with FLOP counting
repo_url: https://github.com/AIcrowd/mechestim

theme:
  name: material
  logo: assets/logo/logo.png
  favicon: assets/logo/logo.png
  palette:
    scheme: default
    primary: indigo
    accent: indigo
  features:
    - navigation.sections
    - navigation.expand
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Your First Budget: getting-started/first-budget.md
  - How-To Guides:
    - Migrate from NumPy: how-to/migrate-from-numpy.md
    - Use Einsum: how-to/use-einsum.md
    - Exploit Symmetry: how-to/exploit-symmetry.md
    - Use Linear Algebra: how-to/use-linalg.md
    - Plan Your Budget: how-to/plan-your-budget.md
    - Debug Budget Overruns: how-to/debug-budget-overruns.md
  - Concepts:
    - FLOP Counting Model: concepts/flop-counting-model.md
    - Operation Categories: concepts/operation-categories.md
  - Architecture:
    - Client-Server Model: architecture/client-server.md
    - Running with Docker: architecture/docker.md
  - API Reference:
    - Counted Operations: api/counted-ops.md
    - Free Operations: api/free-ops.md
    - Budget: api/budget.md
    - FLOP Cost Query: api/flops.md
    - Errors: api/errors.md
  - Operation Audit: reference/operation-audit.md
  - Troubleshooting: troubleshooting/common-errors.md
  - Changelog: changelog.md

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - admonition
  - pymdownx.details
  - tables
```

- [ ] **Step 3: Commit**

```bash
git add docs/assets/logo/logo.png mkdocs.yml
git commit -m "docs: update mkdocs theme with logo and new navigation"
```

---

### Task 2: Rewrite index.md as Persona-Driven Landing Page

**Files:**
- Modify: `docs/index.md`

- [ ] **Step 1: Replace docs/index.md with persona-driven landing page**

```markdown
<img src="assets/logo/logo.png" alt="mechestim logo" style="height: 80px;">

# mechestim

**NumPy-compatible math primitives with analytical FLOP counting.**

Pick the path that matches what you need right now.

## 🚀 I want to get started

- [Installation](./getting-started/installation.md)
- [Your First Budget](./getting-started/first-budget.md)

## 🛠 Something isn't working

- [Common Errors](./troubleshooting/common-errors.md)
- [Error Reference](./api/errors.md)

## 📈 I want to write efficient code with mechestim

- [Migrate from NumPy](./how-to/migrate-from-numpy.md)
- [Use Einsum](./how-to/use-einsum.md)
- [Exploit Symmetry](./how-to/exploit-symmetry.md)
- [Use Linear Algebra](./how-to/use-linalg.md)
- [Plan Your Budget](./how-to/plan-your-budget.md)
- [Debug Budget Overruns](./how-to/debug-budget-overruns.md)

## 🧠 I want to understand how it works

- [FLOP Counting Model](./concepts/flop-counting-model.md) — how costs are computed, why FLOPs
- [Operation Categories](./concepts/operation-categories.md) — free vs counted vs unsupported

## 🏗 I want to understand the sandboxed architecture

- [Client-Server Model](./architecture/client-server.md) — why it exists, how it works
- [Running with Docker](./architecture/docker.md) — local setup with Docker

## Quick example

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    W = me.zeros((256, 256))
    x = me.zeros((256,))
    h = me.einsum('ij,j->i', W, x)
    h = me.maximum(h, 0)
    print(budget.summary())
```

## Installation

```bash
uv add git+https://github.com/AIcrowd/mechestim.git
```

## Full Taxonomy

- **Getting Started:** [Installation](./getting-started/installation.md), [Your First Budget](./getting-started/first-budget.md)
- **How-To:** [Migrate from NumPy](./how-to/migrate-from-numpy.md), [Use Einsum](./how-to/use-einsum.md), [Exploit Symmetry](./how-to/exploit-symmetry.md), [Use Linear Algebra](./how-to/use-linalg.md), [Plan Your Budget](./how-to/plan-your-budget.md), [Debug Budget Overruns](./how-to/debug-budget-overruns.md)
- **Concepts:** [FLOP Counting Model](./concepts/flop-counting-model.md), [Operation Categories](./concepts/operation-categories.md)
- **Architecture:** [Client-Server Model](./architecture/client-server.md), [Running with Docker](./architecture/docker.md)
- **API Reference:** [Counted Ops](./api/counted-ops.md), [Free Ops](./api/free-ops.md), [Budget](./api/budget.md), [FLOP Cost Query](./api/flops.md), [Errors](./api/errors.md)
- **Reference:** [Operation Audit](./reference/operation-audit.md)
- **Troubleshooting:** [Common Errors](./troubleshooting/common-errors.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/index.md
git commit -m "docs: rewrite index.md as persona-driven landing page"
```

---

### Task 3: Create Getting Started — Installation

**Files:**
- Create: `docs/getting-started/installation.md`

- [ ] **Step 1: Create docs/getting-started directory**

```bash
mkdir -p docs/getting-started
```

- [ ] **Step 2: Write installation.md**

```markdown
# Installation

## When to use this page

Use this page when setting up mechestim for the first time.

## Install as a dependency

```bash
uv add git+https://github.com/AIcrowd/mechestim.git
```

## Install for development

```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
```

## ✅ Verify installation

```bash
uv run python -c "import mechestim as me; print(me.__version__)"
```

## 🔍 What you'll see

```
0.2.0
```

If you see a version number, mechestim is installed correctly.

## ⚠️ Common pitfalls

**Symptom:** `ImportError: numpy version mismatch`

**Fix:** mechestim requires NumPy >=2.1.0,<2.2.0. Using `uv` handles this automatically. If you installed manually, check your NumPy version:

```bash
uv run python -c "import numpy; print(numpy.__version__)"
```

## 📎 Related pages

- [Your First Budget](./first-budget.md) — run your first FLOP-counted computation
```

- [ ] **Step 3: Commit**

```bash
git add docs/getting-started/installation.md
git commit -m "docs: add installation guide"
```

---

### Task 4: Create Getting Started — Your First Budget

**Files:**
- Create: `docs/getting-started/first-budget.md`

- [ ] **Step 1: Write first-budget.md**

```markdown
# Your First Budget

## When to use this page

Use this page after installing mechestim to run your first FLOP-counted computation.

## Prerequisites

- [Installation](./installation.md)

## Do this now

Save this as `first_budget.py`:

```python
import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs) -- tensor creation
    W = me.ones((256, 256))
    x = me.ones((256,))

    # Counted operations -- each deducts from the FLOP budget
    h = me.einsum('ij,j->i', W, x)      # matrix-vector multiply
    h = me.maximum(h, 0)                 # ReLU activation
    result = me.sum(h)                   # sum all elements

    # Inspect your budget
    print(budget.summary())
```

Run it:

```bash
uv run python first_budget.py
```

## 🔍 What you'll see

```
mechestim 0.2.0 (numpy 2.1.3 backend) | budget: 1.00e+07 FLOPs
mechestim FLOP Budget Summary
==============================
  Total budget:      10,000,000
  Used:                  65,792  ( 0.7%)
  Remaining:          9,934,208  (99.3%)

  By operation:
    einsum              65,536  (99.6%)  [1 call]
    maximum                256  ( 0.4%)  [1 call]
    sum                    256  ( 0.4%)  [1 call]
```

**Reading the output:**

- **Total budget:** the FLOP limit you set
- **Used / Remaining:** how much of the budget has been consumed
- **By operation:** breakdown of costs per operation type and call count

## ⚠️ Common pitfalls

**Symptom:** `NoBudgetContextError: No active BudgetContext`

**Fix:** All counted operations must run inside a `with me.BudgetContext(...)` block.

**Symptom:** `BudgetExhaustedError`

**Fix:** Your operations exceed the budget. Increase `flop_budget` or reduce computation.

## 📎 Related pages

- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — convert existing NumPy code to mechestim
- [Plan Your Budget](../how-to/plan-your-budget.md) — query operation costs before executing
```

- [ ] **Step 2: Commit**

```bash
git add docs/getting-started/first-budget.md
git commit -m "docs: add your-first-budget tutorial"
```

---

### Task 5: Create How-To — Migrate from NumPy

**Files:**
- Create: `docs/how-to/migrate-from-numpy.md`

- [ ] **Step 1: Create docs/how-to directory**

```bash
mkdir -p docs/how-to
```

- [ ] **Step 2: Write migrate-from-numpy.md**

```markdown
# Migrate from NumPy

## When to use this page

Use this page when converting existing NumPy code to mechestim.

## Prerequisites

- [Installation](../getting-started/installation.md)
- [Your First Budget](../getting-started/first-budget.md)

## The basics

Change your import and wrap computation in a BudgetContext:

**Before (NumPy):**

```python
import numpy as np

W = np.random.randn(256, 256)
x = np.random.randn(256)
h = np.dot(W, x)
h = np.maximum(h, 0)
```

**After (mechestim):**

```python
import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    W = me.random.randn(256, 256)
    x = me.random.randn(256)
    h = me.dot(W, x)
    h = me.maximum(h, 0)
```

## What stays the same

- All arrays are plain `numpy.ndarray` — no custom tensor class
- Function signatures match NumPy for supported operations
- Broadcasting rules are identical
- Array indexing, slicing, and assignment work normally

## What changes

| NumPy | mechestim | Notes |
|-------|-----------|-------|
| `import numpy as np` | `import mechestim as me` | Drop-in replacement |
| Call ops anywhere | Wrap in `BudgetContext` | Required for counted ops |
| `np.linalg.svd(A)` | `me.linalg.svd(A, k=10)` | Truncated SVD with explicit `k` |
| All NumPy ops available | Subset available | Unsupported ops raise `AttributeError` |
| No cost tracking | Automatic FLOP counting | Every counted op deducts from budget |

## ⚠️ Common pitfalls

**Symptom:** `AttributeError: module 'mechestim' has no attribute 'fft'`

**Fix:** Not all NumPy operations are supported. See [Operation Categories](../concepts/operation-categories.md) for the full list. The error message includes guidance on alternatives.

**Symptom:** Using `np.linalg.svd` instead of `me.linalg.svd`

**Fix:** If you import NumPy alongside mechestim, make sure to use `me.` for operations you want counted. Operations called through `np.` bypass FLOP counting entirely.

## 📎 Related pages

- [Operation Categories](../concepts/operation-categories.md) — what's supported and what isn't
- [Operation Audit](../reference/operation-audit.md) — full list of all operations
```

- [ ] **Step 3: Commit**

```bash
git add docs/how-to/migrate-from-numpy.md
git commit -m "docs: add NumPy migration guide"
```

---

### Task 6: Create How-To — Use Einsum

**Files:**
- Create: `docs/how-to/use-einsum.md`

- [ ] **Step 1: Write use-einsum.md**

```markdown
# Use Einsum

## When to use this page

Use this page to understand `me.einsum` — the core computation primitive in mechestim.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Common patterns

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    A = me.ones((256, 256))
    B = me.ones((256, 256))
    x = me.ones((256,))

    # Matrix-vector multiply: cost = m × k
    y = me.einsum('ij,j->i', A, x)           # 256 × 256 = 65,536 FLOPs

    # Matrix multiply: cost = m × k × n
    C = me.einsum('ij,jk->ik', A, B)         # 256 × 256 × 256 = 16,777,216 FLOPs

    # Outer product: cost = i × j
    outer = me.einsum('i,j->ij', x, x)       # 256 × 256 = 65,536 FLOPs

    # Trace: cost = i
    tr = me.einsum('ii->', A)                 # 256 FLOPs

    # Batched matmul: cost = b × m × k × n
    batch = me.ones((4, 256, 256))
    out = me.einsum('bij,bjk->bik', batch, batch)  # 4 × 256 × 256 × 256 FLOPs

    print(budget.summary())
```

## Cost formula

The FLOP cost of `me.einsum` is the product of all index dimensions in the subscript string:

```
cost = product of all unique index sizes
```

For `'ij,jk->ik'` with shapes `(256, 256)` and `(256, 256)`:
- Indices: i=256, j=256, k=256
- Cost: 256 × 256 × 256 = 16,777,216

## me.dot and me.matmul

`me.dot(A, B)` and `me.matmul(A, B)` are equivalent to the corresponding einsum and have the same FLOP cost.

## ⚠️ Common pitfalls

**Symptom:** Unexpectedly high FLOP cost

**Fix:** Check all index dimensions. A subscript like `'ijkl,jklm->im'` multiplies all five dimension sizes together. Use `me.flops.einsum_cost()` to preview costs before executing.

## 📎 Related pages

- [Exploit Symmetry](./exploit-symmetry.md) — reduce einsum costs with repeated operands
- [Plan Your Budget](./plan-your-budget.md) — query costs before executing
```

- [ ] **Step 2: Commit**

```bash
git add docs/how-to/use-einsum.md
git commit -m "docs: add einsum usage guide"
```

---

### Task 7: Create How-To — Exploit Symmetry

**Files:**
- Create: `docs/how-to/exploit-symmetry.md`

- [ ] **Step 1: Write exploit-symmetry.md**

```markdown
# Exploit Symmetry Savings

## When to use this page

Use this page to halve (or more) the FLOP cost of einsum operations that use the same array multiple times.

## Prerequisites

- [Use Einsum](./use-einsum.md)

## How it works

When you pass the **same Python object** as multiple operands to `me.einsum`, mechestim detects this automatically and divides the FLOP cost by the symmetry factor.

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    x = me.ones((10, 256))
    A = me.ones((10, 10))

    # x is passed TWICE (same object) -- cost is divided by 2
    result = me.einsum('ai,bi,ab->', x, x, A)
    print(f"Cost with symmetry: {budget.flops_used:,}")     # 12,800

with me.BudgetContext(flop_budget=10**8) as budget:
    x = me.ones((10, 256))
    y = me.ones((10, 256))  # different object, same values
    A = me.ones((10, 10))

    # x and y are DIFFERENT objects -- no symmetry savings
    result = me.einsum('ai,bi,ab->', x, y, A)
    print(f"Cost without symmetry: {budget.flops_used:,}")  # 25,600
```

## Symmetry detection rules

- **Same object** (`id(x) == id(y)`): symmetry detected, cost reduced
- **Copy** (`y = x.copy()`): different object, no savings
- **Slice** (`y = x[:]`): different object, no savings
- **Same values, different object**: no savings — mechestim checks object identity, not value equality

mechestim validates that the arrays are actually numerically equal (within tolerance) when symmetry is detected. If they differ, `SymmetryError` is raised.

## Cost formula

For `n` repeated operands, the cost is divided by `n!` (n factorial):

- 2 repeated → cost / 2
- 3 repeated → cost / 6

## ⚠️ Common pitfalls

**Symptom:** Expected symmetry savings but got full cost

**Fix:** Check that you're passing the same Python object, not a copy. `x.copy()`, `x[:]`, `np.array(x)` all create new objects.

## 📎 Related pages

- [Use Einsum](./use-einsum.md) — einsum basics and patterns
```

- [ ] **Step 2: Commit**

```bash
git add docs/how-to/exploit-symmetry.md
git commit -m "docs: add symmetry savings guide"
```

---

### Task 8: Create How-To — Use Linear Algebra

**Files:**
- Create: `docs/how-to/use-linalg.md`

- [ ] **Step 1: Write use-linalg.md**

```markdown
# Use Linear Algebra

## When to use this page

Use this page to learn how to use `me.linalg` operations.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Available operations

### me.linalg.svd — Truncated SVD

Compute the top-k singular value decomposition.

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    A = me.random.randn(256, 256)

    # Truncated SVD: returns top-k singular values/vectors
    U, S, Vt = me.linalg.svd(A, k=10)

    print(f"U shape: {U.shape}")    # (256, 10)
    print(f"S shape: {S.shape}")    # (10,)
    print(f"Vt shape: {Vt.shape}")  # (10, 256)
    print(f"Cost: {budget.flops_used:,} FLOPs")  # 655,360
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | ndarray | Input matrix of shape (m, n) |
| `k` | int | Number of singular values to compute |

**Cost:** m × n × k FLOPs

**Returns:** `(U, S, Vt)` where U is (m, k), S is (k,), Vt is (k, n)

### Query cost before running

```python
cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")  # 655,360
```

## Unsupported linalg operations

Calling a NumPy linalg function that isn't available in mechestim raises an `AttributeError` with a message explaining what operations are supported:

```python
me.linalg.solve(A, b)
# AttributeError: module 'mechestim.linalg' has no attribute 'solve'.
# mechestim.linalg currently supports: svd
```

## ⚠️ Common pitfalls

**Symptom:** Using `numpy.linalg.svd` instead of `me.linalg.svd`

**Fix:** Operations called through `numpy` directly bypass FLOP counting. Always use `me.linalg.svd`.

## 📎 Related pages

- [Plan Your Budget](./plan-your-budget.md) — query SVD cost with `me.flops.svd_cost()`
- [Operation Audit](../reference/operation-audit.md) — full list of supported operations
```

- [ ] **Step 2: Commit**

```bash
git add docs/how-to/use-linalg.md
git commit -m "docs: add linear algebra guide"
```

---

### Task 9: Create How-To — Plan Your Budget

**Files:**
- Create: `docs/how-to/plan-your-budget.md`

- [ ] **Step 1: Write plan-your-budget.md**

```markdown
# Plan Your Budget

## When to use this page

Use this page to learn how to query operation costs before running them.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Cost query functions

These functions work **outside** a BudgetContext — they compute costs from shapes without executing anything.

```python
import mechestim as me

# Einsum cost
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")         # 16,777,216

# SVD cost
cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")            # 655,360

# Pointwise cost (unary/binary ops)
cost = me.flops.pointwise_cost(shape=(256, 256))
print(f"Pointwise cost: {cost:,}")      # 65,536

# Reduction cost
cost = me.flops.reduction_cost(input_shape=(256, 256))
print(f"Reduction cost: {cost:,}")      # 65,536
```

## Budget breakdown example

Plan a multi-step computation before executing:

```python
import mechestim as me

# Plan
steps = [
    ("einsum ij,j->i", me.flops.einsum_cost('ij,j->i', shapes=[(256, 256), (256,)])),
    ("ReLU (maximum)", me.flops.pointwise_cost(shape=(256,))),
    ("sum reduction", me.flops.reduction_cost(input_shape=(256,))),
]

total = sum(cost for _, cost in steps)
print(f"{'Operation':<20} {'FLOPs':>12}")
print("-" * 34)
for name, cost in steps:
    print(f"{name:<20} {cost:>12,}")
print("-" * 34)
print(f"{'Total':<20} {total:>12,}")
```

Output:

```
Operation                   FLOPs
----------------------------------
einsum ij,j->i             65,536
ReLU (maximum)                256
sum reduction                 256
----------------------------------
Total                      66,048
```

## 📎 Related pages

- [Use Einsum](./use-einsum.md) — understand einsum cost formulas
- [Debug Budget Overruns](./debug-budget-overruns.md) — diagnose after the fact
```

- [ ] **Step 2: Commit**

```bash
git add docs/how-to/plan-your-budget.md
git commit -m "docs: add budget planning guide"
```

---

### Task 10: Create How-To — Debug Budget Overruns

**Files:**
- Create: `docs/how-to/debug-budget-overruns.md`

- [ ] **Step 1: Write debug-budget-overruns.md**

```markdown
# Debug Budget Overruns

## When to use this page

Use this page when you hit a `BudgetExhaustedError` and need to find which operations are using the most FLOPs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Reading the budget summary

Call `budget.summary()` at any point inside the BudgetContext:

```python
import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    A = me.ones((256, 256))
    x = me.ones((256,))

    h = me.einsum('ij,j->i', A, x)
    h = me.exp(h)
    h = me.sum(h)

    print(budget.summary())
```

The summary shows cost per operation type, sorted by highest cost first.

## Inspecting the operation log

For per-call detail, use `budget.op_log`:

```python
for record in budget.op_log:
    print(f"{record.op_name:<16} cost={record.flop_cost:>12,}  cumulative={record.cumulative:>12,}")
```

Each `OpRecord` contains:

| Field | Description |
|-------|-------------|
| `op_name` | Operation name (e.g., `"einsum"`, `"exp"`) |
| `subscripts` | Einsum subscript string, or `None` |
| `shapes` | Tuple of input shapes |
| `flop_cost` | FLOP cost of this single call |
| `cumulative` | Running total after this call |

## ⚠️ Common pitfalls

**Symptom:** `BudgetExhaustedError` but summary shows budget was nearly full

**Fix:** The budget is checked **before** execution. The failing operation's cost is in the error message — compare it with `budget.flops_remaining`.

## 📎 Related pages

- [Plan Your Budget](./plan-your-budget.md) — predict costs before running
- [Common Errors](../troubleshooting/common-errors.md) — all error types explained
```

- [ ] **Step 2: Commit**

```bash
git add docs/how-to/debug-budget-overruns.md
git commit -m "docs: add budget debugging guide"
```

---

### Task 11: Create Concepts — FLOP Counting Model

**Files:**
- Create: `docs/concepts/flop-counting-model.md`

- [ ] **Step 1: Create docs/concepts directory**

```bash
mkdir -p docs/concepts
```

- [ ] **Step 2: Write flop-counting-model.md**

```markdown
# FLOP Counting Model

## When to use this page

Use this page to understand how mechestim counts FLOPs and why it uses analytical counting instead of runtime measurement.

## Why FLOPs instead of wall-clock time

- **Deterministic:** The same code always produces the same FLOP count, regardless of hardware
- **Hardware-independent:** A matmul costs the same FLOPs on a laptop and a server
- **Reproducible:** No variance from CPU scheduling, cache effects, or thermal throttling
- **Composable:** You can sum individual operation costs to predict total cost

## How costs are computed

mechestim computes FLOP costs **analytically from tensor shapes**, not by measuring execution time.

1. You call a counted operation (e.g., `me.einsum('ij,j->i', W, x)`)
2. mechestim computes the cost from the shapes: 256 × 256 = 65,536 FLOPs
3. The cost is checked against the remaining budget
4. If within budget: the operation executes and the cost is deducted
5. If over budget: `BudgetExhaustedError` is raised, the operation does **not** execute

## Cost formulas by category

| Category | Formula | Example |
|----------|---------|---------|
| **Einsum** | Product of all index dimensions | `'ij,jk->ik'` with (256,256) × (256,256) → 256³ |
| **Unary** (exp, log, sqrt, ...) | numel(output) | shape (256, 256) → 65,536 |
| **Binary** (add, multiply, ...) | numel(output) | shape (256, 256) → 65,536 |
| **Reduction** (sum, mean, max, ...) | numel(input) | shape (256, 256) → 65,536 |
| **SVD** | m × n × k | (256, 256, k=10) → 655,360 |
| **Dot / Matmul** | Equivalent einsum cost | (256, 256) @ (256, 256) → 256³ |
| **Free ops** | 0 | zeros, reshape, etc. |

## FLOP multiplier

The `flop_multiplier` parameter in `BudgetContext` scales all costs:

```python
with me.BudgetContext(flop_budget=10**6, flop_multiplier=2.0) as budget:
    # Every operation costs 2× its normal FLOP count
    ...
```

This is useful for experimentation or adjusting the difficulty of a budget constraint.

## 📎 Related pages

- [Operation Categories](./operation-categories.md) — which operations are free, counted, or unsupported
- [Plan Your Budget](../how-to/plan-your-budget.md) — query costs before running
```

- [ ] **Step 3: Commit**

```bash
git add docs/concepts/flop-counting-model.md
git commit -m "docs: add FLOP counting model concept page"
```

---

### Task 12: Create Concepts — Operation Categories

**Files:**
- Create: `docs/concepts/operation-categories.md`

- [ ] **Step 1: Write operation-categories.md**

```markdown
# Operation Categories

## When to use this page

Use this page to understand which operations cost FLOPs, which are free, and which are unsupported.

## Three categories

Every NumPy function falls into one of three categories in mechestim:

### 🟢 Free operations (0 FLOPs)

Operations that involve no arithmetic computation — just memory allocation, reshaping, or data movement.

**Examples:** `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `empty`, `reshape`, `transpose`, `concatenate`, `stack`, `split`, `squeeze`, `expand_dims`, `ravel`, `take`, `where`, `copy`, `astype`, `asarray`, `array_equal`

**Random operations** are also free: `me.random.randn`, `me.random.normal`, `me.random.seed`, etc.

### 🟡 Counted operations (cost > 0)

Operations that perform arithmetic. Cost is computed analytically from tensor shapes.

| Sub-category | Cost formula | Examples |
|-------------|-------------|----------|
| Unary | numel(output) | `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `ceil`, `floor` |
| Binary | numel(output) | `add`, `multiply`, `maximum`, `divide`, `power`, `subtract` |
| Reduction | numel(input) | `sum`, `mean`, `max`, `min`, `std`, `var`, `argmax`, `nansum` |
| Einsum | product of all index dims | `me.einsum(...)` |
| Dot/Matmul | equivalent einsum | `me.dot(A, B)`, `A @ B` |
| SVD | m × n × k | `me.linalg.svd(A, k=10)` |

### 🔴 Unsupported operations

Operations not in the mechestim allowlist. Calling them raises an `AttributeError` with a message explaining what's available.

```python
me.fft.fft(x)
# AttributeError: module 'mechestim.fft' has no attribute 'fft'.
# mechestim.fft currently supports: (none)
```

## 📎 Related pages

- [Operation Audit](../reference/operation-audit.md) — complete list of every operation and its category
- [FLOP Counting Model](./flop-counting-model.md) — how costs are calculated
- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — what changes when moving from NumPy
```

- [ ] **Step 2: Commit**

```bash
git add docs/concepts/operation-categories.md
git commit -m "docs: add operation categories concept page"
```

---

### Task 13: Create Architecture — Client-Server Model

**Files:**
- Create: `docs/architecture/client-server.md`

- [ ] **Step 1: Create docs/architecture directory**

```bash
mkdir -p docs/architecture
```

- [ ] **Step 2: Write client-server.md**

```markdown
# Client-Server Model

## When to use this page

Use this page to understand how mechestim's client-server architecture works and why it exists.

## Why client-server?

In competition evaluation, participant code runs in an **isolated container** that cannot import NumPy directly. This prevents participants from bypassing FLOP counting by calling NumPy functions outside mechestim.

The client-server model enforces this isolation:

```
┌──────────────────────┐         ┌──────────────────────┐
│  Participant Container│         │   Server Container   │
│                      │         │                      │
│  import mechestim    │  ZMQ    │  mechestim library   │
│  as me               │◄───────►│  (real NumPy)        │
│                      │  IPC/   │                      │
│  # No NumPy here!    │  TCP    │  Budget enforcement  │
│  # Only client proxy │         │  Array storage       │
│                      │         │  FLOP counting       │
└──────────────────────┘         └──────────────────────┘
```

## How it works

1. **Server** runs the real mechestim library backed by NumPy. It stores all arrays, enforces budgets, and counts FLOPs.

2. **Client** is a drop-in replacement (`import mechestim as me`) that proxies every operation to the server over ZMQ (msgpack-encoded messages).

3. **Arrays stay on the server.** The client holds lightweight `RemoteArray` handles that reference server-side data. When you call `me.einsum(...)`, the client sends the operation and handle IDs to the server, which executes it and returns a new handle.

4. **Budget enforcement happens server-side.** The client cannot manipulate FLOP counts.

## Communication protocol

- **Transport:** ZMQ (REQ/REP pattern)
- **Serialization:** msgpack with binary-safe array payloads
- **Default endpoint:** `ipc:///tmp/mechestim.sock` (configurable via `MECHESTIM_SERVER_URL`)
- **Timeout:** 30 seconds per request

## API compatibility

Code written for the local library works unchanged with the client:

```python
# This code works with BOTH the local library and the client
import mechestim as me

with me.BudgetContext(flop_budget=10**6) as budget:
    x = me.zeros((256,))
    W = me.random.randn(256, 256)
    h = me.einsum('ij,j->i', W, x)
    print(budget.summary())
```

## When to use which

| Use case | Package | Install path |
|----------|---------|-------------|
| Development, testing, research | `mechestim` (local library) | `uv add git+...` or `uv sync` from repo |
| Competition evaluation, sandboxed environments | `mechestim-client` + `mechestim-server` | Docker containers |

## Three packages in this repo

| Package | Location | Description |
|---------|----------|-------------|
| `mechestim` | `src/mechestim/` | Local library — full NumPy backend, direct execution |
| `mechestim-client` | `mechestim-client/` | Client proxy — no NumPy dependency, forwards ops to server |
| `mechestim-server` | `mechestim-server/` | Server — runs real mechestim, manages sessions and arrays |

## 📎 Related pages

- [Running with Docker](./docker.md) — set up client-server locally
- [Your First Budget](../getting-started/first-budget.md) — getting started with the local library
```

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/client-server.md
git commit -m "docs: add client-server architecture guide"
```

---

### Task 14: Create Architecture — Running with Docker

**Files:**
- Create: `docs/architecture/docker.md`

- [ ] **Step 1: Write docker.md**

```markdown
# Running with Docker

## When to use this page

Use this page to run the client-server model locally, either with Docker Compose or manually.

## Prerequisites

- [Client-Server Model](./client-server.md) — understand why the architecture exists
- Docker and Docker Compose installed

## With Docker Compose

The `docker/` directory contains a ready-to-use setup:

```bash
cd docker
docker compose up --build
```

This starts two containers:

| Service | Image | Role |
|---------|-------|------|
| `backend` | `Dockerfile.server` | Runs mechestim server, listens on IPC socket |
| `participant` | `Dockerfile.participant` | Runs participant code with mechestim-client only |

The containers share an IPC socket volume for communication.

## Without Docker

Start the server manually:

```bash
# Terminal 1: Start the server
cd mechestim-server
uv run python -m mechestim_server --url ipc:///tmp/mechestim.sock
```

```bash
# Terminal 2: Run client code
export MECHESTIM_SERVER_URL=ipc:///tmp/mechestim.sock
cd mechestim-client
uv run python your_script.py
```

For TCP (e.g., across machines):

```bash
# Server
uv run python -m mechestim_server --url tcp://0.0.0.0:15555

# Client
export MECHESTIM_SERVER_URL=tcp://server-host:15555
uv run python your_script.py
```

## ⚠️ Common pitfalls

**Symptom:** `Connection refused` or `timeout`

**Fix:** Ensure the server is running before starting the client. Check that `MECHESTIM_SERVER_URL` matches the server's `--url` argument.

**Symptom:** Port conflict

**Fix:** Change the port in both the server `--url` and client `MECHESTIM_SERVER_URL`.

## 📎 Related pages

- [Client-Server Model](./client-server.md) — architecture overview
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture/docker.md
git commit -m "docs: add Docker deployment guide"
```

---

### Task 15: Create Reference — Operation Audit

**Files:**
- Create: `docs/reference/operation-audit.md`

- [ ] **Step 1: Create docs/reference directory**

```bash
mkdir -p docs/reference
```

- [ ] **Step 2: Write operation-audit.md**

This page is generated from the registry data. Write a script to extract it, then write the page.

First, generate the operation tables by running:

```bash
uv run python -c "
from mechestim._registry import REGISTRY

categories = {}
for name, info in REGISTRY.items():
    cat = info.get('category', 'unknown')
    categories.setdefault(cat, []).append(name)

print('# Operation Audit')
print()
print('Complete list of every NumPy operation and its mechestim category.')
print('Generated from the operation registry.')
print()
print('## Summary')
print()
print(f'| Category | Count |')
print(f'|----------|-------|')
for cat in ['free', 'counted_unary', 'counted_binary', 'counted_reduction', 'counted_custom', 'blacklisted']:
    print(f'| {cat} | {len(categories.get(cat, []))} |')
print()

for cat, label, desc in [
    ('free', 'Free Operations (0 FLOPs)', 'No FLOP cost. Tensor creation, reshaping, indexing, random.'),
    ('counted_unary', 'Counted Unary Operations', 'Cost: numel(output) per call.'),
    ('counted_binary', 'Counted Binary Operations', 'Cost: numel(output) per call.'),
    ('counted_reduction', 'Counted Reduction Operations', 'Cost: numel(input) per call.'),
    ('counted_custom', 'Counted Custom Operations', 'Bespoke cost formulas per operation.'),
    ('blacklisted', 'Unsupported Operations', 'Raises AttributeError if called.'),
]:
    names = sorted(categories.get(cat, []))
    print(f'## {label}')
    print()
    print(desc)
    print()
    # Print as a wrapped list
    for i in range(0, len(names), 8):
        chunk = names[i:i+8]
        print(', '.join(f'\`{n}\`' for n in chunk) + (',' if i + 8 < len(names) else ''))
    print()
" > docs/reference/operation-audit.md
```

Review the generated file and adjust formatting if needed. The output should have:

- Summary table with counts per category
- One section per category with the full list of operation names
- Operations displayed as inline code in comma-separated lists

- [ ] **Step 3: Commit**

```bash
git add docs/reference/operation-audit.md
git commit -m "docs: add operation audit reference page"
```

---

### Task 16: Create Troubleshooting — Common Errors

**Files:**
- Create: `docs/troubleshooting/common-errors.md`

- [ ] **Step 1: Create docs/troubleshooting directory**

```bash
mkdir -p docs/troubleshooting
```

- [ ] **Step 2: Write common-errors.md**

```markdown
# Common Errors

## When to use this page

Use this page when you encounter an error from mechestim and need to understand what went wrong.

---

## BudgetExhaustedError

**Symptom:**

```
mechestim.errors.BudgetExhaustedError: einsum would cost 16,777,216 FLOPs but only 1,000,000 remain
```

**Why:** The operation you called would exceed the remaining FLOP budget. The operation did **not** execute.

**Fix:** Increase `flop_budget` in your `BudgetContext`, or reduce the cost of your computation. Use `budget.summary()` to see which operations are consuming the most FLOPs.

---

## NoBudgetContextError

**Symptom:**

```
mechestim.errors.NoBudgetContextError: No active BudgetContext. Wrap your code in `with mechestim.BudgetContext(...):`
```

**Why:** You called a counted operation (like `me.einsum`, `me.exp`, etc.) outside a `BudgetContext`.

**Fix:** Wrap your computation in a `BudgetContext`:

```python
with me.BudgetContext(flop_budget=10_000_000) as budget:
    # your code here
```

---

## AttributeError: module 'mechestim' has no attribute '...'

**Symptom:**

```
AttributeError: module 'mechestim' has no attribute 'fft'. mechestim does not support this operation.
```

**Why:** The NumPy function you're trying to use is not in mechestim's allowlist.

**Fix:** Check [Operation Categories](../concepts/operation-categories.md) for supported operations, or see the [Operation Audit](../reference/operation-audit.md) for the complete list.

---

## RuntimeError: Cannot nest BudgetContexts

**Symptom:**

```
RuntimeError: Cannot nest BudgetContexts
```

**Why:** You opened a `BudgetContext` inside another one. Only one can be active per thread.

**Fix:** Restructure your code to use a single `BudgetContext`.

---

## SymmetryError

**Symptom:**

```
mechestim.errors.SymmetryError: Tensor not symmetric along dims (0, 1): max deviation = 0.5
```

**Why:** You passed the same array object to multiple einsum operands, but the array values don't satisfy the symmetry that mechestim detected.

**Fix:** This usually indicates a bug — the same Python object is expected to have identical values. Check that you haven't mutated the array between creating it and calling einsum.

---

## 📎 Related pages

- [Debug Budget Overruns](../how-to/debug-budget-overruns.md) — diagnose which operations are expensive
- [Error Reference (API)](../api/errors.md) — full error class documentation
```

- [ ] **Step 3: Commit**

```bash
git add docs/troubleshooting/common-errors.md
git commit -m "docs: add troubleshooting page"
```

---

### Task 17: Create Example Scripts

**Files:**
- Create: `examples/01_basic_usage.py`
- Create: `examples/02_einsum_patterns.py`
- Create: `examples/03_symmetry_savings.py`
- Create: `examples/04_svd_usage.py`
- Create: `examples/05_budget_planning.py`

- [ ] **Step 1: Create examples directory**

```bash
mkdir -p examples
```

- [ ] **Step 2: Write 01_basic_usage.py**

```python
"""Basic mechestim usage — BudgetContext, free ops, counted ops, summary.

Run: uv run python examples/01_basic_usage.py
"""
import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs)
    W = me.ones((256, 256))
    x = me.ones((256,))

    # Counted operations
    h = me.einsum('ij,j->i', W, x)  # matrix-vector multiply
    h = me.maximum(h, 0)             # ReLU
    h = me.exp(h)                    # exponential
    total = me.sum(h)                # reduction

    print(budget.summary())
    print(f"\nResult: {total}")
```

- [ ] **Step 3: Write 02_einsum_patterns.py**

```python
"""Common einsum patterns and their FLOP costs.

Run: uv run python examples/02_einsum_patterns.py
"""
import mechestim as me

patterns = [
    ("Matrix-vector", "ij,j->i", [(256, 256), (256,)]),
    ("Matrix multiply", "ij,jk->ik", [(256, 256), (256, 256)]),
    ("Outer product", "i,j->ij", [(256,), (256,)]),
    ("Trace", "ii->", [(256, 256)]),
    ("Bilinear form", "ai,bi,ab->", [(10, 256), (10, 256), (10, 10)]),
]

print(f"{'Pattern':<20} {'Subscripts':<15} {'FLOPs':>12}")
print("-" * 50)
for name, subs, shapes in patterns:
    cost = me.flops.einsum_cost(subs, shapes=shapes)
    print(f"{name:<20} {subs:<15} {cost:>12,}")
```

- [ ] **Step 4: Write 03_symmetry_savings.py**

```python
"""Symmetry detection in einsum — same-object savings.

Run: uv run python examples/03_symmetry_savings.py
"""
import mechestim as me

# With symmetry: x passed twice (same object)
with me.BudgetContext(flop_budget=10**8) as budget:
    x = me.ones((10, 256))
    A = me.ones((10, 10))
    result = me.einsum('ai,bi,ab->', x, x, A)
    cost_symmetric = budget.flops_used

# Without symmetry: y is a copy (different object)
with me.BudgetContext(flop_budget=10**8) as budget:
    x = me.ones((10, 256))
    y = x.copy()
    A = me.ones((10, 10))
    result = me.einsum('ai,bi,ab->', x, y, A)
    cost_no_symmetry = budget.flops_used

print(f"Cost with symmetry:    {cost_symmetric:>10,} FLOPs")
print(f"Cost without symmetry: {cost_no_symmetry:>10,} FLOPs")
print(f"Savings:               {cost_no_symmetry - cost_symmetric:>10,} FLOPs ({100 * (1 - cost_symmetric / cost_no_symmetry):.0f}%)")
```

- [ ] **Step 5: Write 04_svd_usage.py**

```python
"""Truncated SVD usage and cost.

Run: uv run python examples/04_svd_usage.py
"""
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    A = me.random.randn(256, 256)

    U, S, Vt = me.linalg.svd(A, k=10)

    print(f"Input shape:  {A.shape}")
    print(f"U shape:      {U.shape}")
    print(f"S shape:      {S.shape}")
    print(f"Vt shape:     {Vt.shape}")
    print(f"FLOP cost:    {budget.flops_used:,}")
    print(f"\nPredicted cost: {me.flops.svd_cost(m=256, n=256, k=10):,}")
```

- [ ] **Step 6: Write 05_budget_planning.py**

```python
"""Budget planning — query costs before executing.

Run: uv run python examples/05_budget_planning.py
"""
import mechestim as me

# Plan a two-layer forward pass
width = 256
budget_limit = 500_000

steps = [
    ("Layer 1: W1 @ x", me.flops.einsum_cost('ij,j->i', shapes=[(width, width), (width,)])),
    ("Layer 1: ReLU", me.flops.pointwise_cost(shape=(width,))),
    ("Layer 2: W2 @ h1", me.flops.einsum_cost('ij,j->i', shapes=[(width, width), (width,)])),
    ("Layer 2: ReLU", me.flops.pointwise_cost(shape=(width,))),
    ("Output: mean", me.flops.reduction_cost(input_shape=(width,))),
]

total = sum(cost for _, cost in steps)
fits = "YES" if total <= budget_limit else "NO"

print(f"Budget: {budget_limit:,} FLOPs")
print(f"{'Operation':<25} {'FLOPs':>10}")
print("-" * 37)
for name, cost in steps:
    print(f"{name:<25} {cost:>10,}")
print("-" * 37)
print(f"{'Total':<25} {total:>10,}")
print(f"Fits in budget? {fits}")

# Now execute it
if total <= budget_limit:
    with me.BudgetContext(flop_budget=budget_limit) as budget:
        x = me.random.randn(width)
        W1 = me.random.randn(width, width)
        W2 = me.random.randn(width, width)

        h1 = me.einsum('ij,j->i', W1, x)
        h1 = me.maximum(h1, 0)
        h2 = me.einsum('ij,j->i', W2, h1)
        h2 = me.maximum(h2, 0)
        result = me.mean(h2)

        print(f"\nActual usage: {budget.flops_used:,} / {budget_limit:,} FLOPs")
```

- [ ] **Step 7: Commit**

```bash
git add examples/
git commit -m "docs: add 5 runnable example scripts"
```

---

### Task 18: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update install command and add docs link**

In `README.md`, replace the Installation section:

Find:
```markdown
### Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
```
```

Replace with:
```markdown
### Installation

```bash
uv add git+https://github.com/AIcrowd/mechestim.git
```
```

Add a "Documentation" link after the closing `</div>` and `---` line, before the description paragraph:

Find:
```markdown
---

**mechestim** is a drop-in
```

Replace with:
```markdown
---

**[📚 Full Documentation](docs/index.md)**

**mechestim** is a drop-in
```

Also update the Development section:

Find:
```markdown
```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
uv run pytest                  # 140 tests
uv run mkdocs serve            # local docs at http://127.0.0.1:8000
```
```

No changes needed — this section already uses `uv`.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README install to uv, add docs link"
```

---

### Task 19: Remove Old quickstart.md

**Files:**
- Delete: `docs/quickstart.md`

- [ ] **Step 1: Remove quickstart.md (replaced by getting-started/ pages)**

```bash
git rm docs/quickstart.md
```

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: remove old quickstart.md (replaced by getting-started/)"
```

---

### Task 20: Verify Build and Final Review

- [ ] **Step 1: Verify mkdocs builds without errors**

```bash
uv run mkdocs build --strict 2>&1
```

Expected: Build succeeds with no errors. Warnings about missing pages are OK during development.

- [ ] **Step 2: Verify all example scripts run**

```bash
uv run python examples/01_basic_usage.py
uv run python examples/02_einsum_patterns.py
uv run python examples/03_symmetry_savings.py
uv run python examples/04_svd_usage.py
uv run python examples/05_budget_planning.py
```

Expected: All scripts run without errors and produce readable output.

- [ ] **Step 3: Verify docs site locally**

```bash
uv run mkdocs serve
```

Open http://127.0.0.1:8000 and check:
- Logo appears in header
- Navigation shows all new sections
- All pages render correctly
- Code blocks have syntax highlighting
- Links between pages work

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "docs: fix issues found during verification"
```
