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

**After (mechestim) — simplest form:**

```python
import mechestim as me

# No setup needed — global default budget tracks FLOPs automatically
W = me.random.randn(256, 256)
x = me.random.randn(256)
h = me.dot(W, x)
h = me.maximum(h, 0)

me.budget_summary()  # see what you spent
```

**After (mechestim) — with explicit budget control:**

```python
import mechestim as me

with me.BudgetContext(flop_budget=20_000_000) as budget:
    W = me.random.randn(256, 256)
    x = me.random.randn(256)
    h = me.dot(W, x)
    h = me.maximum(h, 0)
```

## What stays the same

- Function signatures match NumPy for supported operations
- Broadcasting rules are identical
- Array indexing, slicing, and assignment work normally

## What changes

| NumPy | mechestim | Notes |
|-------|-----------|-------|
| `import numpy as np` | `import mechestim as me` | Drop-in replacement |
| Call ops anywhere | Works anywhere too | A global default budget auto-activates; use explicit `BudgetContext` for limits and namespacing |
| `np.linalg.svd(A)` | `me.linalg.svd(A, k=10)` | Truncated SVD with explicit `k` |
| Plain `ndarray` only | `SymmetricTensor` available | Wrap with `me.as_symmetric()` for cost savings |
| All NumPy ops available | Most available, 32 blacklisted | I/O and config ops raise `AttributeError` |
| No cost tracking | Automatic FLOP counting | Every counted op deducts from budget |

## Common pitfalls

**Symptom:** `AttributeError` when calling an I/O or config function (e.g., `me.save`, `me.seterr`)

**Fix:** 32 operations are blacklisted because they are I/O, configuration, or datetime functions with no FLOP cost. See [Operation Categories](../concepts/operation-categories.md) for the full list. Use `numpy` directly for these.

**Symptom:** Using `np.linalg.svd` instead of `me.linalg.svd`

**Fix:** If you import NumPy alongside mechestim, make sure to use `me.` for operations you want counted. Operations called through `np.` bypass FLOP counting entirely.

## 📎 Related pages

- [Operation Categories](../concepts/operation-categories.md) — what's supported and what isn't
- [Operation Audit](../reference/operation-audit.md) — full list of all operations
