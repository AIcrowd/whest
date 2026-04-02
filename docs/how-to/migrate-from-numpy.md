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
