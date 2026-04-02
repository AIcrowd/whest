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
