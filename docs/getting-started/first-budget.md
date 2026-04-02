# Your First Budget

## When to use this page

Use this page after installing mechestim to run your first FLOP-counted computation.

## Prerequisites

- [Installation](./installation.md)

## Do this now

Save this as `first_budget.py`:

```python
import math
import mechestim as me

depth = 10   # number of layers
width = 256  # hidden dimension

with me.BudgetContext(flop_budget=50_000_000) as budget:
    # --- Free: tensor creation costs 0 FLOPs ---
    # Initialise weights with variance 2/fan_in for stable gradients
    scale = math.sqrt(2.0 / width)
    weights = [
        me.array(me.random.randn(width, width) * scale)
        for _ in range(depth)
    ]
    x = me.random.randn(width)

    # --- Counted: each operation deducts from the FLOP budget ---
    h = x
    for W in weights:
        h = me.einsum('ij,j->i', W, h)  # matrix-vector multiply
        h = me.maximum(h, 0)             # ReLU activation

    result = me.sum(h)

    # Inspect your budget
    print(budget.summary())
```

Run it:

```bash
uv run python first_budget.py
```

## 🔍 What you'll see

```
mechestim 0.2.0 (numpy 2.1.3 backend) | budget: 5.00e+07 FLOPs
mechestim FLOP Budget Summary
==============================
  Total budget:      50,000,000
  Used:                 658,176  (1.3%)
  Remaining:         49,341,824  (98.7%)

  By operation:
    einsum                655,360  ( 99.6%)  [10 calls]
    maximum                 2,560  (  0.4%)  [10 calls]
    sum                       256  (  0.0%)  [1 call]
```

**Reading the output:**

- **Total budget:** the FLOP limit you set
- **Used / Remaining:** how much of the budget has been consumed
- **By operation:** breakdown of costs per operation type and call count
- Notice how the 10-layer MLP uses most of its FLOPs on `einsum` (matrix multiplies), while activations (`maximum`) are comparatively cheap

## ⚠️ Common pitfalls

**Symptom:** `NoBudgetContextError: No active BudgetContext`

**Fix:** All counted operations must run inside a `with me.BudgetContext(...)` block.

**Symptom:** `BudgetExhaustedError`

**Fix:** Your operations exceed the budget. Increase `flop_budget` or reduce computation.

## 📎 Related pages

- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — convert existing NumPy code to mechestim
- [Plan Your Budget](../how-to/plan-your-budget.md) — query operation costs before executing
