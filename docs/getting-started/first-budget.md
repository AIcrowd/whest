# Your First Budget

## When to use this page

Use this page after installing mechestim to run your first FLOP-counted computation.

## Prerequisites

- [Installation](./installation.md)

## Quickest possible start

You do not need to set up a budget context to start counting FLOPs. mechestim activates a global default context the first time any counted operation runs. The default budget is 1e15 FLOPs (configurable via the `MECHESTIM_DEFAULT_BUDGET` environment variable).

Save this as `first_budget.py`:

```python
import math
import mechestim as me

depth = 10   # number of layers
width = 256  # hidden dimension

# No BudgetContext needed — the global default activates automatically
scale = math.sqrt(2.0 / width)   # Kaiming init scale; free (no FLOPs)
weights = [
    me.array(me.random.randn(width, width) * scale)
    for _ in range(depth)
]
x = me.random.randn(width)

h = x
for W in weights:
    h = me.einsum('ij,j->i', W, h)  # matrix-vector multiply: counted
    h = me.maximum(h, 0)             # ReLU activation: counted

result = me.sum(h)                   # reduction: counted

# Print a Rich-formatted summary across all namespaces
me.budget_summary()
```

Run it:

```bash
uv run python first_budget.py
```

## 🔍 What you'll see

```
┌─────────────────────────────────────────────────────────┐
│              mechestim FLOP Budget Summary               │
├──────────────┬───────────────┬────────────┬─────────────┤
│ Namespace    │ Budget        │ Used       │ Remaining   │
├──────────────┼───────────────┼────────────┼─────────────┤
│ (default)    │ 1.00e+15      │ 658,176    │ ~1.00e+15   │
└──────────────┴───────────────┴────────────┴─────────────┘

  (default) — by operation
    einsum    655,360  ( 99.6%)  [10 calls]
    maximum     2,560  (  0.4%)  [10 calls]
    sum           256  (  0.0%)   [1 call]
```

**Reading the output:**

- **Namespace:** `(default)` is the auto-created global context; named contexts appear here once you add them
- **Used / Remaining:** how much of the budget has been consumed across all calls
- **By operation:** breakdown of costs per operation type and call count
- The 10-layer MLP spends almost all FLOPs on `einsum` (matrix multiplies); activations (`maximum`) are comparatively cheap

## Explicit context with a namespace

When you want a tighter budget or cleaner grouping in summaries, wrap operations in a `BudgetContext` and give it a namespace:

```python
import math
import mechestim as me

depth = 10
width = 256

with me.BudgetContext(flop_budget=50_000_000, namespace="mlp-forward") as budget:
    scale = math.sqrt(2.0 / width)
    weights = [
        me.array(me.random.randn(width, width) * scale)
        for _ in range(depth)
    ]
    x = me.random.randn(width)

    h = x
    for W in weights:
        h = me.einsum('ij,j->i', W, h)  # 65,536 FLOPs each pass
        h = me.maximum(h, 0)             # 256 FLOPs each pass

    result = me.sum(h)                   # 256 FLOPs

me.budget_summary()
```

```
┌──────────────────────────────────────────────────────────┐
│              mechestim FLOP Budget Summary                │
├───────────────┬────────────┬──────────┬──────────────────┤
│ Namespace     │ Budget     │ Used     │ Remaining        │
├───────────────┼────────────┼──────────┼──────────────────┤
│ mlp-forward   │ 50,000,000 │ 658,176  │ 49,341,824       │
└───────────────┴────────────┴──────────┴──────────────────┘

  mlp-forward — by operation
    einsum    655,360  ( 99.6%)  [10 calls]
    maximum     2,560  (  0.4%)  [10 calls]
    sum           256  (  0.0%)   [1 call]
```

## Decorator form

Use `@me.budget` to attach a budget directly to a function. The namespace defaults to the function name if omitted:

```python
import math
import mechestim as me

@me.budget(flop_budget=50_000_000, namespace="mlp-forward")
def run_mlp(depth: int = 10, width: int = 256):
    scale = math.sqrt(2.0 / width)
    weights = [
        me.array(me.random.randn(width, width) * scale)
        for _ in range(depth)
    ]
    x = me.random.randn(width)

    h = x
    for W in weights:
        h = me.einsum('ij,j->i', W, h)
        h = me.maximum(h, 0)

    return me.sum(h)

run_mlp()
me.budget_summary()
```

Each call to `run_mlp()` draws from the same `mlp-forward` namespace budget. Call `me.budget_data()` to retrieve the summary as a plain dict for programmatic use:

```python
data = me.budget_data()
# {'mlp-forward': {'budget': 50_000_000, 'used': 658176, 'remaining': 49341824, ...}}
```

## ⚠️ Common pitfalls

**Symptom:** `BudgetExhaustedError`

**Fix:** Your operations exceed the budget you set. Increase `flop_budget` on the `BudgetContext` (or decorator), reduce computation, or rely on the global default context which has a 1e15 FLOP ceiling.

**Note on `NoBudgetContextError`:** This error no longer triggers in normal use. The global default context activates automatically on first use, so bare calls outside any `with` block are safe.

## 📎 Related pages

- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — convert existing NumPy code to mechestim
- [Plan Your Budget](../how-to/plan-your-budget.md) — query operation costs before executing
