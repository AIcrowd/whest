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
    h = me.einsum('ij,j->i', W, h)  # matrix-vector multiply: 256 × 256 = 65,536 FLOPs
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
│ (default)    │ 1.00e+15      │ 1,969,152  │ ~1.00e+15   │
└──────────────┴───────────────┴────────────┴─────────────┘

  (default) — by operation
    einsum        1,310,720  ( 66.6%)  [10 calls]
    random.randn    655,616  ( 33.3%)  [11 calls]
    maximum           2,560  (  0.1%)  [10 calls]
    sum                 256  (  0.0%)   [1 call]
```

**Reading the output:**

- **Namespace:** `(default)` is the auto-created global context; named contexts appear here once you add them
- **Used / Remaining:** how much of the budget has been consumed across all calls
- **By operation:** breakdown of costs per operation type and call count
- The 10-layer MLP spends two-thirds of FLOPs on `einsum` (matrix multiplies) and one-third on random number generation; activations (`maximum`) are comparatively cheap

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
        h = me.einsum('ij,j->i', W, h)  # matrix-vector multiply: 256 × 256 = 65,536 FLOPs
        h = me.maximum(h, 0)             # 256 FLOPs each pass

    result = me.sum(h)                   # 256 FLOPs

me.budget_summary()
```

```
┌──────────────────────────────────────────────────────────┐
│              mechestim FLOP Budget Summary                │
├───────────────┬────────────┬───────────┬─────────────────┤
│ Namespace     │ Budget     │ Used      │ Remaining       │
├───────────────┼────────────┼───────────┼─────────────────┤
│ mlp-forward   │ 50,000,000 │ 1,969,152 │ 48,030,848      │
└───────────────┴────────────┴───────────┴─────────────────┘

  mlp-forward — by operation
    einsum        1,310,720  ( 66.6%)  [10 calls]
    random.randn    655,616  ( 33.3%)  [11 calls]
    maximum           2,560  (  0.1%)  [10 calls]
    sum                 256  (  0.0%)   [1 call]
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

Each call to `run_mlp()` draws from the same `mlp-forward` namespace budget. Call `me.budget_summary_dict()` to retrieve the summary as a plain dict for programmatic use:

```python
data = me.budget_summary_dict()
# {'flop_budget': ..., 'flops_used': ..., 'flops_remaining': ..., 'operations': {...}}

# For per-namespace breakdown:
data = me.budget_summary_dict(by_namespace=True)
# data["by_namespace"]["mlp-forward"]["flops_used"] -> 1313536
```

## Configuring the global default budget

The global default budget is 1e15 FLOPs (1 quadrillion). You can change this via the `MECHESTIM_DEFAULT_BUDGET` environment variable:

```bash
# Set a smaller default budget (e.g., 1 billion FLOPs)
export MECHESTIM_DEFAULT_BUDGET=1e9
uv run python your_script.py
```

The env var is read once when the global default is first created (on the first counted operation). It accepts any numeric value that Python's `float()` can parse (e.g., `1e9`, `1000000000`, `5e12`).

## ⚠️ Common pitfalls

**Symptom:** `BudgetExhaustedError`

**Fix:** Your operations exceed the budget you set. Increase `flop_budget` on the `BudgetContext` (or decorator), reduce computation, or rely on the global default context which has a 1e15 FLOP ceiling (configurable via `MECHESTIM_DEFAULT_BUDGET`).

**Note on `NoBudgetContextError`:** This error no longer triggers in normal use. The global default context activates automatically on first use, so bare calls outside any `with` block are safe.

## 📎 Related pages

- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — convert existing NumPy code to mechestim
- [Plan Your Budget](../how-to/plan-your-budget.md) — query operation costs before executing
