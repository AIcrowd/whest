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
