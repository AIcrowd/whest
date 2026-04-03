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
| **Einsum** | Product of all index dimensions | `'ij,jk->ik'` with (256,256) × (256,256) → $256^3$ |
| **Unary** (exp, log, sqrt, ...) | $\text{numel}(\text{output})$ | shape (256, 256) → 65,536 |
| **Binary** (add, multiply, ...) | $\text{numel}(\text{output})$ | shape (256, 256) → 65,536 |
| **Reduction** (sum, mean, max, ...) | $\text{numel}(\text{input})$ | shape (256, 256) → 65,536 |
| **SVD** | $m \cdot n \cdot k$ | (256, 256, k=10) → 655,360 |
| **Solve** | $2n^3/3 + n^2 \cdot n_{\text{rhs}}$ (LU) | (256, 256) solve → ~11.1M |
| **Dot / Matmul** | Equivalent einsum cost | (256, 256) @ (256, 256) → $256^3$ |
| **Free ops** | 0 | zeros, reshape, etc. |

## Symmetry savings

When a tensor is a `SymmetricTensor`, costs are reduced based on the number of unique elements rather than total elements. For a symmetric $n \times n$ matrix, there are $n(n+1)/2$ unique elements instead of $n^2$.

| Category | Symmetric cost | Standard cost |
|----------|---------------|---------------|
| **Pointwise** (unary/binary) | unique_elements | $\text{numel}(\text{output})$ |
| **Reduction** | unique_elements | $\text{numel}(\text{input})$ |
| **Einsum** (symmetric input) | Scaled by unique/total ratio | Full product |
| **Einsum** (symmetric output) | Divided by group factorial | Full product |
| **Solve** | $n^3/3 + n \cdot n_{\text{rhs}}$ (Cholesky) | $2n^3/3 + n^2 \cdot n_{\text{rhs}}$ (LU) |
| **Det / Slogdet** | $n^3/3$ (Cholesky) | $n^3$ (LU) |
| **Inv** | $n^3/3 + n^3/2$ | $n^3$ |

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage details.

## Multi-operand einsum cost

For einsums with 3+ operands, mechestim decomposes the contraction into a
sequence of pairwise steps along the optimal contraction path (found via
opt_einsum). The optimizer is symmetry-aware in its ordering: it uses
symmetric costs to decide which pair to contract at each step, so the
returned path may differ from the dense-optimal path when symmetric tensors
are present. The total cost is the sum of pairwise step costs, where each
step's cost is reduced by the symmetry of its inputs and output:

```
total_cost = sum(step.flop_cost for step in path.steps)
```

The per-step cost formula correctly distinguishes between summed and
surviving indices in symmetric groups. For a symmetric group such as S_3
on {i,j,k}, if index i is summed out in that step, only the S_2 subgroup
on the surviving indices {j,k} contributes a symmetry reduction. Summed
indices receive no symmetry discount.

Symmetry propagates through intermediates: if an early contraction produces
a symmetric intermediate, subsequent steps that consume it benefit from
the reduced element count, and the optimizer factors this into its ordering
decisions. Use `me.einsum_path()` to inspect the per-step breakdown. See
[Use Einsum](../how-to/use-einsum.md) for examples.

## FLOP multiplier

The `flop_multiplier` parameter in `BudgetContext` scales all costs:

```python
with me.BudgetContext(flop_budget=10**6, flop_multiplier=2.0) as budget:
    # Every operation costs 2× its normal FLOP count
    ...
```

This is useful for experimentation or adjusting the difficulty of a budget constraint.

## Namespaces

The `namespace` parameter in `BudgetContext` is a display-only label for grouping operations in summaries:

```python
with me.BudgetContext(flop_budget=10**6, namespace="training") as budget:
    # Operations are tagged with "training" for display
    ...
```

Namespaces do not affect FLOP counting or budget enforcement — they only appear in `me.budget_summary()` output.

## 📎 Related pages

- [Operation Categories](./operation-categories.md) — which operations are free, counted, or unsupported
- [Plan Your Budget](../how-to/plan-your-budget.md) — query costs before running
