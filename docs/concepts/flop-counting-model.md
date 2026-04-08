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
2. mechestim computes the cost from the shapes: 2 × 256 × 256 = 131,072 FLOPs
3. The cost is checked against the remaining budget
4. If within budget: the operation executes and the cost is deducted
5. If over budget: `BudgetExhaustedError` is raised, the operation does **not** execute

## Cost formulas by category

| Category | Formula | Example |
|----------|---------|---------|
| **Einsum** | Per-step: product of dims × op_factor | `'ij,jk->ik'` → 2 × 3 × 4 × 5 = 120 |
| **Unary** (exp, log, sqrt, ...) | $\text{numel}(\text{output})$ | shape (256, 256) → 65,536 |
| **Binary** (add, multiply, ...) | $\text{numel}(\text{output})$ | shape (256, 256) → 65,536 |
| **Reduction** (sum, mean, max, ...) | $\text{numel}(\text{input})$ | shape (256, 256) → 65,536 |
| **SVD** | $m \cdot n \cdot k$ | (256, 256, k=10) → 655,360 |
| **Solve** | $2n^3/3 + n^2 \cdot n_{\text{rhs}}$ (LU) | (256, 256) solve → ~11.1M |
| **Dot / Matmul** | Same as einsum | (256, 256) @ (256, 256) → 2 × 256³ |
| **Free ops** | 0 | zeros, reshape, etc. |

### Sorting & search

| Category | Formula | Example |
|----------|---------|---------|
| **Sort / Argsort** | $n \cdot \lceil\log_2 n\rceil$ per slice | shape (4, 8), axis=-1 → 4 × 8 × 3 = 96 |
| **Lexsort** | $k \cdot n \cdot \lceil\log_2 n\rceil$ | 2 keys of length 8 → 2 × 8 × 3 = 48 |
| **Partition** | $n$ per slice | shape (100,), kth=50 → 100 |
| **Searchsorted** | $m \cdot \lceil\log_2 n\rceil$ | 5 queries into 1024 → 5 × 10 = 50 |
| **Unique** | $n \cdot \lceil\log_2 n\rceil$ | 8 elements → 8 × 3 = 24 |
| **Set ops** | $(n+m) \cdot \lceil\log_2(n+m)\rceil$ | 4 + 4 elements → 8 × 3 = 24 |

### Histogram & counting

| Category | Formula | Example |
|----------|---------|---------|
| **Histogram** | $n \cdot \lceil\log_2 \text{bins}\rceil$ | 100 elements, 8 bins → 100 × 3 = 300 |
| **Bincount** | $n$ | 100 elements → 100 |

### Random sampling

| Category | Formula | Example |
|----------|---------|---------|
| **Simple samplers** | $\text{numel}(\text{output})$ | shape (10, 20) → 200 |
| **Shuffle / Permutation** | $n \cdot \lceil\log_2 n\rceil$ | 16 elements → 16 × 4 = 64 |

## Symmetry savings

When a tensor is a `SymmetricTensor`, costs are reduced based on the number of unique elements rather than total elements. For a symmetric $n \times n$ matrix, there are $n(n+1)/2$ unique elements instead of $n^2$.

| Category | Symmetric cost | Standard cost |
|----------|---------------|---------------|
| **Pointwise** (unary/binary) | unique_elements | $\text{numel}(\text{output})$ |
| **Reduction** | unique_elements | $\text{numel}(\text{input})$ |
| **Einsum** (symmetric output) | `dense_cost × unique_output / total_output` (exact via `C(n+k-1, k)`) | Full product |
| **Solve** | $n^3/3 + n \cdot n_{\text{rhs}}$ (Cholesky) | $2n^3/3 + n^2 \cdot n_{\text{rhs}}$ (LU) |
| **Det / Slogdet** | $n^3/3$ (Cholesky) | $n^3$ (LU) |
| **Inv** | $n^3/3 + n^3/2$ | $n^3$ |

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage details.

### Induced symmetry from equal operands

When the same Python object is passed as multiple operands to `me.einsum`,
detection identifies this and computes additional output symmetry that
cannot be seen from per-operand symmetry alone. These **induced** groups
are merged with per-operand groups via the same `propagate_symmetry`
machinery, and the enriched `output_symmetry` feeds into the cost formula
`cost = dense × unique_elements / total_elements`.

See the
[exploit-symmetry guide](../how-to/exploit-symmetry.md#automatic-symmetry-from-repeated-operands)
for usage examples.

## Einsum cost model

Every einsum — regardless of the number of operands — is decomposed into
pairwise contraction steps along an optimal path (found via mechestim's
[opt_einsum fork](../api/opt-einsum.md)).
The total cost is the sum of per-step costs:

```
total_cost = sum(step.flop_cost for step in path.steps)
```

For each pairwise step, the dense cost is:

```
dense_step_cost = (product of all index dimensions) × op_factor
```

where `op_factor = 2` when indices are summed (multiply + add) and
`op_factor = 1` when no indices are summed (outer product).

When the output of a step has symmetry (inherited from symmetric inputs via
symmetry propagation), the cost is reduced:

```
step_cost = dense_step_cost × (unique_output_elements / total_output_elements)
```

where `unique_output_elements` is computed exactly using the stars-and-bars
formula `C(n+k-1, k)` for each symmetric group of `k` indices of size `n`.

For a simple two-operand einsum like `'ij,jk->ik'`, there is one step,
so the total cost equals the step cost. For multi-operand einsums (3+
tensors), the optimizer finds the pairwise ordering that minimizes the
total cost.

When symmetric tensors are present, the optimizer is symmetry-aware: it
uses symmetric costs to decide which pair to contract at each step, so the
returned path may differ from the dense-optimal path. Symmetry propagates
through intermediates — if an early contraction produces a symmetric
intermediate, subsequent steps benefit from the reduced element count, and
the optimizer factors this into its ordering decisions.

Use `me.einsum_path()` to inspect the per-step breakdown. See
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
