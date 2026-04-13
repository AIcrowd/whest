# FLOP Counting Model

## When to use this page

Use this page to understand how whest counts FLOPs and why it uses analytical counting instead of runtime measurement.

## Convention: FMA = 1 operation

This codebase counts a fused multiply-add (a * b + c) as a **single operation**.
Hardware FMA units execute this in one instruction; the common textbook
convention of counting it as 2 (one multiply + one add) is **not** used here.
All cost formulae reflect this: a matrix multiply of dimensions
(m, k) x (k, n) costs m*k*n operations, not 2*m*k*n.

## Why FLOPs instead of wall-clock time

- **Deterministic:** The same code always produces the same FLOP count, regardless of hardware
- **Hardware-independent:** A matmul costs the same FLOPs on a laptop and a server
- **Reproducible:** No variance from CPU scheduling, cache effects, or thermal throttling
- **Composable:** You can sum individual operation costs to predict total cost

## How costs are computed

whest computes FLOP costs **analytically from tensor shapes**, not by measuring execution time.

1. You call a counted operation (e.g., `we.einsum('ij,j->i', W, x)`)
2. whest computes the cost from the shapes: 256 × 256 = 65,536 FLOPs
3. The cost is checked against the remaining budget
4. If within budget: the operation executes and the cost is deducted
5. If over budget: `BudgetExhaustedError` is raised, the operation does **not** execute

## Cost formulas by category

Each formula below gives the **analytical base cost**. When per-operation
[weights](#per-operation-weights) are loaded, the base cost is multiplied
by the operation's weight to give the final deducted cost.

| Category | Formula | Example |
|----------|---------|---------|
| **Einsum** | Per-step: product of all index dims | `'ij,jk->ik'` → 3 × 4 × 5 = 60 |
| **Unary** (exp, log, sqrt, ...) | $\text{numel}(\text{output})$ | shape (256, 256) → 65,536 |
| **Binary** (add, multiply, ...) | $\text{numel}(\text{output})$ | shape (256, 256) → 65,536 |
| **Reduction** (sum, mean, max, ...) | $\text{numel}(\text{input})$ | shape (256, 256) → 65,536 |
| **SVD** | $m \cdot n \cdot k$ | (256, 256, k=10) → 655,360 |
| **Solve** | $n^3/3 + n^2 \cdot n_{\text{rhs}}$ (LU) | (256, 256) solve → ~5.6M |
| **Dot / Matmul** | Same as einsum | (256, 256) @ (256, 256) → 256³ |
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
| **Einsum** (symmetric contraction) | Symmetry-reduced (see below) | Full product |
| **Solve** | $n^3/3 + n \cdot n_{\text{rhs}}$ (Cholesky) | $n^3/3 + n^2 \cdot n_{\text{rhs}}$ (LU) |
| **Det / Slogdet** | $n^3/3$ (Cholesky) | $n^3$ (LU) |
| **Inv** | $n^3/3 + n^3/2$ | $n^3$ |

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage details.

### Subgraph symmetry detection

Symmetry that reduces einsum costs comes from two complementary sources,
both unified under the **subgraph symmetry detection** algorithm:

1. **Declared per-operand symmetry.** When an operand is wrapped with
   `we.as_symmetric()`, its symmetry groups are embedded in the bipartite
   graph as U-vertex equivalence classes. These propagate into intermediate
   tensors automatically.

2. **Induced symmetry from repeated operands.** When the same Python object
   is passed at multiple operand positions, the subgraph oracle detects this
   via Python identity (`is`) and derives symmetry groups on the output that
   cannot be seen from per-operand metadata alone.

The oracle builds a bipartite graph once per `contract_path` call and
evaluates symmetry lazily per subset of operands encountered during path
search. Both sources are merged via the same group-merging machinery, so a
tensor that is both `SymmetricTensor` and also repeated in the subscript
benefits from both contributions simultaneously.

See the
[exploit-symmetry guide](../how-to/exploit-symmetry.md#automatic-symmetry-from-repeated-operands)
for usage examples, and the
[subgraph symmetry explanation](../explanation/subgraph-symmetry.md)
for the algorithm walkthrough.

## Einsum cost model

Every einsum — regardless of the number of operands — is decomposed into
pairwise contraction steps along an optimal path (found via whest's
[opt_einsum fork](../api/opt-einsum.md)).
The total cost is the sum of per-step costs:

```
total_cost = sum(step.flop_cost for step in path.steps)
```

### Per-step cost

For each pairwise step, the **dense** cost is:

```
dense_step_cost = product of all index dimensions
```

Each fused multiply-add (FMA) counts as 1 operation (see
[Convention](#convention-fma--1-operation) above), so the cost of a
contraction step is simply the product of all index dimensions — there is
no factor-of-2 distinction between inner products and outer products.

When symmetry is present, whest reduces each step's cost based on
the structure of the contraction.

### Symmetric contraction cost

Each pairwise step's cost is reduced by two independent multiplicative
factors — one for the output (V-side) indices and one for the inner
(W-side) contracted indices:

```
step_cost = dense_step_cost
          × (unique_output_elements / total_output_elements)
          × (unique_inner_elements / total_inner_elements)
```

Each ratio is computed exactly using **Burnside's lemma** over the
permutation group detected for that step by the
[`SubgraphSymmetryOracle`](../explanation/subgraph-symmetry.md). For the
full symmetric group S$_k$ on $k$ equal-sized axes, Burnside reduces to
the stars-and-bars formula $\binom{n+k-1}{k}$; for proper subgroups like
$C_k$ or block groups the oracle returns the exact generators and
Burnside counts over the enumerated elements.

The **output (V-side) reduction** is always applied when the step's
intermediate has a non-trivial permutation group on its free indices —
only the unique output elements need to be computed.

The **inner (W-side) reduction** is applied only when *all* labels in
the detected inner group are present as contracted indices in that
specific pairwise step. If any of those labels were contracted at an
earlier step and no longer appear in the current step, the inner
reduction is skipped (the per-step table shows this as `[W: ...]` when
detected-but-not-applied versus `[W✓: ...]` when applied). Inner
symmetry can be toggled globally with
`we.configure(use_inner_symmetry=False)`.

The two factors are independent; outer-product contractions (no summed
indices) and non-uniform index dimensions are handled by the same
formula, since Burnside's lemma makes no assumption about uniform sizes
beyond requiring axes in the same orbit to share a dimension.

### Multi-operand contractions

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

Use `we.einsum_path()` to inspect the per-step breakdown. See
[Use Einsum](../how-to/use-einsum.md) for examples.

## Per-operation weights

The analytical formulas above treat all operations within a category as
equally expensive -- `exp`, `log`, `sin`, and `abs` all cost
$\text{numel}(\text{output})$ FLOPs. In reality, `exp` decomposes into
a minimax polynomial approximation requiring approximately 14 FP
instructions per element, while `abs` is a single bit manipulation.

Per-operation **weights** correct for this. Each weight is a multiplicative
constant applied on top of the analytical formula:

```
actual_cost = analytical_formula(shape) × weight(op_name)
```

| Operation | Analytical formula | Weight | Effective cost (256x256) |
|-----------|--------------------|--------|----------------|
| `add` | $\text{numel}(\text{output})$ | 1 | 65,536 |
| `exp` | $\text{numel}(\text{output})$ | 16 | 1,048,576 |
| `sin` | $\text{numel}(\text{output})$ | 16 | 1,048,576 |
| `matmul` | $m \cdot k \cdot n$ | 1 | 16,777,216 |
| `linalg.cholesky` | $n^3$ | 4 | 67,108,864 |

Weights are measured using the overhead-subtracted correction-factor
methodology described in [FLOP Weight Calibration Results](../reference/empirical-weights.md).
The formula is:

$$w(\text{op}) = \max\bigl(\alpha_{\text{raw}}(\text{op}) - \text{overhead}_{\text{category}}, \ 0\bigr)$$

where $\alpha_{\text{raw}}$ is the median ratio of hardware-observed FP
instructions to analytical FLOPs (FMA = 1 op), measured via
`fp_arith_inst_retired` performance counters. The ufunc dispatch overhead
(measured from `np.abs`, which generates zero FP arithmetic) is subtracted
per category to remove numpy implementation noise from the weight.

BLAS-backed operations (contractions, linalg) have weights near 1.0 because
their tight FMA loops execute almost exactly 1 hardware FP instruction per
analytical FLOP, with no ufunc overhead to subtract.

Integer and bitwise operations (`bitwise_and`, `gcd`, `lcm`, etc.) use the
`instructions` hardware counter (total retired instructions) because they
do not retire `fp_arith_inst_retired` events. Their weights are derived from
instruction counts normalized the same way as FP operations.

Weights are loaded from a JSON config file. Without a config file, all
weights default to 1.0 -- the analytical formulas apply unchanged.

### How weights are applied

Weights are applied centrally in `BudgetContext.deduct()`. Every counted
operation passes its `op_name` to `deduct()`, which looks up the weight
and multiplies it into the cost:

```python
adjusted_cost = analytical_cost × flop_multiplier × weight(op_name)
```

This means weights compose with `flop_multiplier` and with symmetry
reductions -- symmetry reduces the element count, the weight scales the
per-element cost, and both apply independently.

### Loading a weights config

Set the `WHEST_WEIGHTS_FILE` environment variable to load weights at
import time:

```bash
export WHEST_WEIGHTS_FILE=/path/to/weights.json
```

The JSON file must have a `"weights"` key mapping operation names to floats:

```json
{
  "weights": {
    "add": 1.0,
    "exp": 16.0,
    "sin": 16.0,
    "matmul": 1.0,
    "linalg.cholesky": 4.0
  }
}
```

Operations not listed in the file default to 1.0. See
[Calibrate Weights](../how-to/calibrate-weights.md) for how to generate
this file.

### Where weights come from

Weights can be determined in two ways:

1. **Hardware performance counters** (Linux `perf stat`) -- counts actual
   floating-point instructions retired by the CPU, weighted by SIMD width.
   This gives the true number of basic FP ops per high-level operation.

2. **Wall-clock time normalization** -- measures `time(op) / time(add)` as
   a relative proxy. Less precise than hardware counters but works on any
   platform.

The `benchmarks/` package in this repository automates both methods. See
[Calibrate Weights](../how-to/calibrate-weights.md).

## FLOP multiplier

The `flop_multiplier` parameter in `BudgetContext` scales all costs:

```python
with we.BudgetContext(flop_budget=10**6, flop_multiplier=2.0) as budget:
    # Every operation costs 2× its normal FLOP count
    ...
```

This is useful for experimentation or adjusting the difficulty of a budget
constraint. Note that `flop_multiplier` and per-operation weights are
independent — `flop_multiplier` scales *all* operations uniformly, while
weights scale each operation individually.

## Namespaces

The `namespace` parameter in `BudgetContext` is a display-only label for grouping operations in summaries:

```python
with we.BudgetContext(flop_budget=10**6, namespace="training") as budget:
    # Operations are tagged with "training" for display
    ...
```

Namespaces do not affect FLOP counting or budget enforcement — they only appear in `we.budget_summary()` output.

## Related pages

- [Operation Categories](./operation-categories.md) — which operations are free, counted, or unsupported
- [Plan Your Budget](../how-to/plan-your-budget.md) — query costs before running
- [Calibrate Weights](../how-to/calibrate-weights.md) — measure per-operation weights empirically
