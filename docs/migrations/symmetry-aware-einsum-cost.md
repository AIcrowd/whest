# Migration: Symmetry-Aware Einsum Cost (JS-mirrored)

This is a behavior change in the FLOP cost charged for einsum operations
involving SymmetricTensor inputs.

## What changed

The einsum cost model was rewritten to match the canonical specification
described in the [Symmetry Detection Deep Dive](/docs/understanding/symmetry-detection/).
The new model:

- Computes a path-independent direct-event count: `(k-1)·∏ M_a + ∏ α_a`
  per independent component.
- Uses a 5-regime classification ladder (`trivial`, `functionalProjection`,
  `singleton`, `young`, `partitionCount`) plus an explicit `unavailable` state.
- Charges `k · ∏ n_ℓ` (dense) when the typed-partition budget is exceeded —
  conservative and gaming-resistant.

The previous model charged a per-step `cost · unique / total` ratio per pairwise
contraction, which depended on the contraction path opt_einsum picked.

## How to migrate

If your code reads `path_info.optimized_cost` or relies on `BudgetContext.spent`
matching specific FLOP integers, expect numbers to shift for any expression with
declared symmetry. Trivial-symmetry expressions are unchanged.

To inspect the new cost decomposition:

```python
import flopscope as fps
import numpy as np

A = fps.as_symmetric(np.zeros((4, 4, 4)), symmetry=(0, 1, 2))  # S_3
cost = fps.einsum_accumulation_cost('ijk,abc->ic', A, A)

print(f'total = {cost.total}')
print(f'mu = (k-1) * prod(M) = {cost.mu}')
print(f'alpha = prod(alpha_a) = {cost.alpha}')
for component in cost.per_component:
    print(f'  {component.labels}: M={component.m}, '
          f'alpha={component.alpha}, regime={component.regime_id}')
```

## What didn't change

- Path optimization picks the same paths it did with the dense (no-symmetry)
  cost model. Execution is unchanged.
- `SymmetricTensor` class, `as_symmetric`, declared symmetry validation — all
  unchanged.
- Non-einsum operations (sum, mean, etc.) still use `_symmetry_adjusted_cost`
  with the existing `unique / dense` ratio.

## Future work

The reduction-cost API hooks (`aggregate_reduction`) are committed as stubs
raising `NotImplementedError`. A follow-up sprint implements ufunc.reduce-aware
cost calculation reusing the same per-component machinery.

---

## Reduction cost rewrite

PR #91 also rewrites the cost surface for `np.ufunc.reduce`-shaped operations (`sum`, `prod`, `max`, `min`, `all`, `any`, `mean`) and for the partition-style reductions (`median`, `percentile`, `quantile`). They now use the same orbit-aware model as einsum.

**What this means concretely (closes [#56](https://github.com/AIcrowd/flopscope/issues/56)):**

```python
import numpy as np
import flopscope as flops

# Before PR #91:  sum on (10,) charged n flops = 10
# After PR #91:   sum on (10,) charges (n - 1) = 9
print(flops.reduction_accumulation_cost(np.zeros(10)).total)  # → 9

# mean on (10,) adds one divide per output orbit
print(flops.reduction_accumulation_cost(np.zeros(10), extra_ops=1).total)  # → 10
```

For inputs with declared symmetry, both terms drop — see the new [Symmetry-aware FLOP counting](/docs/understanding/symmetry-detection/) page for worked examples.

`fnp.median`, `fnp.percentile`, `fnp.quantile` use a separate Tier-2 selection-style cost; inspect with `flops.tier2_reduction_cost(...)`.

---

## FMA convention is now configurable

The `FMA_COST` constant in `flopscope._cost_model` has been removed. Read the current value with `flops.fma_cost()` (a function, not a constant) and override with `flops.configure(fma_cost=...)`. Only values `1` (default, hardware convention) and `2` (textbook) are valid.

```python
# Before:
from flopscope._cost_model import FMA_COST  # removed

# After:
import flopscope as flops
flops.fma_cost()            # → 1 by default
flops.configure(fma_cost=2)  # textbook convention
```

All flopscope cost surfaces consult `fma_cost()` uniformly — no module diverges.

---

## New public inspection and cache API

PR #91 promotes several utilities to the top-level `flopscope.*` surface so participant code doesn't need private imports.

| Function | What it does |
|---|---|
| `flops.einsum_accumulation_cost(subs, *operands)` | Returns an `AccumulationCost` for an einsum expression. Path-independent. |
| `flops.reduction_accumulation_cost(a, axis=None, ...)` | Returns an `AccumulationCost` for an additive reduction. |
| `flops.tier2_reduction_cost(a, axis=None, *, dense_per_output_cost=None)` | Returns the FLOP total for a selection-style reduction (`median` / `percentile` / `quantile`). |
| `flops.einsum_clear_caches()` | Clears the einsum path and accumulation-cost caches. |
| `flops.einsum_cache_info()` | Returns `{"path": CacheInfo, "accumulation": CacheInfo}`. |
| `flops.reduction_clear_cache()` | Clears the reduction accumulation-cost cache. |
| `flops.reduction_cache_info()` | Returns the CacheInfo for the reduction cache. |
| `flops.clear_cache()` | Clears all flopscope caches (einsum + reduction) in one call. |
| `flops.fma_cost()` | Returns the current FMA-convention setting (1 or 2). |

See the new [Symmetry-aware FLOP counting](/docs/understanding/symmetry-detection/) page for usage examples.
