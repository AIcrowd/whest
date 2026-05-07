# Migration: Symmetry-Aware Einsum Cost (JS-mirrored)

This is a behavior change in the FLOP cost charged for einsum operations
involving SymmetricTensor inputs.

## What changed

The einsum cost model was rewritten to match the canonical specification at
`website/components/symmetry-aware-einsum-contractions/`. The new model:

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
