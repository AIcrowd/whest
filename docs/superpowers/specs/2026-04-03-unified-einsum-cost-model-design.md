# Unified Einsum Cost Model

**Date:** 2026-04-03
**Status:** Draft

## Summary

Remove all legacy einsum cost mechanisms (repeated-operand detection, hand-rolled factorial division, separate `einsum_cost` formula). Every einsum cost — budget deduction, pre-flight estimation, PathInfo reporting — comes from a single source: `_opt_einsum.contract_path()`. Switch to the standard FLOP convention (multiply-add = 2 FLOPs). Rewrite documentation with a single coherent narrative.

## Problem

The codebase currently has two competing cost systems:

1. **Legacy `einsum_cost()` in `_flops.py`**: hand-rolled formula with `product_of_labels // (repeated_factorial × symmetric_factorial) × (unique/total)`. Used for budget deduction.
2. **`_opt_einsum.contract_path()`**: symmetry-aware path optimizer that computes per-step costs via `symmetric_flop_count`. Used for PathInfo/StepInfo reporting.

The `einsum()` function calls BOTH: contract_path for the path, then einsum_cost for the budget. These can disagree. The legacy system also has the `_detect_repeated_operands` hack (Python `id()` comparison) which is a different concept from tensor symmetry.

The documentation reflects this layering — three separate symmetry mechanisms, two cost formulas, confusing for users.

## Code Changes

### Remove from `_einsum.py`

- Delete `_detect_repeated_operands()` function
- Delete `optimize=False` code path — ALL einsums go through opt_einsum
- Budget deduction changes from `einsum_cost(full_expression)` to `path_info.optimized_cost`

The new `einsum()` flow:

```
1. Extract SymmetryInfo from SymmetricTensor inputs
2. Convert SymmetryInfo → IndexSymmetry
3. contract_path(subscripts, *shapes, ..., input_symmetries=...) → (path, PathInfo)
4. budget.deduct(path_info.optimized_cost)
5. Execute pairwise steps
6. Optionally wrap output as SymmetricTensor
```

### Rewrite `_flops.py` — `einsum_cost()`

Remove the hand-rolled formula. The function becomes a thin wrapper around `_opt_einsum.contract_path`:

```python
def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    operand_symmetries: list[SymmetryInfo | None] | None = None,
) -> int:
    """FLOP cost of an einsum operation.

    Delegates to the symmetry-aware opt_einsum path optimizer.
    Returns the total cost along the optimal contraction path.
    """
    # Convert SymmetryInfo → IndexSymmetry
    index_syms = _convert_symmetries(operand_symmetries, subscripts) if operand_symmetries else None
    _, path_info = contract_path(subscripts, *shapes, shapes=True, input_symmetries=index_syms)
    return path_info.optimized_cost
```

**Removed parameters:**
- `repeated_operand_indices` — dropped (was the `id()` hack)
- `symmetric_dims` — output symmetry is declared on the tensor via `as_symmetric`, not as a cost parameter

**Kept parameters:**
- `subscripts`, `shapes` — unchanged
- `operand_symmetries` — converted to IndexSymmetry and passed to contract_path

### FLOP convention change

Switch from mechestim's "product of labels" to opt_einsum's standard convention that includes `op_factor`:

| Operation | Old cost | New cost | Why |
|-----------|---------|---------|-----|
| `ij,jk->ik` (3,4)×(4,5) | 60 | 120 | op_factor=2 (multiply + add for inner product) |
| `ii->` trace (10,10) | 10 | 20 | op_factor=2 (inner product) |
| `i,j->ij` outer (3,)×(4,) | 12 | 12 | op_factor=1 (no summation) |
| `bij,bjk->bik` batch matmul | 120 | 240 | op_factor=2 |

This is the standard convention: a multiply-accumulate counts as 2 FLOPs (one multiply, one add). The old convention counted only the number of output elements × products, which underestimated by ~2× for contractions with summation.

### Update `_pointwise.py` — `dot()` and `matmul()`

These currently call `einsum_cost` directly. They'll use the simplified version. Since they're always 2-operand, the cost is a single-step from contract_path.

### Test updates

All tests asserting exact FLOP costs need updating:

**`test_einsum.py`:**
- `test_matmul_flop_cost`: 60 → 120
- `test_trace`: 10 → 20
- `test_outer_product`: 12 → 12 (unchanged — no summation)
- `test_batch_matmul`: 120 → 240
- `test_repeated_operand_symmetry`: DELETE (repeated operand detection removed)
- `test_no_symmetry_different_objects`: DELETE (same)
- `test_symmetric_dims_valid`: update cost to match new formula

**`test_symmetric_einsum.py`:**
- `test_symmetric_input_reduces_cost`: 55 → new value from opt_einsum
- `test_plain_input_unchanged`: 100 → new value from opt_einsum

**`test_flops.py`:**
- All einsum_cost tests: update expected values
- Remove repeated_operand tests entirely
- Update symmetric input test values

**`test_einsum_integration.py`:**
- `test_existing_2_operand_behavior`: 60 → 120
- Other tests: verify directional (sym < dense) rather than exact values where possible

## Documentation Rewrite

The documentation currently tells a confusing story with three separate symmetry mechanisms, legacy formulas, and mixed terminology. It needs to be rewritten with a single narrative that a new user can follow linearly.

### Narrative structure (the story we tell)

**The one key concept:** mechestim counts FLOPs analytically. For einsum, it finds the optimal contraction path and sums the per-step costs. When inputs are symmetric, both the path ordering and the per-step costs account for symmetry.

**No more "three mechanisms."** The old docs described three separate symmetry mechanisms (SymmetricTensor, symmetric_dims, repeated operands). The new docs describe one: wrap your tensor with `as_symmetric()`, and everything else is automatic.

### `docs/concepts/flop-counting-model.md` — Full rewrite of einsum section

The cost formula table needs updating for the convention change:

```
| **Einsum** | Product of all index dimensions × op_factor | `'ij,jk->ik'` → 2 × 3 × 4 × 5 = 120 |
```

The symmetry savings table stays but drops the "Einsum (symmetric output) — Divided by group factorial" row — output symmetry is now handled by declaring it on the tensor, not as a formula parameter.

Remove "Multi-operand einsum cost" as a separate section. Instead, make it the *primary* explanation: "einsum always decomposes into pairwise steps along the optimal path. The cost is the sum of step costs."

### `docs/how-to/use-einsum.md` — Simplify dramatically

**Current structure** (6 sections):
1. Common patterns
2. Cost formula
3. dot/matmul
4. Advanced: symmetric tensors (3 sub-mechanisms)
5. Multi-operand contractions
6. Inspecting contraction paths

**New structure** (5 sections):
1. **Common patterns** — same, but update FLOP numbers for new convention
2. **Cost formula** — "The cost of an einsum is the sum of per-step costs along the optimal contraction path. For a single pairwise contraction, the cost is `product_of_all_index_dims × op_factor` where op_factor accounts for multiply-accumulate." One formula, not three.
3. **Symmetric tensors** — one section: "Wrap your tensor with `me.as_symmetric(data, dims)`. The optimizer automatically finds the cheapest path and charges reduced costs. That's it." No sub-sections for repeated operands, symmetric_dims, etc.
4. **Inspecting costs** — `me.einsum_path()` and `me.flops.einsum_cost()` — both return the same numbers because they use the same system.
5. **Common pitfalls** — simplified

### `docs/how-to/exploit-symmetry.md` — Remove einsum-specific legacy

**Current structure:**
- SymmetricTensor
- Declaring symmetric output dimensions
- Repeated operands (auto-detected)
- Combined savings formula
- Symmetry propagation through contraction paths

**New structure:**
- **SymmetricTensor** — creating, automatic cost savings, propagation rules (keep as-is, this is good)
- **Symmetry-aware linalg** — keep as-is
- **Symmetry in einsum** — one clean section: "When you pass a SymmetricTensor to einsum, the path optimizer uses symmetry to choose the contraction order and reduce per-step costs. Use `symmetric_dims` on einsum if you know the output is symmetric — this wraps the result as a SymmetricTensor for downstream savings."
- **Symmetry propagation through contraction paths** — keep the S₃→S₂→none chain explanation, it's valuable
- Drop "Repeated operands" entirely
- Drop "Combined savings" formula (no more factorial stacking)

### `docs/how-to/plan-your-budget.md` — Update einsum_cost example

Update the `me.flops.einsum_cost()` call to use the new signature (no `repeated_operand_indices`). Update expected values for the new convention.

### `docs/api/symmetric.md` — Minor updates

- PathInfo/StepInfo docs stay
- Update any references to "product of all labels" formula

### `docs/reference/cheat-sheet.md` — Auto-generated

This is auto-generated by `scripts/generate_api_docs.py`. The einsum entry will need to show the updated formula with op_factor. May need to update the generation script.

### `docs/getting-started/first-budget.md` — Update FLOP numbers

The example shows `einsum 655,360 (99.6%) [10 calls]` for 10 layers of (256,256)×(256,) matmul. With the convention change, this becomes `1,310,720` (2× for op_factor). Update the sample output.

## Testing

### Cost correctness
- Every test asserting an exact FLOP count is updated for the new convention
- No legacy `repeated_operand_indices` or `symmetric_dims` parameter usage
- `einsum_cost(subscripts, shapes)` matches `contract_path(subscripts, *shapes, shapes=True).optimized_cost` for all test cases

### Backward compatibility
- `einsum()` still accepts `symmetric_dims` kwarg for output wrapping (this stays — it's about output type, not cost)
- `einsum_path()` still works
- `dot()` and `matmul()` still work with updated costs
- All existing functionality preserved, just with different (more accurate) costs

### Documentation
- Every code example in docs produces correct output when copy-pasted
- No references to `repeated_operand_indices` or the factorial formula anywhere in docs
- `me.flops.einsum_cost()` examples show correct values
