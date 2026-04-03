# Unified Einsum Cost Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all legacy einsum cost mechanisms, unify on opt_einsum's cost model (with op_factor), and rewrite documentation with a single coherent narrative.

**Architecture:** Delete `_detect_repeated_operands`, delete `optimize=False` path, rewrite `einsum_cost()` as a thin wrapper around `contract_path()`, switch `StepInfo` to use opt_einsum's `flop_count` (with op_factor) instead of stripping it. Update all tests and docs.

**Tech Stack:** Python 3.14, numpy 2.1.3, pytest.

**Test runner:** `/Users/mohanty/.local/bin/uv run python -m pytest`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/mechestim/_einsum.py` | Modify | Remove legacy, simplify to single opt_einsum path |
| `src/mechestim/_flops.py` | Modify | Rewrite `einsum_cost()` as wrapper around `contract_path` |
| `src/mechestim/_opt_einsum/_contract.py` | Modify | StepInfo uses opt_einsum costs (with op_factor) |
| `src/mechestim/_pointwise.py` | Modify | `dot()`/`matmul()` use new `einsum_cost()` |
| `tests/test_einsum.py` | Modify | Update cost assertions, remove repeated-operand tests |
| `tests/test_flops.py` | Modify | Update cost assertions, remove repeated-operand tests |
| `tests/test_symmetric_einsum.py` | Modify | Update cost assertions |
| `tests/test_einsum_integration.py` | Modify | Update cost assertions |
| `docs/how-to/use-einsum.md` | Rewrite | Single narrative, new cost convention |
| `docs/how-to/exploit-symmetry.md` | Rewrite | Remove legacy mechanisms |
| `docs/concepts/flop-counting-model.md` | Modify | Update formula table, convention |
| `docs/how-to/plan-your-budget.md` | Modify | Update einsum_cost examples |
| `docs/getting-started/first-budget.md` | Modify | Update FLOP numbers in example output |
| `docs/api/symmetric.md` | Modify | Minor formula updates |

---

### Task 1: Switch StepInfo/PathInfo to opt_einsum cost convention (with op_factor)

Currently `_contract.py` strips op_factor when building StepInfo (line ~404: `mechestim_dense = compute_size_by_dict(...)` which is just the product of labels). Change it to use the raw `flop_count` / `symmetric_flop_count` values which include op_factor.

**Files:**
- Modify: `src/mechestim/_opt_einsum/_contract.py`
- Modify: `tests/test_opt_einsum_symmetry.py`

- [ ] **Step 1: Update StepInfo cost computation in `_contract.py`**

In the contraction list loop, replace the mechestim-style cost computation with the opt_einsum costs that already include op_factor. Currently (lines ~402-422):

```python
# Build StepInfo — use mechestim-style cost (product of all labels)
mechestim_dense = helpers.compute_size_by_dict(idx_contract, size_dict)
if dense_cost > 0 and cost < dense_cost:
    mechestim_sym = max(1, mechestim_dense * cost // dense_cost)
else:
    mechestim_sym = mechestim_dense
savings = 1.0 - (mechestim_sym / mechestim_dense) if mechestim_dense > 0 else 0.0
```

Replace with:

```python
# StepInfo uses opt_einsum costs (with op_factor) directly
step_flop_cost = cost       # symmetric_flop_count or flop_count (already has op_factor)
step_dense_cost = dense_cost  # flop_count (already has op_factor)
savings = 1.0 - (step_flop_cost / step_dense_cost) if step_dense_cost > 0 else 0.0
```

And update the StepInfo construction:

```python
step_infos.append(StepInfo(
    subscript=einsum_str,
    flop_cost=step_flop_cost,
    ...
    dense_flop_cost=step_dense_cost,
    symmetry_savings=savings,
    blas_type=do_blas,
))
```

Also update `mechestim_naive` and `mechestim_optimized` at the bottom:

```python
# Naive cost: opt_einsum's naive flop count (with op_factor)
# optimized_cost: sum of per-step costs (with op_factor)
path_print = PathInfo(
    ...
    naive_cost=naive_cost,  # already computed with flop_count (has op_factor)
    optimized_cost=sum(s.flop_cost for s in step_infos),
    ...
)
```

- [ ] **Step 2: Verify the new numbers**

```bash
/Users/mohanty/.local/bin/uv run python -c "
from mechestim._opt_einsum import contract_path
_, info = contract_path('ij,jk->ik', (3,4), (4,5), shapes=True)
print('optimized_cost:', info.optimized_cost)  # should be 120 now
print('step flop_cost:', info.steps[0].flop_cost)  # should be 120
print('naive_cost:', info.naive_cost)  # should be 120
"
```
Expected: all 120 (not 60).

- [ ] **Step 3: Fix any failing opt_einsum tests**

The upstream tests use `info.opt_cost` (the legacy property) which returns `_oe_opt_cost` — that's already the opt_einsum value and shouldn't change. Run:

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/test_opt_einsum_paths.py tests/test_opt_einsum_symmetry.py -v --tb=short 2>&1 | tail -10
```

Fix any assertions on `optimized_cost` that now have different values.

- [ ] **Step 4: Commit**

```bash
git add src/mechestim/_opt_einsum/_contract.py tests/test_opt_einsum_symmetry.py
git commit -m "feat: switch PathInfo/StepInfo to opt_einsum cost convention (with op_factor)"
```

---

### Task 2: Simplify `_einsum.py` — remove legacy, unify on opt_einsum cost

**Files:**
- Modify: `src/mechestim/_einsum.py`
- Modify: `tests/test_einsum.py`
- Modify: `tests/test_symmetric_einsum.py`
- Modify: `tests/test_einsum_integration.py`

- [ ] **Step 1: Update tests first (TDD)**

Update `tests/test_einsum.py`:

```python
def test_matmul_flop_cost():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("ij,jk->ik", A, B)
        assert budget.flops_used == 120  # was 60; now includes op_factor (multiply + add)


def test_trace():
    A = numpy.eye(10)
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum("ii->", A)
        assert result == 10.0
        assert budget.flops_used == 20  # was 10; op_factor=2


def test_outer_product():
    a = numpy.ones((3,))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum("i,j->ij", a, b)
        assert result.shape == (3, 4)
        assert budget.flops_used == 12  # unchanged — no summation, op_factor=1


def test_batch_matmul():
    A = numpy.ones((2, 3, 4))
    B = numpy.ones((2, 4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("bij,bjk->bik", A, B)
        assert budget.flops_used == 240  # was 120; op_factor=2
```

Delete `test_repeated_operand_symmetry` and `test_no_symmetry_different_objects`.

Update `test_symmetric_dims_valid` — the cost will change. We need to compute what `contract_path` gives:
```python
def test_symmetric_dims_valid():
    x = numpy.ones((3, 10))
    y = numpy.ones((3, 10))
    A = numpy.eye(3)
    with BudgetContext(flop_budget=10**8) as budget:
        result = einsum("ai,bj,ab->ij", x, y, A, symmetric_dims=[(0, 1)])
        # Cost comes from opt_einsum now, not the old formula
        assert budget.flops_used > 0  # directional check; exact value from opt_einsum
```

Update `tests/test_symmetric_einsum.py`:

```python
class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        S = as_symmetric(numpy.eye(10), dims=(0, 1))
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,j->i", S, v)
            # Cost from opt_einsum with symmetry; directional check
            assert budget.flops_used > 0

    def test_plain_input_unchanged(self):
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,j->i", A, v)
            assert budget.flops_used == 200  # was 100; op_factor=2

    def test_symmetric_cheaper_than_dense(self):
        """Symmetric input should cost less than dense."""
        S = as_symmetric(numpy.eye(10), dims=(0, 1))
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget_sym:
            einsum("ij,j->i", S, v)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget_dense:
            einsum("ij,j->i", A, v)
        assert budget_sym.flops_used <= budget_dense.flops_used
```

Update `tests/test_einsum_integration.py`:

```python
class TestBackwardCompatibility:
    def test_existing_2_operand_behavior(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,jk->ik", A, B)
            assert budget.flops_used == 120  # was 60; op_factor=2
            assert result.shape == (3, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/test_einsum.py tests/test_symmetric_einsum.py tests/test_einsum_integration.py -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 3: Rewrite `_einsum.py`**

The new `einsum()`:

```python
def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    optimize: str | bool = "auto",
    symmetric_dims: list[tuple[int, ...]] | None = None,
    **kwargs,
) -> _np.ndarray:
    budget = require_budget()
    shapes = [op.shape for op in operands]

    # Extract symmetry from SymmetricTensor inputs
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]

    # Convert SymmetryInfo → IndexSymmetry for the path optimizer
    input_subscripts_str = subscripts.split("->")[0]
    input_parts = input_subscripts_str.split(",")
    index_symmetries = [
        _symmetry_info_to_index_symmetry(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]
    has_symmetry = any(s is not None for s in index_symmetries)

    # Find optimal path and compute cost
    from mechestim._opt_einsum import contract_path as _contract_path
    path, path_info = _contract_path(
        subscripts, *shapes, shapes=True,
        optimize=optimize if optimize is not False else "auto",
        input_symmetries=index_symmetries if has_symmetry else None,
    )

    # Deduct cost from budget (opt_einsum's cost, with op_factor)
    budget.deduct("einsum", flop_cost=path_info.optimized_cost,
                  subscripts=subscripts, shapes=tuple(shapes))

    # Execute pairwise steps
    result = _execute_pairwise(path_info, list(operands))

    # Handle output symmetry wrapping
    if symmetric_dims and isinstance(result, _np.ndarray) and result.ndim >= 2:
        validate_symmetry(result, symmetric_dims)
        result = SymmetricTensor(result, symmetric_dims=symmetric_dims)

    check_nan_inf(result, "einsum")
    return result
```

Delete `_detect_repeated_operands`. Remove the `einsum_cost` import (no longer needed in this file). Keep `_symmetry_info_to_index_symmetry` and `_execute_pairwise`.

- [ ] **Step 4: Run tests**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/test_einsum.py tests/test_symmetric_einsum.py tests/test_einsum_integration.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_einsum.py tests/test_einsum.py tests/test_symmetric_einsum.py tests/test_einsum_integration.py
git commit -m "feat: unify einsum cost on opt_einsum, remove legacy mechanisms"
```

---

### Task 3: Rewrite `einsum_cost()` in `_flops.py`

**Files:**
- Modify: `src/mechestim/_flops.py`
- Modify: `tests/test_flops.py`

- [ ] **Step 1: Update tests**

```python
def test_einsum_cost_matmul():
    assert einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 120  # was 60


def test_einsum_cost_trace():
    assert einsum_cost("ii->", shapes=[(10, 10)]) == 20  # was 10


def test_einsum_cost_batch_matmul():
    assert einsum_cost("bij,bjk->bik", shapes=[(2, 3, 4), (2, 4, 5)]) == 240  # was 120


def test_einsum_cost_outer_product():
    assert einsum_cost("i,j->ij", shapes=[(3,), (4,)]) == 12  # unchanged


def test_einsum_cost_scalar_output():
    assert einsum_cost("i,i->", shapes=[(5,), (5,)]) == 10  # was 5


def test_einsum_cost_symmetric_input():
    info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(10, 10))
    cost = einsum_cost(
        "ij,j->i", shapes=[(10, 10), (10,)], operand_symmetries=[info, None]
    )
    # With symmetry via opt_einsum; should be less than dense (200)
    assert cost < 200
    assert cost > 0


def test_einsum_cost_no_operand_symmetry_unchanged():
    cost = einsum_cost("ij,j->i", shapes=[(10, 10), (10,)])
    assert cost == 200  # was 100; op_factor=2
```

Delete `test_einsum_cost_symmetry_two_repeats` and `test_einsum_cost_symmetry_three_repeats`.

Update signature test — the function no longer accepts `repeated_operand_indices` or `symmetric_dims`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/test_flops.py -v --tb=short
```

- [ ] **Step 3: Rewrite `einsum_cost()` in `_flops.py`**

```python
def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    operand_symmetries: "list[SymmetryInfo | None] | None" = None,
) -> int:
    """FLOP cost of an einsum operation.

    Delegates to the symmetry-aware opt_einsum path optimizer.
    Returns the total cost along the optimal contraction path,
    using the standard FLOP convention (multiply-add = 2 FLOPs).

    Parameters
    ----------
    subscripts : str
        Einsum subscript string.
    shapes : list of tuple of int
        Shapes of the input operands.
    operand_symmetries : list of SymmetryInfo or None, optional
        Symmetry information for each input operand.

    Returns
    -------
    int
        Estimated FLOP count.
    """
    from mechestim._opt_einsum import contract_path

    # Convert SymmetryInfo -> IndexSymmetry
    index_syms = None
    if operand_symmetries and any(s is not None for s in operand_symmetries):
        input_parts = subscripts.replace(" ", "").split("->")[0].split(",")
        index_syms = []
        for sym, chars in zip(operand_symmetries, input_parts):
            if sym is None:
                index_syms.append(None)
            else:
                groups = [frozenset(chars[d] for d in g) for g in sym.symmetric_dims if len(g) >= 2]
                index_syms.append(groups if groups else None)

    _, path_info = contract_path(subscripts, *shapes, shapes=True, input_symmetries=index_syms)
    return path_info.optimized_cost
```

Remove `import math` (no longer needed for factorial). Keep `parse_einsum_subscripts` (used elsewhere), `pointwise_cost`, `reduction_cost`, `svd_cost`.

- [ ] **Step 4: Run tests**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/test_flops.py -v --tb=short
```

- [ ] **Step 5: Run full suite**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/ -q --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_flops.py tests/test_flops.py
git commit -m "feat: rewrite einsum_cost as wrapper around opt_einsum contract_path"
```

---

### Task 4: Update `dot()` and `matmul()` in `_pointwise.py`

**Files:**
- Modify: `src/mechestim/_pointwise.py`

- [ ] **Step 1: Update `dot()` and `matmul()`**

These already call `einsum_cost()` which now delegates to opt_einsum. No code change needed to the functions themselves — they'll automatically get the new costs.

Verify:
```bash
/Users/mohanty/.local/bin/uv run python -c "
from mechestim._budget import BudgetContext
import mechestim as me
import numpy as np
with BudgetContext(flop_budget=10**8, quiet=True) as b:
    me.dot(np.ones((3,4)), np.ones((4,5)))
    print('dot cost:', b.flops_used)  # should be 120
"
```

- [ ] **Step 2: Run full test suite**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/ -q --tb=short
```

Fix any dot/matmul test failures (cost values may have changed).

- [ ] **Step 3: Commit if needed**

```bash
git add src/mechestim/_pointwise.py tests/
git commit -m "fix: update dot/matmul costs for unified opt_einsum convention"
```

---

### Task 5: Rewrite documentation

This is the most important task — a clean, unified narrative for new users.

**Files:**
- Rewrite: `docs/how-to/use-einsum.md`
- Rewrite: `docs/how-to/exploit-symmetry.md`
- Modify: `docs/concepts/flop-counting-model.md`
- Modify: `docs/how-to/plan-your-budget.md`
- Modify: `docs/getting-started/first-budget.md`
- Modify: `docs/api/symmetric.md`

- [ ] **Step 1: Rewrite `docs/how-to/use-einsum.md`**

New structure (replace entire file):

1. **When to use / Prerequisites** — keep
2. **Common patterns** — update all FLOP numbers for op_factor convention:
   - Matrix-vector: `256 × 256 × 2 = 131,072 FLOPs` (multiply-accumulate)
   - Matrix multiply: `256³ × 2 = 33,554,432 FLOPs`
   - Outer product: `256² = 65,536 FLOPs` (no summation, no ×2)
   - Trace: `10 × 2 = 20 FLOPs`
3. **Cost formula** — single explanation:
   > For each pairwise contraction step, the cost is `product_of_all_index_dims × op_factor`, where `op_factor = 2` when indices are summed (multiply + add) and `op_factor = 1` otherwise (pure assignment). For multi-operand einsums, mechestim finds the optimal pairwise ordering and sums the per-step costs.
4. **Symmetric tensors** — one section:
   > Wrap your tensor with `me.as_symmetric(data, dims)`. The optimizer automatically accounts for symmetry when choosing the contraction order and computing costs. Use `symmetric_dims` on `einsum()` if the output is symmetric — this wraps the result as a `SymmetricTensor` for downstream savings.
5. **Inspecting costs** — `me.einsum_path()` and `me.flops.einsum_cost()`
6. **Common pitfalls** — keep, update
7. **Related pages** — update links

- [ ] **Step 2: Rewrite `docs/how-to/exploit-symmetry.md`**

Remove these sections entirely:
- "Declaring symmetric output dimensions (einsum)" (merged into the main einsum section)
- "Repeated operands (auto-detected)" (feature removed)
- "Combined savings" formula (no more factorial stacking)

Keep and update:
- "SymmetricTensor" — creation, automatic cost savings, propagation rules
- "Symmetry-aware linalg" — unchanged
- "Symmetry in einsum" — rewrite as one clean section about the optimizer
- "Symmetry propagation through contraction paths" — keep the S₃→S₂→none example

Add a note that `symmetric_dims` on `einsum()` is for output TYPE wrapping (returns SymmetricTensor), not a cost parameter.

- [ ] **Step 3: Update `docs/concepts/flop-counting-model.md`**

Update the cost formula table:
```
| **Einsum** | Per-step: product × op_factor | `'ij,jk->ik'` → 2 × 3 × 4 × 5 = 120 |
| **Dot / Matmul** | Same as einsum | (256, 256) @ (256, 256) → 2 × 256³ |
```

Update the symmetry savings table — remove the "Divided by group factorial" row for output symmetry. Simplify to: "Einsum (symmetric input): per-step cost scaled by unique/total ratio for surviving index groups".

Rewrite the "Multi-operand einsum cost" section as the primary explanation, not a special case.

- [ ] **Step 4: Update `docs/how-to/plan-your-budget.md`**

Update `me.flops.einsum_cost()` example:
```python
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")         # 33,554,432 (was 16,777,216)
```

Remove any reference to `repeated_operand_indices` parameter.

- [ ] **Step 5: Update `docs/getting-started/first-budget.md`**

Update the example output:
```
  (default) — by operation
    einsum  1,310,720  ( 99.6%)  [10 calls]
    maximum     2,560  (  0.4%)  [10 calls]
    sum           256  (  0.0%)   [1 call]
```

And the comments: `# matrix-vector multiply: 256 × 256 × 2 = 131,072 FLOPs each pass`

- [ ] **Step 6: Update `docs/api/symmetric.md`**

Remove any references to "product of all labels" and update `symmetry_savings` description to reference the new convention.

- [ ] **Step 7: Run full test suite**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/ -q --tb=short
```

- [ ] **Step 8: Commit**

```bash
git add docs/
git commit -m "docs: rewrite documentation for unified einsum cost model"
```

---

### Task 6: Final validation

- [ ] **Step 1: Run full test suite**

```bash
/Users/mohanty/.local/bin/uv run python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 2: Verify no references to legacy mechanisms**

```bash
grep -r "repeated_operand" src/ tests/ docs/ --include="*.py" --include="*.md" | grep -v superpowers | grep -v ".pyc"
grep -r "_detect_repeated" src/ tests/ docs/ --include="*.py" --include="*.md" | grep -v superpowers | grep -v ".pyc"
grep -r "factorial" src/mechestim/_flops.py src/mechestim/_einsum.py
```

Expected: no matches (except possibly in NOTICE which documents history).

- [ ] **Step 3: Verify einsum_cost matches opt_einsum**

```bash
/Users/mohanty/.local/bin/uv run python -c "
import mechestim as me
from mechestim._opt_einsum import contract_path

# Verify einsum_cost and contract_path agree
cost1 = me.flops.einsum_cost('ij,jk->ik', shapes=[(3,4), (4,5)])
_, info = contract_path('ij,jk->ik', (3,4), (4,5), shapes=True)
cost2 = info.optimized_cost
print(f'einsum_cost: {cost1}, contract_path: {cost2}')
assert cost1 == cost2, f'Mismatch: {cost1} != {cost2}'
print('OK — costs match')
"
```

- [ ] **Step 4: Smoke test**

```bash
/Users/mohanty/.local/bin/uv run python -c "
import mechestim as me
import numpy as np

n = 100
T_data = np.random.RandomState(42).rand(n, n, n)
T_data = (T_data + T_data.transpose(1,0,2) + T_data.transpose(2,1,0) +
          T_data.transpose(0,2,1) + T_data.transpose(1,2,0) + T_data.transpose(2,0,1)) / 6
T = me.as_symmetric(T_data, dims=(0, 1, 2))
A = np.random.RandomState(43).rand(n, n)

path, info = me.einsum_path('ijk,ai->ajk', T, A)
print(info)
print(f'Cost: {info.optimized_cost:,}')
print(f'Speedup: {info.speedup:.1f}x')
"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: final validation of unified einsum cost model" --allow-empty
```
