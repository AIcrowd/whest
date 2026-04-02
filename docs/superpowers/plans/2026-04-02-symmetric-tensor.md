# SymmetricTensor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SymmetricTensor` ndarray subclass that carries symmetry metadata through operations, giving automatic FLOP cost reductions for pointwise, einsum, and linalg ops.

**Architecture:** `SymmetricTensor` is a thin `np.ndarray` subclass with a `symmetric_dims` attribute. A `SymmetryInfo` frozen dataclass extracts metadata for cost functions. Pointwise factory functions, einsum, and linalg wrappers check for `SymmetricTensor` inputs and pass `SymmetryInfo` to their cost functions. Propagation follows algebraic rules: unary pointwise passes through symmetry; binary requires matching dims; reductions and most linalg ops strip symmetry.

**Tech Stack:** Python 3.11+, NumPy 2.x, pytest

**Spec:** `docs/superpowers/specs/2026-04-02-symmetric-tensor-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mechestim/_symmetric.py` (CREATE) | `SymmetryInfo` dataclass, `SymmetricTensor` ndarray subclass, `as_symmetric()` factory, `validate_symmetry()` helper |
| `src/mechestim/_flops.py` (MODIFY) | Add `symmetry_info` param to `pointwise_cost`, `reduction_cost`, `einsum_cost` |
| `src/mechestim/_pointwise.py` (MODIFY) | Update factory functions to detect `SymmetricTensor`, pass `SymmetryInfo` to cost, propagate symmetry |
| `src/mechestim/_einsum.py` (MODIFY) | Extract `SymmetryInfo` from `SymmetricTensor` inputs, return `SymmetricTensor` when output is symmetric |
| `src/mechestim/linalg/_decompositions.py` (MODIFY) | Add symmetry-aware cost paths to `eigh`, `cholesky`; validate symmetric input |
| `src/mechestim/linalg/_solvers.py` (MODIFY) | Add Cholesky cost path to `solve` when input is `SymmetricTensor` |
| `src/mechestim/linalg/_properties.py` (MODIFY) | Add Cholesky cost path to `det`, `slogdet` when input is `SymmetricTensor` |
| `src/mechestim/errors.py` (MODIFY) | No changes needed — `SymmetryError` already exists |
| `src/mechestim/__init__.py` (MODIFY) | Export `SymmetricTensor`, `SymmetryInfo`, `as_symmetric` |
| `src/mechestim/flops.py` (MODIFY) | Re-export `SymmetryInfo` |
| `tests/test_symmetric.py` (CREATE) | Tests for `SymmetryInfo`, `SymmetricTensor`, `as_symmetric` |
| `tests/test_symmetric_pointwise.py` (CREATE) | Tests for symmetry-aware pointwise ops |
| `tests/test_symmetric_einsum.py` (CREATE) | Tests for symmetry-aware einsum |
| `tests/test_symmetric_linalg.py` (CREATE) | Tests for symmetry-aware linalg ops |

---

### Task 1: `SymmetryInfo` dataclass

**Files:**
- Create: `src/mechestim/_symmetric.py`
- Create: `tests/test_symmetric.py`

- [ ] **Step 1: Write the failing tests for SymmetryInfo**

```python
# tests/test_symmetric.py
"""Tests for SymmetricTensor and SymmetryInfo."""
import math
import numpy
import pytest

from mechestim._symmetric import SymmetryInfo


class TestSymmetryInfo:
    def test_single_group_unique_elements(self):
        """A (0,1)-symmetric 5x5 matrix has 5*6/2 = 15 unique elements."""
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
        assert info.unique_elements == 15

    def test_single_group_symmetry_factor(self):
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
        assert info.symmetry_factor == 2

    def test_partial_symmetry_unique_elements(self):
        """(0,1) and (2,3) symmetric on (4,4,3,3): 4*5/2 * 3*4/2 = 60."""
        info = SymmetryInfo(symmetric_dims=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
        assert info.unique_elements == 10 * 6  # 60

    def test_partial_symmetry_factor(self):
        info = SymmetryInfo(symmetric_dims=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
        assert info.symmetry_factor == 4  # 2! * 2!

    def test_three_way_symmetry(self):
        """(0,1,2)-symmetric 3x3x3 tensor: 3! = 6 factor, 10 unique elements."""
        info = SymmetryInfo(symmetric_dims=[(0, 1, 2)], shape=(3, 3, 3))
        assert info.symmetry_factor == 6
        # C(3+3-1, 3) = C(5,3) = 10
        assert info.unique_elements == 10

    def test_mixed_symmetric_and_free_dims(self):
        """(0,1)-symmetric on (5,5,8): 15 * 8 = 120 unique elements."""
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5, 8))
        assert info.unique_elements == 120

    def test_frozen(self):
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
        with pytest.raises(AttributeError):
            info.shape = (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/mechestim/.claude/worktrees/loving-curran && python -m pytest tests/test_symmetric.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mechestim._symmetric'`

- [ ] **Step 3: Implement SymmetryInfo**

```python
# src/mechestim/_symmetric.py
"""SymmetricTensor: ndarray subclass with symmetry metadata."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np


@dataclass(frozen=True)
class SymmetryInfo:
    """Immutable descriptor of tensor symmetry, passed to cost functions.

    Parameters
    ----------
    symmetric_dims : list of tuple of int
        Groups of dimension indices that are symmetric under permutation.
        E.g., [(0, 1)] for a symmetric matrix, [(0, 1), (2, 3)] for partial symmetry.
    shape : tuple of int
        Full tensor shape.
    """
    symmetric_dims: list[tuple[int, ...]]
    shape: tuple[int, ...]

    def __post_init__(self):
        # Normalize to sorted tuples in a list
        object.__setattr__(
            self, 'symmetric_dims',
            [tuple(sorted(g)) for g in self.symmetric_dims]
        )

    @property
    def unique_elements(self) -> int:
        """Number of unique elements accounting for all symmetry groups.

        For each symmetric group of k dims with size n, the number of unique
        entries is C(n + k - 1, k) = (n+k-1)! / (k! * (n-1)!).
        Free (non-symmetric) dims contribute their full size.
        """
        symmetric_dim_indices = set()
        result = 1
        for group in self.symmetric_dims:
            symmetric_dim_indices.update(group)
            n = self.shape[group[0]]  # all dims in group have same size
            k = len(group)
            # C(n + k - 1, k)
            result *= math.comb(n + k - 1, k)
        # Multiply by free dims
        for i, s in enumerate(self.shape):
            if i not in symmetric_dim_indices:
                result *= s
        return result

    @property
    def symmetry_factor(self) -> int:
        """Product of factorials of group sizes."""
        factor = 1
        for group in self.symmetric_dims:
            factor *= math.factorial(len(group))
        return factor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/mechestim/.claude/worktrees/loving-curran && python -m pytest tests/test_symmetric.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_symmetric.py tests/test_symmetric.py
git commit -m "feat: add SymmetryInfo dataclass with unique_elements and symmetry_factor"
```

---

### Task 2: `SymmetricTensor` ndarray subclass and `as_symmetric` factory

**Files:**
- Modify: `src/mechestim/_symmetric.py`
- Modify: `tests/test_symmetric.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_symmetric.py`:

```python
from mechestim._symmetric import SymmetricTensor, as_symmetric
from mechestim.errors import SymmetryError


class TestSymmetricTensor:
    def test_is_ndarray(self):
        data = numpy.eye(3)
        S = as_symmetric(data, dims=(0, 1))
        assert isinstance(S, numpy.ndarray)
        assert isinstance(S, SymmetricTensor)

    def test_symmetric_dims_attribute(self):
        data = numpy.eye(3)
        S = as_symmetric(data, dims=(0, 1))
        assert S.symmetric_dims == [(0, 1)]

    def test_symmetry_info_property(self):
        data = numpy.eye(3)
        S = as_symmetric(data, dims=(0, 1))
        info = S.symmetry_info
        assert isinstance(info, SymmetryInfo)
        assert info.shape == (3, 3)
        assert info.symmetric_dims == [(0, 1)]

    def test_validates_symmetric(self):
        data = numpy.array([[1.0, 2.0], [2.0, 3.0]])
        S = as_symmetric(data, dims=(0, 1))
        assert isinstance(S, SymmetricTensor)

    def test_rejects_nonsymmetric(self):
        data = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(SymmetryError):
            as_symmetric(data, dims=(0, 1))

    def test_multiple_groups(self):
        # Build a (0,1) and (2,3) symmetric 4-tensor
        data = numpy.zeros((3, 3, 2, 2))
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    for l in range(2):
                        data[i, j, k, l] = (i + j) * 10 + (k + l)
        S = as_symmetric(data, dims=[(0, 1), (2, 3)])
        assert S.symmetric_dims == [(0, 1), (2, 3)]

    def test_single_tuple_dims_shorthand(self):
        """dims=(0, 1) is shorthand for dims=[(0, 1)]."""
        data = numpy.eye(3)
        S = as_symmetric(data, dims=(0, 1))
        assert S.symmetric_dims == [(0, 1)]

    def test_copy_preserves_symmetry(self):
        data = numpy.eye(3)
        S = as_symmetric(data, dims=(0, 1))
        S2 = S.copy()
        assert isinstance(S2, SymmetricTensor)
        assert S2.symmetric_dims == [(0, 1)]

    def test_shape_dtype_preserved(self):
        data = numpy.eye(4, dtype=numpy.float32)
        S = as_symmetric(data, dims=(0, 1))
        assert S.shape == (4, 4)
        assert S.dtype == numpy.float32

    def test_slice_loses_symmetry(self):
        data = numpy.eye(4)
        S = as_symmetric(data, dims=(0, 1))
        row = S[0]
        assert not isinstance(row, SymmetricTensor)

    def test_validation_tolerance(self):
        """Small deviations within tolerance should pass."""
        data = numpy.eye(3)
        data[0, 1] = 0.5
        data[1, 0] = 0.5 + 1e-8  # within atol=1e-6
        S = as_symmetric(data, dims=(0, 1))
        assert isinstance(S, SymmetricTensor)

    def test_validation_exceeds_tolerance(self):
        data = numpy.eye(3)
        data[0, 1] = 0.5
        data[1, 0] = 0.5 + 1e-3  # exceeds atol=1e-6
        with pytest.raises(SymmetryError):
            as_symmetric(data, dims=(0, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_symmetric.py::TestSymmetricTensor -v`
Expected: FAIL with `ImportError: cannot import name 'SymmetricTensor'`

- [ ] **Step 3: Implement SymmetricTensor and as_symmetric**

Add to `src/mechestim/_symmetric.py`:

```python
from mechestim.errors import SymmetryError


def validate_symmetry(data: np.ndarray, symmetric_dims: list[tuple[int, ...]]) -> None:
    """Validate that data is symmetric along declared dimension groups.

    Raises SymmetryError if any pair of swapped dimensions within a group
    produces values differing beyond tolerance.
    """
    for group in symmetric_dims:
        if len(group) < 2:
            continue
        # Check all dims in group have the same size
        sizes = [data.shape[d] for d in group]
        if len(set(sizes)) > 1:
            raise ValueError(
                f"Symmetric dims {group} must have equal sizes, "
                f"got {dict(zip(group, sizes))}"
            )
        # Check all pairwise transpositions
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                axes = list(range(data.ndim))
                axes[group[i]], axes[group[j]] = axes[group[j]], axes[group[i]]
                transposed = data.transpose(axes)
                if not np.allclose(data, transposed, atol=1e-6, rtol=1e-5):
                    max_dev = float(np.max(np.abs(data - transposed)))
                    raise SymmetryError(dims=group, max_deviation=max_dev)


class SymmetricTensor(np.ndarray):
    """ndarray subclass carrying symmetry metadata for FLOP cost reductions.

    Do not instantiate directly — use ``as_symmetric()``.
    """

    def __new__(cls, input_array, symmetric_dims: list[tuple[int, ...]]):
        obj = np.asarray(input_array).view(cls)
        obj._symmetric_dims = [tuple(sorted(g)) for g in symmetric_dims]
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._symmetric_dims = getattr(obj, '_symmetric_dims', None)

    @property
    def symmetric_dims(self) -> list[tuple[int, ...]] | None:
        return self._symmetric_dims

    @property
    def symmetry_info(self) -> SymmetryInfo:
        return SymmetryInfo(
            symmetric_dims=self._symmetric_dims,
            shape=self.shape,
        )

    def __reduce__(self):
        # Support pickling
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self._symmetric_dims,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self._symmetric_dims = state[-1]
        super().__setstate__(state[:-1])

    def __getitem__(self, key):
        result = super().__getitem__(key)
        # Slicing may break symmetry — return plain ndarray
        if isinstance(result, SymmetricTensor):
            return np.asarray(result)
        return result

    def copy(self, order='C'):
        result = super().copy(order=order)
        # copy() goes through __array_finalize__ which preserves _symmetric_dims
        return result


def as_symmetric(
    data: np.ndarray,
    dims: tuple[int, ...] | list[tuple[int, ...]],
) -> SymmetricTensor:
    """Validate and wrap an array as a SymmetricTensor.

    Parameters
    ----------
    data : numpy.ndarray
        The array to wrap.
    dims : tuple of int, or list of tuple of int
        Symmetry groups. A single tuple ``(0, 1)`` is shorthand for
        ``[(0, 1)]``. For partial symmetry: ``[(0, 1), (2, 3)]``.

    Returns
    -------
    SymmetricTensor

    Raises
    ------
    SymmetryError
        If validation fails (data not symmetric within tolerance).
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    # Normalize dims: single tuple -> list of one tuple
    if isinstance(dims, tuple) and all(isinstance(d, int) for d in dims):
        symmetric_dims = [dims]
    else:
        symmetric_dims = [tuple(g) for g in dims]
    validate_symmetry(data, symmetric_dims)
    return SymmetricTensor(data, symmetric_dims=symmetric_dims)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_symmetric.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_symmetric.py tests/test_symmetric.py
git commit -m "feat: add SymmetricTensor ndarray subclass and as_symmetric factory"
```

---

### Task 3: Export `SymmetricTensor`, `SymmetryInfo`, `as_symmetric` from public API

**Files:**
- Modify: `src/mechestim/__init__.py`
- Modify: `src/mechestim/flops.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_symmetric.py

class TestPublicAPI:
    def test_import_from_mechestim(self):
        import mechestim as me
        assert hasattr(me, 'SymmetricTensor')
        assert hasattr(me, 'SymmetryInfo')
        assert hasattr(me, 'as_symmetric')

    def test_import_symmetry_info_from_flops(self):
        from mechestim.flops import SymmetryInfo
        assert SymmetryInfo is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_symmetric.py::TestPublicAPI -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add exports**

In `src/mechestim/__init__.py`, after the `SymmetryError` import block, add:

```python
# --- Symmetric tensor ---
from mechestim._symmetric import (  # noqa: F401
    SymmetricTensor,
    SymmetryInfo,
    as_symmetric,
)
```

In `src/mechestim/flops.py`, add:

```python
from mechestim._symmetric import SymmetryInfo  # noqa: F401
```

And add `"SymmetryInfo"` to the `__all__` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_symmetric.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/__init__.py src/mechestim/flops.py
git commit -m "feat: export SymmetricTensor, SymmetryInfo, as_symmetric from public API"
```

---

### Task 4: Symmetry-aware `pointwise_cost` and `reduction_cost`

**Files:**
- Modify: `src/mechestim/_flops.py`
- Modify: `tests/test_flops.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flops.py`:

```python
from mechestim._symmetric import SymmetryInfo


def test_pointwise_cost_symmetric():
    """Symmetric 5x5 matrix: 15 unique elements instead of 25."""
    info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
    assert pointwise_cost(shape=(5, 5), symmetry_info=info) == 15


def test_pointwise_cost_partial_symmetry():
    """(0,1) and (2,3) symmetric on (4,4,3,3): 60 unique."""
    info = SymmetryInfo(symmetric_dims=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
    assert pointwise_cost(shape=(4, 4, 3, 3), symmetry_info=info) == 60


def test_pointwise_cost_no_symmetry_unchanged():
    """Without symmetry_info, cost is unchanged."""
    assert pointwise_cost(shape=(5, 5)) == 25


def test_reduction_cost_symmetric():
    """Reduction cost also benefits from symmetry."""
    info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
    assert reduction_cost(input_shape=(5, 5), axis=None, symmetry_info=info) == 15


def test_reduction_cost_no_symmetry_unchanged():
    assert reduction_cost(input_shape=(5, 5), axis=None) == 25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_flops.py::test_pointwise_cost_symmetric -v`
Expected: FAIL with `TypeError: pointwise_cost() got an unexpected keyword argument 'symmetry_info'`

- [ ] **Step 3: Update pointwise_cost and reduction_cost**

In `src/mechestim/_flops.py`, add the import at the top:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mechestim._symmetric import SymmetryInfo
```

Update `pointwise_cost`:

```python
def pointwise_cost(shape: tuple[int, ...], symmetry_info: SymmetryInfo | None = None) -> int:
    """Calculate the FLOP cost of a pointwise operation."""
    if symmetry_info is not None:
        return max(symmetry_info.unique_elements, 1)
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)
```

Update `reduction_cost`:

```python
def reduction_cost(input_shape: tuple[int, ...], axis: int | None = None, symmetry_info: SymmetryInfo | None = None) -> int:
    """Calculate the FLOP cost of a reduction operation."""
    if symmetry_info is not None:
        return max(symmetry_info.unique_elements, 1)
    result = 1
    for dim in input_shape:
        result *= dim
    return max(result, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_flops.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_flops.py tests/test_flops.py
git commit -m "feat: add symmetry_info param to pointwise_cost and reduction_cost"
```

---

### Task 5: Symmetry-aware `einsum_cost`

**Files:**
- Modify: `src/mechestim/_flops.py`
- Modify: `tests/test_flops.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flops.py`:

```python
def test_einsum_cost_symmetric_input():
    """einsum('ij,j->i', S, v) with symmetric S should cost n*(n+1)/2."""
    info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(10, 10))
    cost = einsum_cost(
        "ij,j->i",
        shapes=[(10, 10), (10,)],
        operand_symmetries=[info, None],
    )
    # base_cost = 10 * 10 = 100, S has 55 unique elements over 100 total
    # symmetry_factor from operand = 100 // 55 is not integer, so we use
    # unique_elements approach: iterate over unique combos of symmetric indices
    # For 'ij,j->i': i and j are both iterated. S is symmetric in (i,j) which
    # are the 0th and 1st dims. The cost should be n*(n+1)/2 = 55.
    assert cost == 55


def test_einsum_cost_no_operand_symmetry_unchanged():
    """Without operand_symmetries, cost is unchanged."""
    cost = einsum_cost("ij,j->i", shapes=[(10, 10), (10,)])
    assert cost == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_flops.py::test_einsum_cost_symmetric_input -v`
Expected: FAIL with `TypeError: einsum_cost() got an unexpected keyword argument 'operand_symmetries'`

- [ ] **Step 3: Update einsum_cost**

In `src/mechestim/_flops.py`, update `einsum_cost`:

```python
def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    repeated_operand_indices: list[int] | None = None,
    symmetric_dims: list[tuple[int, ...]] | None = None,
    operand_symmetries: list[SymmetryInfo | None] | None = None,
) -> int:
    """Calculate the FLOP cost of an einsum operation.

    When operand_symmetries is provided, the cost accounts for symmetric
    input operands by computing the ratio of unique elements to total elements
    and applying it as a divisor.
    """
    inputs, output = parse_einsum_subscripts(subscripts)
    label_dims: dict[str, int] = {}
    for operand_labels, shape in zip(inputs, shapes):
        for label, dim in zip(operand_labels, shape):
            if label in label_dims:
                if label_dims[label] != dim:
                    raise ValueError(f"Inconsistent dimension for label '{label}': {label_dims[label]} vs {dim}")
            else:
                label_dims[label] = dim
    all_labels = set()
    for operand_labels in inputs:
        all_labels.update(operand_labels)
    all_labels.update(output)
    base_cost = 1
    for label in all_labels:
        base_cost *= label_dims[label]

    symmetry_factor = 1
    if repeated_operand_indices and len(repeated_operand_indices) >= 2:
        symmetry_factor *= math.factorial(len(repeated_operand_indices))
    if symmetric_dims:
        for group in symmetric_dims:
            symmetry_factor *= math.factorial(len(group))

    cost = base_cost // symmetry_factor

    # Apply input operand symmetry savings
    if operand_symmetries:
        for i, sym_info in enumerate(operand_symmetries):
            if sym_info is None:
                continue
            # Compute ratio of unique to total elements for this operand
            total = 1
            for d in sym_info.shape:
                total *= d
            if total > 0:
                unique = sym_info.unique_elements
                # Scale cost by unique/total ratio for this operand
                cost = cost * unique // total

    return max(cost, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_flops.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_flops.py tests/test_flops.py
git commit -m "feat: add operand_symmetries param to einsum_cost"
```

---

### Task 6: Symmetry-aware pointwise factory functions

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Create: `tests/test_symmetric_pointwise.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_symmetric_pointwise.py
"""Tests for symmetry-aware pointwise operations."""
import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._symmetric import SymmetricTensor, as_symmetric


class TestUnarySymmetry:
    def test_exp_symmetric_cost(self):
        """exp of symmetric 10x10 matrix costs 55 (not 100)."""
        import mechestim as me
        data = numpy.eye(10)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.exp(S)
            assert budget.flops_used == 55  # 10*11/2

    def test_exp_symmetric_returns_symmetric(self):
        """exp of SymmetricTensor returns SymmetricTensor."""
        import mechestim as me
        data = numpy.eye(4)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me.exp(S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_dims == [(0, 1)]

    def test_log_symmetric_cost(self):
        """log also gets symmetry savings."""
        import mechestim as me
        data = numpy.eye(5) + 1  # ensure positive
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me.log(S)
            assert budget.flops_used == 15  # 5*6/2

    def test_plain_array_unchanged(self):
        """Plain arrays still cost numel."""
        import mechestim as me
        data = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me.exp(data)
            assert budget.flops_used == 100


class TestBinarySymmetry:
    def test_add_both_symmetric_same_dims(self):
        """Adding two symmetric matrices with same dims preserves symmetry."""
        import mechestim as me
        A = as_symmetric(numpy.eye(5), dims=(0, 1))
        B = as_symmetric(numpy.eye(5) * 2, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.add(A, B)
            assert budget.flops_used == 15
            assert isinstance(result, SymmetricTensor)

    def test_add_different_dims_no_symmetry(self):
        """Adding symmetric matrices with different dims strips symmetry."""
        import mechestim as me
        A = as_symmetric(numpy.eye(5), dims=(0, 1))
        B = numpy.ones((5, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.add(A, B)
            assert budget.flops_used == 25
            assert not isinstance(result, SymmetricTensor)

    def test_multiply_scalar_preserves_symmetry(self):
        """Multiplying symmetric tensor by scalar preserves symmetry."""
        import mechestim as me
        A = as_symmetric(numpy.eye(5), dims=(0, 1))
        scalar = numpy.asarray(3.0)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.multiply(A, scalar)
            assert isinstance(result, SymmetricTensor)


class TestReductionSymmetry:
    def test_sum_symmetric_cost(self):
        """Reduction of symmetric matrix gets cost savings."""
        import mechestim as me
        data = numpy.eye(10)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me.sum(S)
            assert budget.flops_used == 55

    def test_sum_returns_plain(self):
        """Reductions return plain arrays (rank change)."""
        import mechestim as me
        data = numpy.eye(4)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me.sum(S)
            assert not isinstance(result, SymmetricTensor)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_symmetric_pointwise.py -v`
Expected: FAIL — `budget.flops_used == 100` (not 55), result not `SymmetricTensor`

- [ ] **Step 3: Update pointwise factory functions**

In `src/mechestim/_pointwise.py`, add import at top:

```python
from mechestim._symmetric import SymmetricTensor, SymmetryInfo
```

Update `_counted_unary`:

```python
def _counted_unary(np_func, op_name: str):
    def wrapper(x):
        budget = require_budget()
        validate_ndarray(x)
        sym_info = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        cost = pointwise_cost(x.shape, symmetry_info=sym_info)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,))
        result = np_func(x)
        check_nan_inf(result, op_name)
        # Unary pointwise preserves symmetry (algebraic guarantee)
        if sym_info is not None:
            result = SymmetricTensor(result, symmetric_dims=sym_info.symmetric_dims)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    return wrapper
```

Update `_counted_binary`:

```python
def _counted_binary(np_func, op_name: str):
    def wrapper(x, y):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        # Propagate symmetry if both inputs have matching symmetric_dims
        x_sym = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        y_sym = y.symmetry_info if isinstance(y, SymmetricTensor) else None
        # Also treat scalar broadcast as preserving symmetry
        x_is_scalar = (x.ndim == 0)
        y_is_scalar = (y.ndim == 0)
        if x_sym and y_sym and x_sym.symmetric_dims == y_sym.symmetric_dims:
            out_sym_info = SymmetryInfo(symmetric_dims=x_sym.symmetric_dims, shape=output_shape)
            out_sym_dims = x_sym.symmetric_dims
        elif x_sym and y_is_scalar:
            out_sym_info = SymmetryInfo(symmetric_dims=x_sym.symmetric_dims, shape=output_shape)
            out_sym_dims = x_sym.symmetric_dims
        elif y_sym and x_is_scalar:
            out_sym_info = SymmetryInfo(symmetric_dims=y_sym.symmetric_dims, shape=output_shape)
            out_sym_dims = y_sym.symmetric_dims
        else:
            out_sym_info = None
            out_sym_dims = None
        cost = pointwise_cost(output_shape, symmetry_info=out_sym_info)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape))
        result = np_func(x, y)
        check_nan_inf(result, op_name)
        if out_sym_dims is not None:
            result = SymmetricTensor(result, symmetric_dims=out_sym_dims)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    return wrapper
```

Update `_counted_reduction`:

```python
def _counted_reduction(np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False):
    def wrapper(a, axis=None, **kwargs):
        budget = require_budget()
        validate_ndarray(a)
        sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
        cost = reduction_cost(a.shape, axis, symmetry_info=sym_info) * cost_multiplier
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
        # Reductions always return plain arrays (rank change loses symmetry)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    cost_desc = f"numel(input) * {cost_multiplier} FLOPs" if cost_multiplier > 1 else "numel(input) FLOPs"
    if extra_output:
        cost_desc += " + numel(output)"
    attach_docstring(wrapper, np_func, "counted_reduction", cost_desc)
    return wrapper
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_symmetric_pointwise.py tests/test_pointwise.py -v`
Expected: All tests PASS (both new and existing)

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_pointwise.py tests/test_symmetric_pointwise.py
git commit -m "feat: symmetry-aware pointwise ops — cost savings and propagation"
```

---

### Task 7: Symmetry-aware einsum

**Files:**
- Modify: `src/mechestim/_einsum.py`
- Create: `tests/test_symmetric_einsum.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_symmetric_einsum.py
"""Tests for symmetry-aware einsum."""
import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._einsum import einsum
from mechestim._symmetric import SymmetricTensor, as_symmetric


class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        """einsum with SymmetricTensor input gets FLOP savings."""
        S = as_symmetric(numpy.eye(10), dims=(0, 1))
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum('ij,j->i', S, v)
            assert budget.flops_used == 55  # n*(n+1)/2

    def test_plain_input_unchanged(self):
        """Plain arrays get full cost."""
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum('ij,j->i', A, v)
            assert budget.flops_used == 100


class TestEinsumSymmetricOutput:
    def test_symmetric_dims_returns_symmetric_tensor(self):
        """einsum with symmetric_dims returns SymmetricTensor."""
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            result = einsum('ki,kj->ij', X, X, symmetric_dims=[(0, 1)])
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_dims == [(0, 1)]

    def test_without_symmetric_dims_returns_plain(self):
        """einsum without symmetric_dims returns plain ndarray."""
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum('ij,jk->ik', A, B)
            assert not isinstance(result, SymmetricTensor)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_symmetric_einsum.py -v`
Expected: FAIL — cost is 100 not 55, result not `SymmetricTensor`

- [ ] **Step 3: Update einsum**

In `src/mechestim/_einsum.py`:

```python
"""Einsum with analytical FLOP counting and symmetry detection."""
from __future__ import annotations

import numpy as _np

from mechestim._flops import einsum_cost
from mechestim._symmetric import SymmetricTensor, SymmetryInfo, validate_symmetry
from mechestim._validation import check_nan_inf, require_budget
from mechestim.errors import SymmetryError


def _detect_repeated_operands(operands: tuple) -> list[int] | None:
    """Detect operands that are the same Python object (via id()).
    Returns indices of the largest repeated group, or None."""
    seen: dict[int, list[int]] = {}
    for i, op in enumerate(operands):
        obj_id = id(op)
        if obj_id not in seen:
            seen[obj_id] = []
        seen[obj_id].append(i)
    for indices in seen.values():
        if len(indices) >= 2:
            return indices
    return None


def _validate_symmetric_dims(result: _np.ndarray, symmetric_dims: list[tuple[int, ...]]) -> None:
    """Validate that the result actually has the claimed symmetry."""
    validate_symmetry(result, symmetric_dims)


def einsum(subscripts: str, *operands: _np.ndarray, symmetric_dims: list[tuple[int, ...]] | None = None) -> _np.ndarray:
    """Evaluate Einstein summation with FLOP counting.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    *operands : numpy.ndarray
        Input arrays. If any are SymmetricTensor, their symmetry is
        used to reduce FLOP cost.
    symmetric_dims : list of tuple of int, optional
        Output dimension symmetry groups. Validated at runtime.
        If validation passes, result is returned as SymmetricTensor.

    Returns
    -------
    numpy.ndarray or SymmetricTensor
    """
    budget = require_budget()
    shapes = [op.shape for op in operands]
    repeated = _detect_repeated_operands(operands)

    # Extract symmetry info from SymmetricTensor inputs
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None
        for op in operands
    ]
    has_operand_symmetry = any(s is not None for s in operand_symmetries)

    cost = einsum_cost(
        subscripts,
        shapes=list(shapes),
        repeated_operand_indices=repeated,
        symmetric_dims=symmetric_dims,
        operand_symmetries=operand_symmetries if has_operand_symmetry else None,
    )
    budget.deduct("einsum", flop_cost=cost, subscripts=subscripts, shapes=tuple(shapes))

    result = _np.einsum(subscripts, *operands)

    if symmetric_dims and isinstance(result, _np.ndarray) and result.ndim >= 2:
        _validate_symmetric_dims(result, symmetric_dims)
        result = SymmetricTensor(result, symmetric_dims=symmetric_dims)

    check_nan_inf(result, "einsum")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_symmetric_einsum.py tests/test_einsum.py -v`
Expected: All tests PASS (both new and existing)

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_einsum.py tests/test_symmetric_einsum.py
git commit -m "feat: symmetry-aware einsum — input savings and SymmetricTensor output"
```

---

### Task 8: Symmetry-aware linalg ops

**Files:**
- Modify: `src/mechestim/linalg/_decompositions.py`
- Modify: `src/mechestim/linalg/_solvers.py`
- Modify: `src/mechestim/linalg/_properties.py`
- Create: `tests/test_symmetric_linalg.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_symmetric_linalg.py
"""Tests for symmetry-aware linalg operations."""
import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._symmetric import SymmetricTensor, as_symmetric
import mechestim.linalg as la


class TestEighValidation:
    def test_eigh_accepts_symmetric_tensor(self):
        """eigh on SymmetricTensor should work without re-validating."""
        A = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        S = as_symmetric(A, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            vals, vecs = la.eigh(S)
            assert vals.shape == (2,)

    def test_eigh_returns_plain_arrays(self):
        """eigh output is not SymmetricTensor (eigenvalues are 1D)."""
        A = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        S = as_symmetric(A, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            vals, vecs = la.eigh(S)
            assert not isinstance(vals, SymmetricTensor)
            assert not isinstance(vecs, SymmetricTensor)


class TestSolveSymmetric:
    def test_solve_symmetric_uses_cholesky_cost(self):
        """solve with SymmetricTensor input should use n^3/3 cost."""
        n = 10
        A = numpy.eye(n) * 2.0
        S = as_symmetric(A, dims=(0, 1))
        b = numpy.ones(n)
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.solve(S, b)
            assert budget.flops_used == n ** 3 // 3  # Cholesky cost

    def test_solve_plain_uses_lu_cost(self):
        """solve with plain array uses n^3 cost."""
        n = 10
        A = numpy.eye(n) * 2.0
        b = numpy.ones(n)
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.solve(A, b)
            assert budget.flops_used == n ** 3  # LU cost

    def test_solve_returns_plain(self):
        """solve output is not SymmetricTensor."""
        A = numpy.eye(3) * 2.0
        S = as_symmetric(A, dims=(0, 1))
        b = numpy.ones(3)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = la.solve(S, b)
            assert not isinstance(result, SymmetricTensor)


class TestDetSymmetric:
    def test_det_symmetric_uses_cholesky_cost(self):
        """det of SymmetricTensor uses n^3/3 cost."""
        n = 10
        A = numpy.eye(n) * 2.0
        S = as_symmetric(A, dims=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.det(S)
            assert budget.flops_used == n ** 3 // 3


class TestInvSymmetric:
    def test_inv_symmetric_returns_symmetric(self):
        """inv of SymmetricTensor should return SymmetricTensor."""
        A = numpy.eye(3) * 2.0
        S = as_symmetric(A, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = la.inv(S)
            assert isinstance(result, SymmetricTensor)

    def test_inv_symmetric_cost(self):
        """inv of SymmetricTensor uses cheaper cost."""
        n = 10
        A = numpy.eye(n) * 2.0
        S = as_symmetric(A, dims=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.inv(S)
            # Cholesky + back-substitution: n^3/3 + n^3/2 = 5n^3/6
            expected = n ** 3 // 3 + n ** 3 // 2
            assert budget.flops_used == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_symmetric_linalg.py -v`
Expected: FAIL — costs don't match symmetric expectations

- [ ] **Step 3: Update solve**

In `src/mechestim/linalg/_solvers.py`, add import:

```python
from mechestim._symmetric import SymmetricTensor, as_symmetric
```

Update `solve_cost`:

```python
def solve_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of solving Ax = b for (n, n) matrix A.
    Formula: n^3 (LU) or n^3/3 (Cholesky for symmetric positive definite).
    """
    if symmetric:
        return max(n ** 3 // 3, 1)
    return max(n ** 3, 1)
```

Update `solve`:

```python
def solve(a, b):
    """Solve linear system with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"First argument must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = solve_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.solve", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.solve(a, b)
```

Update `inv_cost`:

```python
def inv_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of matrix inverse of an (n, n) matrix.
    Formula: n^3 (LU) or n^3/3 + n^3/2 (Cholesky + back-sub for symmetric).
    """
    if symmetric:
        return max(n ** 3 // 3 + n ** 3 // 2, 1)
    return max(n ** 3, 1)
```

Update `inv`:

```python
def inv(a):
    """Matrix inverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = inv_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.inv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    result = _np.linalg.inv(a)
    # Inverse of symmetric matrix is symmetric
    if is_symmetric:
        result = as_symmetric(result, dims=(0, 1))
    return result
```

- [ ] **Step 4: Update det and slogdet**

In `src/mechestim/linalg/_properties.py`, add import:

```python
from mechestim._symmetric import SymmetricTensor
```

Update `det_cost`:

```python
def det_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of determinant. Formula: n^3 (LU) or n^3/3 (Cholesky for symmetric)."""
    if symmetric:
        return max(n ** 3 // 3, 1)
    return max(n ** 3, 1)
```

Update `det`:

```python
def det(a):
    """Determinant with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = det_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.det", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.det(a)
```

Update `slogdet_cost` and `slogdet` similarly:

```python
def slogdet_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of sign and log-determinant. Same as det."""
    if symmetric:
        return max(n ** 3 // 3, 1)
    return max(n ** 3, 1)


def slogdet(a):
    """Sign and log-determinant with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = slogdet_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.slogdet", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.slogdet(a)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_symmetric_linalg.py tests/test_linalg_solvers.py tests/test_linalg_properties.py tests/test_linalg_decompositions.py -v`
Expected: All tests PASS (both new and existing)

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/linalg/_solvers.py src/mechestim/linalg/_properties.py tests/test_symmetric_linalg.py
git commit -m "feat: symmetry-aware linalg ops — solve, inv, det use Cholesky costs"
```

---

### Task 9: Update flops.py re-exports and run full test suite

**Files:**
- Modify: `src/mechestim/flops.py`

- [ ] **Step 1: Verify flops re-exports still work**

The cost functions `solve_cost`, `inv_cost`, `det_cost`, `slogdet_cost` now accept a `symmetric` param. Since it defaults to `False`, existing callers are unaffected. No re-export changes needed beyond the `SymmetryInfo` already added in Task 3.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS. No regressions.

- [ ] **Step 3: Commit any fixes if needed**

Only commit if Step 2 revealed issues that needed fixing.

---

### Task 10: Integration test — end-to-end symmetric workflow

**Files:**
- Modify: `tests/test_symmetric.py`

- [ ] **Step 1: Write the integration test**

Append to `tests/test_symmetric.py`:

```python
class TestEndToEnd:
    def test_covprop_workflow(self):
        """Simulate a covprop-like workflow: build covariance, do pointwise, solve."""
        import mechestim as me
        n, d = 5, 20
        X = numpy.random.randn(d, n)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            # Build symmetric covariance: X^T X -> symmetric
            cov = me.einsum('ki,kj->ij', X, X, symmetric_dims=[(0, 1)])
            assert isinstance(cov, SymmetricTensor)
            cov_cost = budget.flops_used

            # Pointwise on symmetric matrix — should get savings
            before = budget.flops_used
            exp_cov = me.exp(cov)
            pointwise_cost_actual = budget.flops_used - before
            assert isinstance(exp_cov, SymmetricTensor)
            assert pointwise_cost_actual == n * (n + 1) // 2  # 15

            # Solve with symmetric matrix — should use Cholesky cost
            # Make it positive definite first
            cov_pd = cov + me.multiply(me.as_symmetric(numpy.eye(n), dims=(0, 1)), numpy.asarray(float(n)))
            b = numpy.ones(n)
            before = budget.flops_used
            x = me.linalg.solve(cov_pd, b)
            solve_cost_actual = budget.flops_used - before
            assert not isinstance(x, SymmetricTensor)
            assert solve_cost_actual == n ** 3 // 3  # Cholesky

    def test_symmetry_preserved_through_chain(self):
        """Chain of unary ops preserves symmetry."""
        import mechestim as me
        data = numpy.eye(4) + 0.5
        S = me.as_symmetric(data, dims=(0, 1))

        with BudgetContext(flop_budget=10**8, quiet=True):
            r1 = me.exp(S)
            assert isinstance(r1, SymmetricTensor)
            r2 = me.log(r1)
            assert isinstance(r2, SymmetricTensor)
            r3 = me.sqrt(me.abs(r2))
            assert isinstance(r3, SymmetricTensor)

    def test_symmetry_lost_on_matmul(self):
        """Matmul does not preserve symmetry."""
        import mechestim as me
        A = me.as_symmetric(numpy.eye(3), dims=(0, 1))
        B = numpy.ones((3, 3))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.einsum('ij,jk->ik', A, B)
            assert not isinstance(result, SymmetricTensor)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_symmetric.py::TestEndToEnd -v`
Expected: All PASS

- [ ] **Step 3: Run the full test suite one final time**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_symmetric.py
git commit -m "test: add end-to-end integration tests for SymmetricTensor workflow"
```
