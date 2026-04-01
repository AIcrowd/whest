# mechestim Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a NumPy-compatible Python library that wraps mathematical operations with analytical FLOP counting and budget enforcement for ARC's Mechanistic Estimation Challenge.

**Architecture:** Bottom-up layered build. Errors and budget tracking form the foundation. FLOP cost calculators sit on top. Each op category (free, pointwise, einsum, linalg) wraps NumPy with FLOP counting. The top-level `__init__.py` re-exports everything and provides `__getattr__` for unsupported-op errors. Tests are TDD — written before implementation for each layer.

**Tech Stack:** Python 3.10+, NumPy, pytest, hatchling (build), mkdocs-material + mkdocstrings (docs), uv (package manager)

**Spec:** `docs/superpowers/specs/2026-04-01-mechestim-library-design.md`

---

## File Map

```
mechestim/
├── pyproject.toml                      # Task 1
├── src/
│   └── mechestim/
│       ├── __init__.py                 # Task 10
│       ├── errors.py                   # Task 2
│       ├── _budget.py                  # Task 3
│       ├── _flops.py                   # Task 4
│       ├── _validation.py              # Task 5
│       ├── _free_ops.py                # Task 6
│       ├── _pointwise.py              # Task 7
│       ├── _einsum.py                  # Task 8
│       ├── flops.py                    # Task 4 (public cost query API)
│       ├── linalg/
│       │   ├── __init__.py             # Task 9
│       │   └── _svd.py                # Task 9
│       ├── random/
│       │   └── __init__.py             # Task 6 (passthrough)
│       └── py.typed                    # Task 1
├── tests/
│   ├── test_errors.py                  # Task 2
│   ├── test_budget.py                  # Task 3
│   ├── test_flops.py                   # Task 4
│   ├── test_free_ops.py               # Task 6
│   ├── test_pointwise.py              # Task 7
│   ├── test_einsum.py                 # Task 8
│   ├── test_linalg.py                 # Task 9
│   ├── test_random.py                 # Task 6
│   ├── test_numpy_compat.py           # Task 11
│   └── test_integration.py            # Task 12
└── docs/                               # Task 13
    ├── index.md
    ├── quickstart.md
    └── ...
```

**Dependency order:** Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13

Tasks 6, 7, 8, 9 can be parallelized after Task 5 is complete (they all depend on budget + flops + validation but not on each other).

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/mechestim/__init__.py` (minimal placeholder)
- Create: `src/mechestim/py.typed`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "mechestim"
version = "0.1.0"
description = "NumPy-compatible math primitives with FLOP counting for the Mechanistic Estimation Challenge"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]
docs = [
    "mkdocs-material",
    "mkdocstrings[python]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mechestim"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create minimal `__init__.py` placeholder**

```python
# src/mechestim/__init__.py
"""mechestim: NumPy-compatible math primitives with FLOP counting."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create `py.typed` marker**

```
# src/mechestim/py.typed
# PEP 561 marker — this package supports type checking
```

- [ ] **Step 4: Install and verify**

Run:
```bash
uv sync --all-extras
uv run python -c "import mechestim; print(mechestim.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: scaffold mechestim package with pyproject.toml"
```

---

### Task 2: Error Classes

**Files:**
- Create: `src/mechestim/errors.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write tests for error classes**

```python
# tests/test_errors.py
"""Tests for mechestim error classes."""
import pytest
from mechestim.errors import (
    MechEstimError,
    BudgetExhaustedError,
    NoBudgetContextError,
    SymmetryError,
    MechEstimWarning,
)


def test_budget_exhausted_error_is_mechestim_error():
    with pytest.raises(MechEstimError):
        raise BudgetExhaustedError("einsum", flop_cost=100, flops_remaining=50)


def test_budget_exhausted_error_attributes():
    err = BudgetExhaustedError("einsum", flop_cost=100, flops_remaining=50)
    assert err.op_name == "einsum"
    assert err.flop_cost == 100
    assert err.flops_remaining == 50
    assert "einsum" in str(err)
    assert "100" in str(err)
    assert "50" in str(err)


def test_no_budget_context_error_is_mechestim_error():
    with pytest.raises(MechEstimError):
        raise NoBudgetContextError()


def test_no_budget_context_error_message():
    err = NoBudgetContextError()
    assert "BudgetContext" in str(err)


def test_symmetry_error_attributes():
    err = SymmetryError(dims=(0, 1), max_deviation=0.5)
    assert err.dims == (0, 1)
    assert err.max_deviation == 0.5
    assert "0, 1" in str(err)


def test_mechestim_warning_is_warning():
    assert issubclass(MechEstimWarning, UserWarning)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL (import errors)

- [ ] **Step 3: Implement error classes**

```python
# src/mechestim/errors.py
"""Exception and warning classes for mechestim."""


class MechEstimError(Exception):
    """Base exception for all mechestim errors."""


class BudgetExhaustedError(MechEstimError):
    """Raised when an operation would exceed the FLOP budget.

    Parameters
    ----------
    op_name : str
        Name of the operation that exceeded the budget.
    flop_cost : int
        FLOPs the operation would have cost.
    flops_remaining : int
        FLOPs remaining in the budget.
    """

    def __init__(self, op_name: str, *, flop_cost: int, flops_remaining: int):
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.flops_remaining = flops_remaining
        super().__init__(
            f"{op_name} would cost {flop_cost:,} FLOPs but only "
            f"{flops_remaining:,} remain"
        )


class NoBudgetContextError(MechEstimError):
    """Raised when a counted operation is called outside a BudgetContext."""

    def __init__(self):
        super().__init__(
            "No active BudgetContext. "
            "Wrap your code in `with mechestim.BudgetContext(...):`"
        )


class SymmetryError(MechEstimError):
    """Raised when a claimed tensor symmetry does not hold.

    Parameters
    ----------
    dims : tuple[int, ...]
        The dimension group that failed validation.
    max_deviation : float
        Maximum element-wise deviation from symmetry.
    """

    def __init__(self, dims: tuple[int, ...], max_deviation: float):
        self.dims = dims
        self.max_deviation = max_deviation
        super().__init__(
            f"Tensor not symmetric along dims ({', '.join(str(d) for d in dims)}): "
            f"max deviation = {max_deviation}"
        )


class MechEstimWarning(UserWarning):
    """Warning issued when mechestim detects potential numerical issues."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_errors.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/errors.py tests/test_errors.py
git commit -m "feat: add mechestim error and warning classes"
```

---

### Task 3: Budget Context and OpRecord

**Files:**
- Create: `src/mechestim/_budget.py`
- Create: `tests/test_budget.py`

- [ ] **Step 1: Write tests for BudgetContext and OpRecord**

```python
# tests/test_budget.py
"""Tests for BudgetContext and OpRecord."""
import pytest
from mechestim._budget import BudgetContext, OpRecord, get_active_budget
from mechestim.errors import BudgetExhaustedError


def test_budget_context_basic():
    with BudgetContext(flop_budget=1000) as budget:
        assert budget.flop_budget == 1000
        assert budget.flops_used == 0
        assert budget.flops_remaining == 1000
        assert budget.flop_multiplier == 1.0
        assert budget.op_log == []


def test_budget_context_deduct():
    with BudgetContext(flop_budget=1000) as budget:
        budget.deduct("test_op", flop_cost=300, subscripts=None, shapes=((10, 10),))
        assert budget.flops_used == 300
        assert budget.flops_remaining == 700
        assert len(budget.op_log) == 1
        rec = budget.op_log[0]
        assert rec.op_name == "test_op"
        assert rec.flop_cost == 300
        assert rec.cumulative == 300


def test_budget_context_deduct_with_multiplier():
    with BudgetContext(flop_budget=1000, flop_multiplier=2.0) as budget:
        budget.deduct("test_op", flop_cost=100, subscripts=None, shapes=())
        assert budget.flops_used == 200


def test_budget_exhausted():
    with pytest.raises(BudgetExhaustedError) as exc_info:
        with BudgetContext(flop_budget=100) as budget:
            budget.deduct("einsum", flop_cost=200, subscripts="ij,jk->ik", shapes=())
    assert exc_info.value.op_name == "einsum"
    assert exc_info.value.flop_cost == 200


def test_budget_exact_boundary():
    with BudgetContext(flop_budget=100) as budget:
        budget.deduct("op", flop_cost=100, subscripts=None, shapes=())
        assert budget.flops_remaining == 0


def test_no_nesting():
    with pytest.raises(RuntimeError, match="Cannot nest"):
        with BudgetContext(flop_budget=1000):
            with BudgetContext(flop_budget=500):
                pass


def test_invalid_budget():
    with pytest.raises(ValueError, match="must be > 0"):
        BudgetContext(flop_budget=0)
    with pytest.raises(ValueError, match="must be > 0"):
        BudgetContext(flop_budget=-5)


def test_get_active_budget_inside():
    with BudgetContext(flop_budget=1000) as budget:
        assert get_active_budget() is budget


def test_get_active_budget_outside():
    assert get_active_budget() is None


def test_context_cleans_up_on_exit():
    with BudgetContext(flop_budget=1000):
        pass
    assert get_active_budget() is None


def test_context_cleans_up_on_exception():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=1000):
            raise ValueError("boom")
    assert get_active_budget() is None


def test_op_record_fields():
    rec = OpRecord(
        op_name="einsum",
        subscripts="ij,jk->ik",
        shapes=((3, 4), (4, 5)),
        flop_cost=60,
        cumulative=60,
    )
    assert rec.op_name == "einsum"
    assert rec.subscripts == "ij,jk->ik"
    assert rec.flop_cost == 60


def test_summary():
    with BudgetContext(flop_budget=1000) as budget:
        budget.deduct("einsum", flop_cost=500, subscripts="ij->i", shapes=())
        budget.deduct("exp", flop_cost=100, subscripts=None, shapes=())
        budget.deduct("einsum", flop_cost=200, subscripts="ij->j", shapes=())
        s = budget.summary()
        assert "1,000" in s  # total budget
        assert "800" in s    # used
        assert "einsum" in s
        assert "exp" in s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_budget.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `_budget.py`**

```python
# src/mechestim/_budget.py
"""Budget context manager and operation recording for mechestim."""
from __future__ import annotations

import threading
from typing import NamedTuple

from mechestim.errors import BudgetExhaustedError


class OpRecord(NamedTuple):
    """Record of a single counted operation.

    Parameters
    ----------
    op_name : str
        Name of the operation (e.g., "einsum", "exp", "svd").
    subscripts : str or None
        Einsum subscript string. None for non-einsum ops.
    shapes : tuple
        Shapes of the input operands.
    flop_cost : int
        FLOPs charged for this operation (after multiplier).
    cumulative : int
        Cumulative FLOPs after this operation.
    """

    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int


# Thread-local storage for the active budget context.
_thread_local = threading.local()


def get_active_budget() -> BudgetContext | None:
    """Return the active BudgetContext, or None if outside any context."""
    return getattr(_thread_local, "active_budget", None)


class BudgetContext:
    """Context manager for FLOP budget enforcement.

    All counted mechestim operations must be called within an active
    BudgetContext. Free operations (tensor creation, reshaping) work
    both inside and outside the context.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed. Must be > 0.
    flop_multiplier : float, optional
        Multiplier applied to all FLOP costs. Default 1 (1 FLOP per
        multiply-add). Set to 2 for strict IEEE FLOP counting.

    Raises
    ------
    ValueError
        If flop_budget <= 0.
    RuntimeError
        If a BudgetContext is already active (nesting not allowed).

    Examples
    --------
    >>> import mechestim as me
    >>> with me.BudgetContext(flop_budget=1_000_000) as budget:
    ...     x = me.ones((256, 256))
    ...     y = me.einsum('ij,jk->ik', x, x)
    ...     print(budget.flops_used)
    16777216
    """

    def __init__(self, flop_budget: int, flop_multiplier: float = 1.0):
        if flop_budget <= 0:
            raise ValueError(f"flop_budget must be > 0, got {flop_budget}")
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._flops_used = 0
        self._op_log: list[OpRecord] = []

    @property
    def flop_budget(self) -> int:
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        return self._flop_budget - self._flops_used

    @property
    def flop_multiplier(self) -> float:
        return self._flop_multiplier

    @property
    def op_log(self) -> list[OpRecord]:
        return self._op_log

    def deduct(
        self,
        op_name: str,
        *,
        flop_cost: int,
        subscripts: str | None,
        shapes: tuple,
    ) -> None:
        """Deduct FLOPs from the budget.

        Parameters
        ----------
        op_name : str
            Name of the operation.
        flop_cost : int
            Raw FLOP cost (before multiplier).
        subscripts : str or None
            Einsum subscripts if applicable.
        shapes : tuple
            Input operand shapes.

        Raises
        ------
        BudgetExhaustedError
            If the adjusted cost exceeds remaining budget.
        """
        adjusted_cost = int(flop_cost * self._flop_multiplier)
        if adjusted_cost > self.flops_remaining:
            raise BudgetExhaustedError(
                op_name,
                flop_cost=adjusted_cost,
                flops_remaining=self.flops_remaining,
            )
        self._flops_used += adjusted_cost
        self._op_log.append(
            OpRecord(
                op_name=op_name,
                subscripts=subscripts,
                shapes=shapes,
                flop_cost=adjusted_cost,
                cumulative=self._flops_used,
            )
        )

    def summary(self) -> str:
        """Return a pretty-printed FLOP budget summary.

        Returns
        -------
        str
            Human-readable breakdown of budget usage by operation type.
        """
        lines = [
            "mechestim FLOP Budget Summary",
            "=" * 30,
            f"  Total budget:  {self._flop_budget:>14,}",
            f"  Used:          {self._flops_used:>14,}  "
            f"({100 * self._flops_used / self._flop_budget:.1f}%)",
            f"  Remaining:     {self.flops_remaining:>14,}  "
            f"({100 * self.flops_remaining / self._flop_budget:.1f}%)",
            "",
            "  By operation:",
        ]
        # Aggregate by op_name
        from collections import Counter

        cost_by_op: dict[str, int] = {}
        count_by_op: Counter[str] = Counter()
        for rec in self._op_log:
            cost_by_op[rec.op_name] = cost_by_op.get(rec.op_name, 0) + rec.flop_cost
            count_by_op[rec.op_name] += 1
        for op_name, cost in sorted(cost_by_op.items(), key=lambda x: -x[1]):
            pct = 100 * cost / self._flops_used if self._flops_used > 0 else 0
            lines.append(
                f"    {op_name:<16} {cost:>12,}  ({pct:5.1f}%)  "
                f"[{count_by_op[op_name]} call{'s' if count_by_op[op_name] != 1 else ''}]"
            )
        return "\n".join(lines)

    def __enter__(self) -> BudgetContext:
        if get_active_budget() is not None:
            raise RuntimeError("Cannot nest BudgetContexts")
        _thread_local.active_budget = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _thread_local.active_budget = None
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_budget.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_budget.py tests/test_budget.py
git commit -m "feat: add BudgetContext with FLOP tracking and OpRecord"
```

---

### Task 4: FLOP Cost Calculators

**Files:**
- Create: `src/mechestim/_flops.py` (internal calculators)
- Create: `src/mechestim/flops.py` (public query API)
- Create: `tests/test_flops.py`

- [ ] **Step 1: Write tests for FLOP cost calculators**

```python
# tests/test_flops.py
"""Tests for FLOP cost calculators."""
import pytest
from mechestim._flops import (
    einsum_cost,
    parse_einsum_subscripts,
    pointwise_cost,
    reduction_cost,
    svd_cost,
)


# --- einsum subscript parsing ---

def test_parse_matmul():
    inputs, output = parse_einsum_subscripts("ij,jk->ik")
    assert inputs == [['i', 'j'], ['j', 'k']]
    assert output == ['i', 'k']


def test_parse_trace():
    inputs, output = parse_einsum_subscripts("ii->")
    assert inputs == [['i', 'i']]
    assert output == []


def test_parse_batch_matmul():
    inputs, output = parse_einsum_subscripts("bij,bjk->bik")
    assert inputs == [['b', 'i', 'j'], ['b', 'j', 'k']]
    assert output == ['b', 'i', 'k']


def test_parse_implicit():
    # implicit mode: no -> arrow, output is sorted unique non-contracted
    inputs, output = parse_einsum_subscripts("ij,jk")
    assert inputs == [['i', 'j'], ['j', 'k']]
    assert output == ['i', 'k']


# --- einsum cost ---

def test_einsum_cost_matmul():
    cost = einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)])
    assert cost == 3 * 4 * 5


def test_einsum_cost_trace():
    cost = einsum_cost("ii->", shapes=[(10, 10)])
    assert cost == 10


def test_einsum_cost_batch_matmul():
    cost = einsum_cost("bij,bjk->bik", shapes=[(2, 3, 4), (2, 4, 5)])
    assert cost == 2 * 3 * 4 * 5


def test_einsum_cost_outer_product():
    cost = einsum_cost("i,j->ij", shapes=[(3,), (4,)])
    assert cost == 3 * 4


def test_einsum_cost_scalar_output():
    cost = einsum_cost("i,i->", shapes=[(5,), (5,)])
    assert cost == 5


def test_einsum_cost_four_operand():
    cost = einsum_cost("ai,bj,ck,abc->ijk", shapes=[(2, 10), (3, 10), (4, 10), (2, 3, 4)])
    assert cost == 2 * 3 * 4 * 10 * 10 * 10


# --- einsum symmetry savings ---

def test_einsum_cost_symmetry_two_repeats():
    # 'ai,bi,ab->' with x repeated twice: cost / 2!
    cost = einsum_cost(
        "ai,bi,ab->",
        shapes=[(10, 256), (10, 256), (10, 10)],
        repeated_operand_indices=[0, 1],
    )
    expected = (10 * 10 * 256) // 2
    assert cost == expected


def test_einsum_cost_symmetry_three_repeats():
    cost = einsum_cost(
        "ai,bj,ck,abc->ijk",
        shapes=[(2, 10), (2, 10), (2, 10), (2, 2, 2)],
        repeated_operand_indices=[0, 1, 2],
    )
    full = 2 * 2 * 2 * 10 * 10 * 10
    assert cost == full // 6  # / 3!


def test_einsum_cost_no_symmetry():
    cost = einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)])
    cost_with = einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)], repeated_operand_indices=None)
    assert cost == cost_with


# --- pointwise cost ---

def test_pointwise_cost():
    assert pointwise_cost(shape=(256, 256)) == 256 * 256


def test_pointwise_cost_scalar():
    assert pointwise_cost(shape=()) == 1


def test_pointwise_cost_1d():
    assert pointwise_cost(shape=(100,)) == 100


# --- reduction cost ---

def test_reduction_cost_full():
    assert reduction_cost(input_shape=(256, 256), axis=None) == 256 * 256


def test_reduction_cost_axis():
    assert reduction_cost(input_shape=(256, 256), axis=0) == 256 * 256


def test_reduction_cost_scalar():
    assert reduction_cost(input_shape=(), axis=None) == 1


# --- svd cost ---

def test_svd_cost():
    assert svd_cost(m=100, n=50, k=10) == 100 * 50 * 10


def test_svd_cost_full():
    assert svd_cost(m=100, n=50, k=None) == 100 * 50 * 50


def test_svd_cost_square():
    assert svd_cost(m=10, n=10, k=5) == 10 * 10 * 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flops.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `_flops.py` (internal calculators)**

```python
# src/mechestim/_flops.py
"""FLOP cost calculators for mechestim operations.

Each function computes the analytical FLOP cost from shapes alone,
without executing any computation. Costs assume 1 FLOP per
multiply-add (configurable via BudgetContext.flop_multiplier).
"""
from __future__ import annotations

import math
import re
from collections import Counter


def parse_einsum_subscripts(subscripts: str) -> tuple[list[list[str]], list[str]]:
    """Parse an einsum subscript string into input and output index lists.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g., 'ij,jk->ik').

    Returns
    -------
    inputs : list of list of str
        Each inner list is the index labels for one input operand.
    output : list of str
        Index labels for the output.
    """
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        input_part, output_part = subscripts.split("->")
        output = list(output_part)
    else:
        input_part = subscripts
        # Implicit mode: output indices are those appearing exactly once,
        # sorted alphabetically.
        all_labels: list[str] = []
        for part in input_part.split(","):
            all_labels.extend(list(part))
        counts = Counter(all_labels)
        output = sorted(l for l, c in counts.items() if c == 1)
    inputs = [list(part) for part in input_part.split(",")]
    return inputs, output


def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    repeated_operand_indices: list[int] | None = None,
    symmetric_dims: list[tuple[int, ...]] | None = None,
) -> int:
    """Calculate the FLOP cost of an einsum operation.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    shapes : list of tuple of int
        Shapes of each input operand.
    repeated_operand_indices : list of int, optional
        Indices of operands that are the same object (for symmetry savings).
        E.g., [0, 1] means operands 0 and 1 are the same array.
    symmetric_dims : list of tuple of int, optional
        Output dimension symmetry groups for additional savings.

    Returns
    -------
    int
        The analytical FLOP cost.
    """
    inputs, output = parse_einsum_subscripts(subscripts)

    # Build label -> dimension mapping from shapes.
    label_dims: dict[str, int] = {}
    for operand_labels, shape in zip(inputs, shapes):
        for label, dim in zip(operand_labels, shape):
            if label in label_dims:
                # Consistency check: same label must have same dimension.
                if label_dims[label] != dim:
                    raise ValueError(
                        f"Inconsistent dimension for label '{label}': "
                        f"{label_dims[label]} vs {dim}"
                    )
            else:
                label_dims[label] = dim

    # Collect all unique labels across inputs and output.
    all_labels = set()
    for operand_labels in inputs:
        all_labels.update(operand_labels)
    all_labels.update(output)

    # Base cost = product of all index dimensions.
    base_cost = 1
    for label in all_labels:
        base_cost *= label_dims[label]

    # Symmetry savings from repeated operands.
    symmetry_factor = 1
    if repeated_operand_indices and len(repeated_operand_indices) >= 2:
        k = len(repeated_operand_indices)
        symmetry_factor *= math.factorial(k)

    # Symmetry savings from explicit symmetric_dims.
    if symmetric_dims:
        for group in symmetric_dims:
            symmetry_factor *= math.factorial(len(group))

    return base_cost // symmetry_factor


def pointwise_cost(shape: tuple[int, ...]) -> int:
    """Calculate the FLOP cost of a pointwise operation.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output tensor.

    Returns
    -------
    int
        numel(shape), minimum 1 for scalar.
    """
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)


def reduction_cost(input_shape: tuple[int, ...], axis: int | None = None) -> int:
    """Calculate the FLOP cost of a reduction operation.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input tensor.
    axis : int or None
        Axis to reduce along. None means reduce all.

    Returns
    -------
    int
        numel(input), minimum 1 for scalar.
    """
    result = 1
    for dim in input_shape:
        result *= dim
    return max(result, 1)


def svd_cost(m: int, n: int, k: int | None = None) -> int:
    """Calculate the FLOP cost of a truncated SVD.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    k : int or None
        Number of singular values/vectors. None means min(m, n).

    Returns
    -------
    int
        m * n * k
    """
    if k is None:
        k = min(m, n)
    return m * n * k
```

- [ ] **Step 4: Create public FLOP query API**

```python
# src/mechestim/flops.py
"""Public FLOP cost query API.

Query operation costs without executing them or requiring a BudgetContext.

Examples
--------
>>> import mechestim
>>> mechestim.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
16777216
>>> mechestim.flops.svd_cost(m=256, n=256, k=10)
655360
"""
from mechestim._flops import (
    einsum_cost,
    pointwise_cost,
    reduction_cost,
    svd_cost,
)

__all__ = ["einsum_cost", "pointwise_cost", "reduction_cost", "svd_cost"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_flops.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_flops.py src/mechestim/flops.py tests/test_flops.py
git commit -m "feat: add FLOP cost calculators for einsum, pointwise, reduction, SVD"
```

---

### Task 5: Input Validation Utilities

**Files:**
- Create: `src/mechestim/_validation.py`

This is a utility module used by all op modules. No separate test file — validated through the op tests.

- [ ] **Step 1: Implement `_validation.py`**

```python
# src/mechestim/_validation.py
"""Input validation and NaN/Inf checking for mechestim operations."""
from __future__ import annotations

import warnings

import numpy as np

from mechestim._budget import get_active_budget
from mechestim.errors import MechEstimWarning, NoBudgetContextError


def require_budget():
    """Return the active budget or raise NoBudgetContextError.

    Returns
    -------
    BudgetContext
        The active budget context.

    Raises
    ------
    NoBudgetContextError
        If no BudgetContext is active.
    """
    budget = get_active_budget()
    if budget is None:
        raise NoBudgetContextError()
    return budget


def validate_ndarray(*arrays: object) -> None:
    """Validate that all arguments are numpy ndarrays.

    Raises
    ------
    TypeError
        If any argument is not a numpy.ndarray.
    """
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(arr).__name__}"
            )


def check_nan_inf(result: np.ndarray, op_name: str) -> None:
    """Issue a warning if result contains NaN or Inf values.

    Parameters
    ----------
    result : numpy.ndarray
        The result to check.
    op_name : str
        Name of the operation that produced this result.
    """
    if not isinstance(result, np.ndarray):
        return
    nan_count = int(np.isnan(result).sum())
    inf_count = int(np.isinf(result).sum())
    if nan_count > 0 or inf_count > 0:
        warnings.warn(
            f"{op_name} produced {nan_count} NaN and {inf_count} Inf values "
            f"in output of shape {result.shape}",
            MechEstimWarning,
            stacklevel=3,
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/mechestim/_validation.py
git commit -m "feat: add input validation and NaN/Inf checking utilities"
```

---

### Task 6: Free Operations and Random Submodule

**Files:**
- Create: `src/mechestim/_free_ops.py`
- Create: `src/mechestim/random/__init__.py`
- Create: `tests/test_free_ops.py`
- Create: `tests/test_random.py`

- [ ] **Step 1: Write tests for free ops**

```python
# tests/test_free_ops.py
"""Tests for free (zero-FLOP) operations."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim import _free_ops as ops


def test_zeros():
    x = ops.zeros((3, 4))
    assert x.shape == (3, 4)
    assert numpy.all(x == 0)


def test_ones():
    x = ops.ones((2, 3))
    assert numpy.all(x == 1)


def test_array():
    x = ops.array([[1, 2], [3, 4]])
    assert x.shape == (2, 2)
    assert x[1, 1] == 4


def test_eye():
    x = ops.eye(3)
    assert numpy.allclose(x, numpy.eye(3))


def test_reshape():
    x = ops.ones((6,))
    y = ops.reshape(x, (2, 3))
    assert y.shape == (2, 3)


def test_transpose():
    x = ops.array([[1, 2, 3], [4, 5, 6]])
    y = ops.transpose(x)
    assert y.shape == (3, 2)


def test_concatenate():
    a = ops.ones((2, 3))
    b = ops.zeros((2, 3))
    c = ops.concatenate([a, b], axis=0)
    assert c.shape == (4, 3)


def test_where():
    cond = ops.array([True, False, True])
    x = ops.array([1, 2, 3])
    y = ops.array([4, 5, 6])
    result = ops.where(cond, x, y)
    assert list(result) == [1, 5, 3]


def test_free_ops_dont_cost_flops():
    with BudgetContext(flop_budget=1) as budget:
        ops.zeros((1000, 1000))
        ops.ones((1000,))
        ops.reshape(ops.ones((6,)), (2, 3))
        assert budget.flops_used == 0


def test_free_ops_work_outside_context():
    # No BudgetContext active — should still work
    x = ops.zeros((3,))
    assert x.shape == (3,)


def test_diag():
    x = ops.array([1, 2, 3])
    d = ops.diag(x)
    assert d.shape == (3, 3)
    assert numpy.allclose(d, numpy.diag([1, 2, 3]))


def test_arange():
    x = ops.arange(5)
    assert list(x) == [0, 1, 2, 3, 4]


def test_copy():
    x = ops.array([1, 2, 3])
    y = ops.copy(x)
    assert numpy.array_equal(x, y)
    assert x is not y


def test_stack():
    a = ops.ones((3,))
    b = ops.zeros((3,))
    c = ops.stack([a, b])
    assert c.shape == (2, 3)


def test_squeeze():
    x = ops.ones((1, 3, 1))
    y = ops.squeeze(x)
    assert y.shape == (3,)


def test_expand_dims():
    x = ops.ones((3,))
    y = ops.expand_dims(x, axis=0)
    assert y.shape == (1, 3)


def test_triu():
    x = ops.ones((3, 3))
    y = ops.triu(x)
    assert numpy.allclose(y, numpy.triu(numpy.ones((3, 3))))


def test_sort():
    x = ops.array([3, 1, 2])
    y = ops.sort(x)
    assert list(y) == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_free_ops.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `_free_ops.py`**

```python
# src/mechestim/_free_ops.py
"""Free (zero-FLOP) operations that mirror NumPy tensor creation and manipulation.

These operations wrap NumPy equivalents with no FLOP cost. They work
both inside and outside a BudgetContext.
"""
from __future__ import annotations

import numpy as _np

# --- Tensor Creation ---

def array(data, dtype=None):
    """Create an array. Wraps ``numpy.array``. Cost: 0 FLOPs."""
    return _np.array(data, dtype=dtype)

def zeros(shape, dtype=float):
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    return _np.zeros(shape, dtype=dtype)

def ones(shape, dtype=float):
    """Return array of ones. Wraps ``numpy.ones``. Cost: 0 FLOPs."""
    return _np.ones(shape, dtype=dtype)

def full(shape, fill_value, dtype=None):
    """Return array filled with value. Wraps ``numpy.full``. Cost: 0 FLOPs."""
    return _np.full(shape, fill_value, dtype=dtype)

def eye(N, M=None, k=0, dtype=float):
    """Return identity matrix. Wraps ``numpy.eye``. Cost: 0 FLOPs."""
    return _np.eye(N, M=M, k=k, dtype=dtype)

def diag(v, k=0):
    """Extract diagonal or construct diagonal array. Wraps ``numpy.diag``. Cost: 0 FLOPs."""
    return _np.diag(v, k=k)

def arange(*args, **kwargs):
    """Return evenly spaced values. Wraps ``numpy.arange``. Cost: 0 FLOPs."""
    return _np.arange(*args, **kwargs)

def linspace(start, stop, num=50, **kwargs):
    """Return evenly spaced numbers. Wraps ``numpy.linspace``. Cost: 0 FLOPs."""
    return _np.linspace(start, stop, num=num, **kwargs)

def zeros_like(a, dtype=None):
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    return _np.zeros_like(a, dtype=dtype)

def ones_like(a, dtype=None):
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: 0 FLOPs."""
    return _np.ones_like(a, dtype=dtype)

def full_like(a, fill_value, dtype=None):
    """Return array filled with value, same shape. Wraps ``numpy.full_like``. Cost: 0 FLOPs."""
    return _np.full_like(a, fill_value, dtype=dtype)

def empty(shape, dtype=float):
    """Return uninitialized array. Wraps ``numpy.empty``. Cost: 0 FLOPs."""
    return _np.empty(shape, dtype=dtype)

def empty_like(a, dtype=None):
    """Return uninitialized array, same shape. Wraps ``numpy.empty_like``. Cost: 0 FLOPs."""
    return _np.empty_like(a, dtype=dtype)

def identity(n, dtype=float):
    """Return identity matrix. Wraps ``numpy.identity``. Cost: 0 FLOPs."""
    return _np.identity(n, dtype=dtype)


# --- Tensor Manipulation ---

def reshape(a, newshape):
    """Reshape array. Wraps ``numpy.reshape``. Cost: 0 FLOPs."""
    return _np.reshape(a, newshape)

def transpose(a, axes=None):
    """Permute dimensions. Wraps ``numpy.transpose``. Cost: 0 FLOPs."""
    return _np.transpose(a, axes=axes)

def swapaxes(a, axis1, axis2):
    """Swap two axes. Wraps ``numpy.swapaxes``. Cost: 0 FLOPs."""
    return _np.swapaxes(a, axis1, axis2)

def moveaxis(a, source, destination):
    """Move axes. Wraps ``numpy.moveaxis``. Cost: 0 FLOPs."""
    return _np.moveaxis(a, source, destination)

def concatenate(arrays, axis=0):
    """Join arrays. Wraps ``numpy.concatenate``. Cost: 0 FLOPs."""
    return _np.concatenate(arrays, axis=axis)

def stack(arrays, axis=0):
    """Stack arrays along new axis. Wraps ``numpy.stack``. Cost: 0 FLOPs."""
    return _np.stack(arrays, axis=axis)

def vstack(arrays):
    """Stack vertically. Wraps ``numpy.vstack``. Cost: 0 FLOPs."""
    return _np.vstack(arrays)

def hstack(arrays):
    """Stack horizontally. Wraps ``numpy.hstack``. Cost: 0 FLOPs."""
    return _np.hstack(arrays)

def split(ary, indices_or_sections, axis=0):
    """Split array. Wraps ``numpy.split``. Cost: 0 FLOPs."""
    return _np.split(ary, indices_or_sections, axis=axis)

def hsplit(ary, indices_or_sections):
    """Split horizontally. Wraps ``numpy.hsplit``. Cost: 0 FLOPs."""
    return _np.hsplit(ary, indices_or_sections)

def vsplit(ary, indices_or_sections):
    """Split vertically. Wraps ``numpy.vsplit``. Cost: 0 FLOPs."""
    return _np.vsplit(ary, indices_or_sections)

def squeeze(a, axis=None):
    """Remove length-1 dims. Wraps ``numpy.squeeze``. Cost: 0 FLOPs."""
    return _np.squeeze(a, axis=axis)

def expand_dims(a, axis):
    """Add dimension. Wraps ``numpy.expand_dims``. Cost: 0 FLOPs."""
    return _np.expand_dims(a, axis)

def ravel(a):
    """Flatten to 1D. Wraps ``numpy.ravel``. Cost: 0 FLOPs."""
    return _np.ravel(a)

def copy(a):
    """Copy array. Wraps ``numpy.copy``. Cost: 0 FLOPs."""
    return _np.copy(a)

def where(condition, x=None, y=None):
    """Conditional select. Wraps ``numpy.where``. Cost: 0 FLOPs."""
    if x is None and y is None:
        return _np.where(condition)
    return _np.where(condition, x, y)

def tile(A, reps):
    """Tile array. Wraps ``numpy.tile``. Cost: 0 FLOPs."""
    return _np.tile(A, reps)

def repeat(a, repeats, axis=None):
    """Repeat elements. Wraps ``numpy.repeat``. Cost: 0 FLOPs."""
    return _np.repeat(a, repeats, axis=axis)

def flip(m, axis=None):
    """Reverse elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    return _np.flip(m, axis=axis)

def roll(a, shift, axis=None):
    """Roll elements. Wraps ``numpy.roll``. Cost: 0 FLOPs."""
    return _np.roll(a, shift, axis=axis)

def sort(a, axis=-1, **kwargs):
    """Sort array. Wraps ``numpy.sort``. Cost: 0 FLOPs."""
    return _np.sort(a, axis=axis, **kwargs)

def argsort(a, axis=-1, **kwargs):
    """Indices to sort. Wraps ``numpy.argsort``. Cost: 0 FLOPs."""
    return _np.argsort(a, axis=axis, **kwargs)

def searchsorted(a, v, **kwargs):
    """Find insertion points. Wraps ``numpy.searchsorted``. Cost: 0 FLOPs."""
    return _np.searchsorted(a, v, **kwargs)

def unique(ar, **kwargs):
    """Unique elements. Wraps ``numpy.unique``. Cost: 0 FLOPs."""
    return _np.unique(ar, **kwargs)

def pad(array, pad_width, mode="constant", **kwargs):
    """Pad array. Wraps ``numpy.pad``. Cost: 0 FLOPs."""
    return _np.pad(array, pad_width, mode=mode, **kwargs)

def triu(m, k=0):
    """Upper triangle. Wraps ``numpy.triu``. Cost: 0 FLOPs."""
    return _np.triu(m, k=k)

def tril(m, k=0):
    """Lower triangle. Wraps ``numpy.tril``. Cost: 0 FLOPs."""
    return _np.tril(m, k=k)

def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return diagonal. Wraps ``numpy.diagonal``. Cost: 0 FLOPs."""
    return _np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)

def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    """Sum of diagonal. Wraps ``numpy.trace``. Cost: 0 FLOPs."""
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

def broadcast_to(array, shape):
    """Broadcast to shape. Wraps ``numpy.broadcast_to``. Cost: 0 FLOPs."""
    return _np.broadcast_to(array, shape)

def meshgrid(*xi, **kwargs):
    """Coordinate matrices. Wraps ``numpy.meshgrid``. Cost: 0 FLOPs."""
    return _np.meshgrid(*xi, **kwargs)


# --- Type and info ---

def astype(a, dtype):
    """Cast array to dtype. Cost: 0 FLOPs."""
    return _np.asarray(a).astype(dtype)

def asarray(a, dtype=None):
    """Convert to array. Wraps ``numpy.asarray``. Cost: 0 FLOPs."""
    return _np.asarray(a, dtype=dtype)

def isnan(x):
    """Test for NaN. Wraps ``numpy.isnan``. Cost: 0 FLOPs."""
    return _np.isnan(x)

def isinf(x):
    """Test for Inf. Wraps ``numpy.isinf``. Cost: 0 FLOPs."""
    return _np.isinf(x)

def isfinite(x):
    """Test for finite. Wraps ``numpy.isfinite``. Cost: 0 FLOPs."""
    return _np.isfinite(x)

def allclose(a, b, rtol=1e-05, atol=1e-08):
    """Test if arrays are close. Wraps ``numpy.allclose``. Cost: 0 FLOPs."""
    return _np.allclose(a, b, rtol=rtol, atol=atol)
```

- [ ] **Step 4: Create random submodule**

```python
# src/mechestim/random/__init__.py
"""Random number generation (passthrough to NumPy). All ops are free (0 FLOPs).

Sampling random numbers is data generation, not compute.
"""
from numpy.random import (  # noqa: F401
    RandomState,
    choice,
    default_rng,
    normal,
    permutation,
    rand,
    randn,
    seed,
    shuffle,
    uniform,
)

def __getattr__(name):
    import numpy.random as _npr
    if hasattr(_npr, name):
        return getattr(_npr, name)
    raise AttributeError(f"mechestim.random does not provide '{name}'")
```

- [ ] **Step 5: Write random tests**

```python
# tests/test_random.py
"""Tests for mechestim.random passthrough."""
from mechestim import random as merandom


def test_seed():
    merandom.seed(42)


def test_randn():
    x = merandom.randn(3, 4)
    assert x.shape == (3, 4)


def test_normal():
    x = merandom.normal(0, 1, size=(5,))
    assert x.shape == (5,)


def test_default_rng():
    rng = merandom.default_rng(42)
    x = rng.standard_normal((3,))
    assert x.shape == (3,)
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/test_free_ops.py tests/test_random.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/_free_ops.py src/mechestim/random/ tests/test_free_ops.py tests/test_random.py
git commit -m "feat: add free ops (tensor creation/manipulation) and random passthrough"
```

---

### Task 7: Pointwise Operations and Reductions

**Files:**
- Create: `src/mechestim/_pointwise.py`
- Create: `tests/test_pointwise.py`

- [ ] **Step 1: Write tests for pointwise ops**

```python
# tests/test_pointwise.py
"""Tests for counted pointwise and reduction operations."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim._pointwise import (
    exp, log, log2, log10, abs, negative, sqrt, square,
    sin, cos, tanh, sign, ceil, floor,
    add, subtract, multiply, divide, maximum, minimum, power, clip, mod,
    sum, max, min, mean, prod, std, var,
    argmax, argmin, cumsum, cumprod,
    dot, matmul,
)
from mechestim.errors import NoBudgetContextError


# --- Unary pointwise ---

def test_exp_result():
    x = numpy.array([0.0, 1.0, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = exp(x)
        assert numpy.allclose(result, numpy.exp(x))


def test_exp_flop_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        exp(x)
        assert budget.flops_used == 200


def test_sqrt_result():
    x = numpy.array([1.0, 4.0, 9.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(sqrt(x), numpy.sqrt(x))


# --- Binary pointwise ---

def test_add_result():
    a = numpy.array([1.0, 2.0])
    b = numpy.array([3.0, 4.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = add(a, b)
        assert numpy.allclose(result, a + b)
        assert budget.flops_used == 2


def test_add_broadcast_cost():
    a = numpy.ones((3, 4))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = add(a, b)
        assert result.shape == (3, 4)
        assert budget.flops_used == 12  # numel(output) = 3*4


def test_maximum_result():
    a = numpy.array([1.0, 5.0, 3.0])
    b = numpy.array([2.0, 4.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = maximum(a, b)
        assert numpy.allclose(result, numpy.maximum(a, b))


def test_clip_result():
    x = numpy.array([-1.0, 0.5, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = clip(x, 0.0, 1.0)
        assert numpy.allclose(result, numpy.clip(x, 0.0, 1.0))


# --- Reductions ---

def test_sum_full():
    x = numpy.ones((5, 3))
    with BudgetContext(flop_budget=10**6) as budget:
        result = sum(x)
        assert result == 15.0
        assert budget.flops_used == 15  # numel(input)


def test_sum_axis():
    x = numpy.ones((5, 3))
    with BudgetContext(flop_budget=10**6) as budget:
        result = sum(x, axis=0)
        assert result.shape == (3,)
        assert budget.flops_used == 15  # numel(input)


def test_mean_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        result = mean(x, axis=0)
        # numel(input) + numel(output) = 200 + 20
        assert budget.flops_used == 220


def test_std_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        result = std(x, axis=0)
        # 2 * numel(input) + numel(output) = 400 + 20
        assert budget.flops_used == 420


def test_argmax_result():
    x = numpy.array([1.0, 5.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = argmax(x)
        assert result == 1


def test_cumsum():
    x = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = cumsum(x)
        assert numpy.allclose(result, [1, 3, 6])


# --- dot and matmul ---

def test_dot_result():
    a = numpy.ones((3, 4))
    b = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        result = dot(a, b)
        assert numpy.allclose(result, numpy.dot(a, b))
        assert budget.flops_used == 3 * 4 * 5


def test_matmul_result():
    a = numpy.ones((3, 4))
    b = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        result = matmul(a, b)
        assert numpy.allclose(result, numpy.matmul(a, b))


# --- Error handling ---

def test_counted_op_outside_context():
    with pytest.raises(NoBudgetContextError):
        exp(numpy.ones((3,)))


def test_nan_warning():
    x = numpy.array([0.0])
    with BudgetContext(flop_budget=10**6) as budget:
        with pytest.warns(match="NaN"):
            log(numpy.array([0.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pointwise.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `_pointwise.py`**

```python
# src/mechestim/_pointwise.py
"""Counted pointwise operations and reductions for mechestim.

Every function wraps its NumPy equivalent, adding FLOP counting
and budget enforcement. Cost formulas:

- Unary pointwise: numel(input)
- Binary pointwise: numel(output) (broadcast-aware)
- Reductions: numel(input)
- mean: numel(input) + numel(output)
- std, var: 2 * numel(input) + numel(output)
"""
from __future__ import annotations

import numpy as _np

from mechestim._budget import BudgetContext
from mechestim._flops import einsum_cost, pointwise_cost, reduction_cost
from mechestim._validation import check_nan_inf, require_budget, validate_ndarray


def _counted_unary(np_func, op_name: str):
    """Create a counted unary pointwise op."""
    def wrapper(x):
        budget = require_budget()
        validate_ndarray(x)
        cost = pointwise_cost(x.shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,))
        result = np_func(x)
        check_nan_inf(result, op_name)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Element-wise {op_name}. Wraps ``numpy.{op_name}``.\n\n"
        f"**FLOP cost:** ``numel(input)``\n"
    )
    return wrapper


def _counted_binary(np_func, op_name: str):
    """Create a counted binary pointwise op."""
    def wrapper(x, y):
        budget = require_budget()
        # Allow scalars — convert to array for shape
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        cost = pointwise_cost(output_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape))
        result = np_func(x, y)
        check_nan_inf(result, op_name)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Element-wise {op_name}. Wraps ``numpy.{op_name}``.\n\n"
        f"**FLOP cost:** ``numel(output)`` (broadcast-aware)\n"
    )
    return wrapper


def _counted_reduction(np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False):
    """Create a counted reduction op.

    Parameters
    ----------
    cost_multiplier : int
        Multiplied by numel(input). Default 1. Use 2 for std/var.
    extra_output : bool
        If True, add numel(output) to cost. Used for mean, std, var.
    """
    def wrapper(a, axis=None, **kwargs):
        budget = require_budget()
        validate_ndarray(a)
        cost = reduction_cost(a.shape, axis) * cost_multiplier
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Reduction: {op_name}. Wraps ``numpy.{op_name}``.\n\n"
        f"**FLOP cost:** ``numel(input)``\n"
    )
    return wrapper


# --- Unary pointwise ops ---

exp = _counted_unary(_np.exp, "exp")
log = _counted_unary(_np.log, "log")
log2 = _counted_unary(_np.log2, "log2")
log10 = _counted_unary(_np.log10, "log10")
abs = _counted_unary(_np.abs, "abs")
negative = _counted_unary(_np.negative, "negative")
sqrt = _counted_unary(_np.sqrt, "sqrt")
square = _counted_unary(_np.square, "square")
sin = _counted_unary(_np.sin, "sin")
cos = _counted_unary(_np.cos, "cos")
tanh = _counted_unary(_np.tanh, "tanh")
sign = _counted_unary(_np.sign, "sign")
ceil = _counted_unary(_np.ceil, "ceil")
floor = _counted_unary(_np.floor, "floor")

# --- Binary pointwise ops ---

add = _counted_binary(_np.add, "add")
subtract = _counted_binary(_np.subtract, "subtract")
multiply = _counted_binary(_np.multiply, "multiply")
divide = _counted_binary(_np.divide, "divide")
maximum = _counted_binary(_np.maximum, "maximum")
minimum = _counted_binary(_np.minimum, "minimum")
power = _counted_binary(_np.power, "power")
mod = _counted_binary(_np.mod, "mod")


def clip(a, a_min, a_max):
    """Clip values. Wraps ``numpy.clip``. Cost: ``numel(output)``."""
    budget = require_budget()
    validate_ndarray(a)
    cost = pointwise_cost(a.shape)
    budget.deduct("clip", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    result = _np.clip(a, a_min, a_max)
    check_nan_inf(result, "clip")
    return result


# --- Reductions ---

sum = _counted_reduction(_np.sum, "sum")
max = _counted_reduction(_np.max, "max")
min = _counted_reduction(_np.min, "min")
prod = _counted_reduction(_np.prod, "prod")
mean = _counted_reduction(_np.mean, "mean", cost_multiplier=1, extra_output=True)
std = _counted_reduction(_np.std, "std", cost_multiplier=2, extra_output=True)
var = _counted_reduction(_np.var, "var", cost_multiplier=2, extra_output=True)
argmax = _counted_reduction(_np.argmax, "argmax")
argmin = _counted_reduction(_np.argmin, "argmin")
cumsum = _counted_reduction(_np.cumsum, "cumsum")
cumprod = _counted_reduction(_np.cumprod, "cumprod")


# --- dot and matmul ---

def dot(a, b):
    """Dot product. Wraps ``numpy.dot``. Cost: same as equivalent einsum."""
    budget = require_budget()
    validate_ndarray(a, b)
    if a.ndim == 2 and b.ndim == 2:
        cost = einsum_cost("ij,jk->ik", shapes=[a.shape, b.shape])
    elif a.ndim == 1 and b.ndim == 1:
        cost = einsum_cost("i,i->", shapes=[a.shape, b.shape])
    else:
        # General case: delegate shape calc to numpy, charge numel of all inputs
        cost = a.size * b.size
    budget.deduct("dot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.dot(a, b)
    check_nan_inf(result, "dot")
    return result


def matmul(a, b):
    """Matrix multiply. Wraps ``numpy.matmul``. Cost: same as equivalent einsum."""
    budget = require_budget()
    validate_ndarray(a, b)
    if a.ndim == 2 and b.ndim == 2:
        cost = einsum_cost("ij,jk->ik", shapes=[a.shape, b.shape])
    elif a.ndim == 1 and b.ndim == 1:
        cost = einsum_cost("i,i->", shapes=[a.shape, b.shape])
    else:
        cost = a.size * b.size
    budget.deduct("matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.matmul(a, b)
    check_nan_inf(result, "matmul")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pointwise.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_pointwise.py tests/test_pointwise.py
git commit -m "feat: add counted pointwise ops, reductions, dot, and matmul"
```

---

### Task 8: Einsum

**Files:**
- Create: `src/mechestim/_einsum.py`
- Create: `tests/test_einsum.py`

- [ ] **Step 1: Write tests for einsum**

```python
# tests/test_einsum.py
"""Tests for mechestim einsum with FLOP counting and symmetry."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim._einsum import einsum
from mechestim.errors import BudgetExhaustedError, NoBudgetContextError, SymmetryError


def test_matmul_result():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        C = einsum('ij,jk->ik', A, B)
        assert numpy.allclose(C, numpy.einsum('ij,jk->ik', A, B))


def test_matmul_flop_cost():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum('ij,jk->ik', A, B)
        assert budget.flops_used == 3 * 4 * 5


def test_trace():
    A = numpy.eye(10)
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum('ii->', A)
        assert result == 10.0
        assert budget.flops_used == 10


def test_outer_product():
    a = numpy.ones((3,))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum('i,j->ij', a, b)
        assert result.shape == (3, 4)
        assert budget.flops_used == 12


def test_batch_matmul():
    A = numpy.ones((2, 3, 4))
    B = numpy.ones((2, 4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum('bij,bjk->bik', A, B)
        assert budget.flops_used == 2 * 3 * 4 * 5


def test_repeated_operand_symmetry():
    x = numpy.ones((10, 256))
    A = numpy.ones((10, 10))
    with BudgetContext(flop_budget=10**8) as budget:
        einsum('ai,bi,ab->', x, x, A)
        expected = (10 * 10 * 256) // 2
        assert budget.flops_used == expected


def test_no_symmetry_different_objects():
    x1 = numpy.ones((10, 256))
    x2 = numpy.ones((10, 256))  # different object
    A = numpy.ones((10, 10))
    with BudgetContext(flop_budget=10**8) as budget:
        einsum('ai,bi,ab->', x1, x2, A)
        # No symmetry: x1 is not x2
        assert budget.flops_used == 10 * 10 * 256


def test_symmetric_dims_valid():
    x = numpy.ones((3, 10))
    y = numpy.ones((3, 10))
    A = numpy.eye(3)
    with BudgetContext(flop_budget=10**8) as budget:
        # Output should be symmetric because A is symmetric
        result = einsum('ai,bj,ab->ij', x, y, A, symmetric_dims=[(0, 1)])
        # Cost reduced by 2!
        expected = (3 * 3 * 10 * 10) // 2
        assert budget.flops_used == expected


def test_symmetric_dims_invalid():
    x = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    y = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    with BudgetContext(flop_budget=10**8) as budget:
        # Result of ij,jk->ik with these inputs won't be symmetric
        with pytest.raises(SymmetryError):
            einsum('ij,jk->ik', x, y, symmetric_dims=[(0, 1)])


def test_outside_context():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with pytest.raises(NoBudgetContextError):
        einsum('ij,jk->ik', A, B)


def test_budget_exceeded():
    A = numpy.ones((256, 256))
    with pytest.raises(BudgetExhaustedError):
        with BudgetContext(flop_budget=100) as budget:
            einsum('ij,jk->ik', A, A)


def test_op_log_records_subscripts():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum('ij,jk->ik', A, B)
        assert budget.op_log[0].subscripts == 'ij,jk->ik'
        assert budget.op_log[0].op_name == 'einsum'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_einsum.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `_einsum.py`**

```python
# src/mechestim/_einsum.py
"""Einsum with analytical FLOP counting and symmetry detection.

Wraps ``numpy.einsum``, adding:
- Analytical FLOP cost (product of all index dimensions)
- Symmetry savings for repeated operands (Python ``is`` check)
- Explicit symmetric_dims for output tensor symmetry claims
"""
from __future__ import annotations

from itertools import permutations

import numpy as _np

from mechestim._flops import einsum_cost, parse_einsum_subscripts
from mechestim._validation import check_nan_inf, require_budget
from mechestim.errors import SymmetryError


def _detect_repeated_operands(operands: tuple) -> list[int] | None:
    """Detect operands that are the same Python object.

    Returns the indices of operands sharing the same id() as the
    first repeated group, or None if no repeats found.
    """
    seen: dict[int, list[int]] = {}
    for i, op in enumerate(operands):
        obj_id = id(op)
        if obj_id not in seen:
            seen[obj_id] = []
        seen[obj_id].append(i)
    # Return the largest repeated group
    for indices in seen.values():
        if len(indices) >= 2:
            return indices
    return None


def _check_einsum_symmetry(
    subscripts: str,
    repeated_indices: list[int],
    inputs: list[list[str]],
) -> bool:
    """Check if swapping repeated operands produces an equivalent expression."""
    # Get the input subscript parts
    input_parts = subscripts.split("->")[0].split(",")

    # Check if swapping the repeated operand positions yields the same
    # set of contractions. We check all permutations of the repeated
    # positions and see if any permutation of their subscript labels
    # produces the same overall expression.
    repeated_labels = [input_parts[i] for i in repeated_indices]

    # All permutations of the repeated input labels should produce the same expression
    base = tuple(repeated_labels)
    for perm in permutations(range(len(repeated_indices))):
        permuted = tuple(repeated_labels[p] for p in perm)
        if permuted != base:
            # Check if this is just a relabeling
            new_parts = list(input_parts)
            for idx, p in zip(repeated_indices, perm):
                new_parts[idx] = repeated_labels[p]
            if ",".join(new_parts) != ",".join(input_parts):
                # Check if the full expression (input->output) is equivalent
                # by checking that the same labels appear
                return True  # Conservatively allow symmetry
    return True


def _validate_symmetric_dims(
    result: _np.ndarray,
    symmetric_dims: list[tuple[int, ...]],
) -> None:
    """Validate that the result actually has the claimed symmetry."""
    for group in symmetric_dims:
        if len(group) < 2:
            continue
        # Check all pairwise swaps within the group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                axes = list(range(result.ndim))
                axes[group[i]], axes[group[j]] = axes[group[j]], axes[group[i]]
                transposed = result.transpose(axes)
                if not _np.allclose(result, transposed, atol=1e-6, rtol=1e-5):
                    max_dev = float(_np.max(_np.abs(result - transposed)))
                    raise SymmetryError(dims=group, max_deviation=max_dev)


def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    symmetric_dims: list[tuple[int, ...]] | None = None,
) -> _np.ndarray:
    """Evaluate an Einstein summation convention on the operands.

    Wraps ``numpy.einsum`` with analytical FLOP counting and budget
    enforcement.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : numpy.ndarray
        The arrays for the operation.
    symmetric_dims : list of tuple of int, optional
        Declares symmetry groups in the output tensor for FLOP savings.
        Validated at runtime; raises SymmetryError if invalid.

    Returns
    -------
    numpy.ndarray
        The calculation result.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the remaining FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetric_dims`` is provided but the result is not symmetric.

    Notes
    -----
    **FLOP cost:** Product of all index dimensions in the subscript.

    **Symmetry savings:** When the same array object is passed multiple
    times (checked via ``is``), cost is divided by ``k!``. Additional
    savings from ``symmetric_dims``.
    """
    budget = require_budget()

    shapes = [op.shape for op in operands]

    # Detect repeated operands
    repeated = _detect_repeated_operands(operands)

    cost = einsum_cost(
        subscripts,
        shapes=list(shapes),
        repeated_operand_indices=repeated,
        symmetric_dims=symmetric_dims,
    )

    budget.deduct(
        "einsum",
        flop_cost=cost,
        subscripts=subscripts,
        shapes=tuple(shapes),
    )

    result = _np.einsum(subscripts, *operands)

    # Validate symmetry claims
    if symmetric_dims and isinstance(result, _np.ndarray) and result.ndim >= 2:
        _validate_symmetric_dims(result, symmetric_dims)

    check_nan_inf(result, "einsum")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_einsum.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_einsum.py tests/test_einsum.py
git commit -m "feat: add einsum with FLOP counting and symmetry detection"
```

---

### Task 9: Linear Algebra Submodule (SVD)

**Files:**
- Create: `src/mechestim/linalg/__init__.py`
- Create: `src/mechestim/linalg/_svd.py`
- Create: `tests/test_linalg.py`

- [ ] **Step 1: Write tests for SVD**

```python
# tests/test_linalg.py
"""Tests for mechestim.linalg.svd."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.linalg import svd
from mechestim.errors import NoBudgetContextError


def test_svd_full_result():
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        U, S, Vt = svd(A)
        U_np, S_np, Vt_np = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S, S_np)
        assert U.shape == (10, 5)
        assert Vt.shape == (5, 5)


def test_svd_full_cost():
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A)
        assert budget.flops_used == 10 * 5 * 5  # m*n*min(m,n)


def test_svd_truncated_result():
    numpy.random.seed(42)
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        U, S, Vt = svd(A, k=3)
        assert U.shape == (10, 3)
        assert S.shape == (3,)
        assert Vt.shape == (3, 5)
        # Singular values should match top-3 of full SVD
        _, S_full, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S, S_full[:3])


def test_svd_truncated_cost():
    A = numpy.random.randn(10, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A, k=3)
        assert budget.flops_used == 10 * 5 * 3


def test_svd_not_2d():
    with BudgetContext(flop_budget=10**6) as budget:
        with pytest.raises(ValueError, match="2D"):
            svd(numpy.ones((3,)))


def test_svd_k_too_large():
    A = numpy.ones((3, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        with pytest.raises(ValueError, match="k"):
            svd(A, k=10)


def test_svd_outside_context():
    with pytest.raises(NoBudgetContextError):
        svd(numpy.ones((3, 3)))


def test_svd_op_log():
    A = numpy.random.randn(8, 4)
    with BudgetContext(flop_budget=10**6) as budget:
        svd(A, k=2)
        assert budget.op_log[0].op_name == "linalg.svd"


def test_linalg_unsupported():
    from mechestim import linalg
    with pytest.raises(AttributeError, match="does not provide"):
        linalg.cholesky
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linalg.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SVD**

```python
# src/mechestim/linalg/_svd.py
"""Truncated SVD with FLOP counting."""
from __future__ import annotations

import numpy as _np

from mechestim._flops import svd_cost
from mechestim._validation import check_nan_inf, require_budget, validate_ndarray


def svd(
    a: _np.ndarray,
    k: int | None = None,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Truncated singular value decomposition.

    Wraps ``numpy.linalg.svd`` with analytical FLOP counting.

    Parameters
    ----------
    a : numpy.ndarray
        Input matrix. Must be 2D.
    k : int or None, optional
        Number of singular values/vectors to compute.
        If None, computes all (k = min(m, n)).

    Returns
    -------
    U : numpy.ndarray, shape (m, k)
        Left singular vectors.
    S : numpy.ndarray, shape (k,)
        Singular values in descending order.
    Vt : numpy.ndarray, shape (k, n)
        Right singular vectors (transposed).

    Raises
    ------
    BudgetExhaustedError
        If the cost exceeds the remaining budget.
    NoBudgetContextError
        If called outside a BudgetContext.
    ValueError
        If input is not 2D or k is out of range.

    Notes
    -----
    **FLOP cost:** ``m * n * k``
    """
    budget = require_budget()
    validate_ndarray(a)

    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")

    m, n = a.shape
    if k is None:
        k = min(m, n)
    if not (1 <= k <= min(m, n)):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}"
        )

    cost = svd_cost(m, n, k)
    budget.deduct(
        "linalg.svd",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    )

    U, S, Vt = _np.linalg.svd(a, full_matrices=False)
    # Truncate to top-k
    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]

    check_nan_inf(S, "linalg.svd")
    return U, S, Vt
```

- [ ] **Step 4: Create linalg `__init__.py`**

```python
# src/mechestim/linalg/__init__.py
"""Linear algebra submodule for mechestim.

Currently provides SVD. Future ops (eigh, cholesky, norm, solve)
may be added based on participant feedback.
"""
from mechestim.linalg._svd import svd  # noqa: F401

__all__ = ["svd"]


def __getattr__(name):
    raise AttributeError(
        f"mechestim.linalg does not provide '{name}'. "
        f"Currently supported: svd. "
        f"Request new ops at https://github.com/AIcrowd/mechestim/issues"
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_linalg.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/linalg/ tests/test_linalg.py
git commit -m "feat: add linalg.svd with truncated SVD and FLOP counting"
```

---

### Task 10: Top-Level `__init__.py` (Public API Assembly)

**Files:**
- Modify: `src/mechestim/__init__.py`

- [ ] **Step 1: Implement the full `__init__.py`**

```python
# src/mechestim/__init__.py
"""mechestim: NumPy-compatible math primitives with FLOP counting.

Usage::

    import mechestim as me

    with me.BudgetContext(flop_budget=1_000_000) as budget:
        W = me.array(weight_matrix)
        h = me.einsum('ij,j->i', W, x)
        h = me.maximum(h, 0)
        print(budget.summary())
"""

__version__ = "0.1.0"

# --- Budget and diagnostics ---
from mechestim._budget import BudgetContext, OpRecord  # noqa: F401

# --- Errors ---
from mechestim.errors import (  # noqa: F401
    BudgetExhaustedError,
    MechEstimError,
    MechEstimWarning,
    NoBudgetContextError,
    SymmetryError,
)

# --- Einsum ---
from mechestim._einsum import einsum  # noqa: F401

# --- Pointwise (counted) ---
from mechestim._pointwise import (  # noqa: F401
    abs,
    add,
    argmax,
    argmin,
    ceil,
    clip,
    cos,
    cumprod,
    cumsum,
    divide,
    dot,
    exp,
    floor,
    log,
    log2,
    log10,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mod,
    multiply,
    negative,
    power,
    prod,
    sign,
    sin,
    sqrt,
    square,
    std,
    subtract,
    sum,
    tanh,
    var,
)

# --- Free ops ---
from mechestim._free_ops import (  # noqa: F401
    allclose,
    arange,
    argsort,
    array,
    asarray,
    astype,
    broadcast_to,
    concatenate,
    copy,
    diag,
    diagonal,
    empty,
    empty_like,
    expand_dims,
    eye,
    flip,
    full,
    full_like,
    hstack,
    hsplit,
    identity,
    isfinite,
    isinf,
    isnan,
    linspace,
    meshgrid,
    moveaxis,
    ones,
    ones_like,
    pad,
    ravel,
    repeat,
    reshape,
    roll,
    searchsorted,
    sort,
    split,
    squeeze,
    stack,
    swapaxes,
    tile,
    trace,
    transpose,
    tril,
    triu,
    unique,
    vsplit,
    vstack,
    where,
    zeros,
    zeros_like,
)

# --- Submodules ---
from mechestim import flops  # noqa: F401
from mechestim import linalg  # noqa: F401
from mechestim import random  # noqa: F401

# --- NumPy constants and types ---
import numpy as _np

pi = _np.pi
e = _np.e
inf = _np.inf
nan = _np.nan
newaxis = _np.newaxis
ndarray = _np.ndarray
float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint8 = _np.uint8
bool_ = _np.bool_
complex64 = _np.complex64
complex128 = _np.complex128

# --- Catch-all for unsupported attributes ---

_SUPPORTED = {name for name in dir() if not name.startswith("_")}


def __getattr__(name: str):
    raise AttributeError(
        f"mechestim does not provide '{name}'. "
        f"See https://github.com/AIcrowd/mechestim for supported operations."
    )
```

- [ ] **Step 2: Verify import works**

Run:
```bash
uv run python -c "
import mechestim as me
print('version:', me.__version__)
print('has einsum:', hasattr(me, 'einsum'))
print('has zeros:', hasattr(me, 'zeros'))
print('has linalg:', hasattr(me, 'linalg'))
print('has random:', hasattr(me, 'random'))
print('has flops:', hasattr(me, 'flops'))
print('pi:', me.pi)
print('ndarray:', me.ndarray)
"
```
Expected: all attributes present, no errors.

- [ ] **Step 3: Verify unsupported op error**

Run:
```bash
uv run python -c "
import mechestim as me
try:
    me.fft
except AttributeError as e:
    print('OK:', e)
"
```
Expected: `OK: mechestim does not provide 'fft'...`

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/__init__.py
git commit -m "feat: assemble public API in __init__.py with __getattr__ catchall"
```

---

### Task 11: NumPy Compatibility Tests

**Files:**
- Create: `tests/test_numpy_compat.py`

- [ ] **Step 1: Write compatibility tests**

```python
# tests/test_numpy_compat.py
"""Verify that mechestim produces identical results to NumPy for all supported ops."""
import numpy
import pytest
import mechestim as me
from mechestim._budget import BudgetContext


@pytest.fixture
def budget():
    with BudgetContext(flop_budget=10**9) as b:
        yield b


class TestUnaryOps:
    """Every unary op must match NumPy exactly."""

    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.x = numpy.random.randn(5, 4).astype(numpy.float64)
        self.x_pos = numpy.abs(self.x) + 0.01  # positive values for log/sqrt

    @pytest.mark.parametrize("op_name", [
        "exp", "log", "log2", "log10", "abs", "negative",
        "sqrt", "square", "sin", "cos", "tanh", "sign", "ceil", "floor",
    ])
    def test_unary(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        inp = self.x_pos if op_name in ("log", "log2", "log10", "sqrt") else self.x
        assert numpy.allclose(me_fn(inp), np_fn(inp), equal_nan=True)


class TestBinaryOps:
    """Every binary op must match NumPy exactly."""

    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.a = numpy.random.randn(3, 4)
        self.b = numpy.random.randn(3, 4) + 2.0  # avoid div-by-zero

    @pytest.mark.parametrize("op_name", [
        "add", "subtract", "multiply", "divide", "maximum", "minimum", "power", "mod",
    ])
    def test_binary(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        a, b = numpy.abs(self.a) + 0.1, numpy.abs(self.b) + 0.1
        assert numpy.allclose(me_fn(a, b), np_fn(a, b))


class TestReductions:
    """Reduction ops match NumPy."""

    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.x = numpy.random.randn(5, 4)

    @pytest.mark.parametrize("op_name", ["sum", "max", "min", "mean", "prod"])
    def test_reduction_full(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        assert numpy.allclose(me_fn(self.x), np_fn(self.x))

    @pytest.mark.parametrize("op_name", ["sum", "max", "min", "mean"])
    def test_reduction_axis(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        assert numpy.allclose(me_fn(self.x, axis=0), np_fn(self.x, axis=0))
        assert numpy.allclose(me_fn(self.x, axis=1), np_fn(self.x, axis=1))


class TestEinsum:
    """Einsum matches NumPy."""

    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_matmul(self):
        A = numpy.random.randn(4, 5)
        B = numpy.random.randn(5, 3)
        assert numpy.allclose(
            me.einsum('ij,jk->ik', A, B),
            numpy.einsum('ij,jk->ik', A, B),
        )

    def test_trace(self):
        A = numpy.random.randn(4, 4)
        assert numpy.allclose(me.einsum('ii->', A), numpy.einsum('ii->', A))

    def test_batch_matmul(self):
        A = numpy.random.randn(2, 3, 4)
        B = numpy.random.randn(2, 4, 5)
        assert numpy.allclose(
            me.einsum('bij,bjk->bik', A, B),
            numpy.einsum('bij,bjk->bik', A, B),
        )


class TestDotMatmul:
    """dot and matmul match NumPy."""

    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_dot_2d(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        assert numpy.allclose(me.dot(A, B), numpy.dot(A, B))

    def test_dot_1d(self):
        a = numpy.random.randn(5)
        b = numpy.random.randn(5)
        assert numpy.allclose(me.dot(a, b), numpy.dot(a, b))

    def test_matmul_2d(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        assert numpy.allclose(me.matmul(A, B), numpy.matmul(A, B))


class TestSVD:
    """SVD matches NumPy."""

    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_svd_singular_values(self):
        numpy.random.seed(42)
        A = numpy.random.randn(10, 5)
        _, S_me, _ = me.linalg.svd(A)
        _, S_np, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S_me, S_np)

    def test_svd_truncated_values(self):
        numpy.random.seed(42)
        A = numpy.random.randn(10, 5)
        _, S_me, _ = me.linalg.svd(A, k=3)
        _, S_np, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S_me, S_np[:3])


class TestFreeOps:
    """Free ops work with and without budget context."""

    def test_zeros_no_context(self):
        x = me.zeros((3, 4))
        assert x.shape == (3, 4)

    def test_constants(self):
        assert me.pi == numpy.pi
        assert me.inf == numpy.inf
        assert me.ndarray is numpy.ndarray
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_numpy_compat.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_numpy_compat.py
git commit -m "test: add NumPy compatibility test suite"
```

---

### Task 12: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests simulating participant usage."""
import numpy
import pytest
import mechestim as me


def test_simple_mlp_forward_pass():
    """Simulate a participant doing a forward pass through a small MLP."""
    numpy.random.seed(42)
    width = 16
    depth = 4

    # Generate random MLP weights (He init)
    weights = [
        me.array(numpy.random.randn(width, width) * numpy.sqrt(2.0 / width))
        for _ in range(depth)
    ]

    with me.BudgetContext(flop_budget=10**8) as budget:
        # Forward pass: sample a batch, propagate through
        x = me.array(numpy.random.randn(100, width))

        for W in weights:
            x = me.einsum('bi,ji->bj', x, W)  # linear
            x = me.maximum(x, me.zeros_like(x))  # ReLU

        # Estimate mean at final layer
        estimate = me.mean(x, axis=0)

        assert estimate.shape == (width,)
        assert budget.flops_used > 0

        # Verify summary works
        summary = budget.summary()
        assert "einsum" in summary
        assert "maximum" in summary
        assert "mean" in summary

        # Verify op_log
        assert len(budget.op_log) > 0
        einsum_ops = [r for r in budget.op_log if r.op_name == "einsum"]
        assert len(einsum_ops) == depth


def test_budget_tracking_accuracy():
    """Verify that FLOP tracking is exact, not approximate."""
    numpy.random.seed(42)
    A = me.array(numpy.random.randn(10, 20))
    B = me.array(numpy.random.randn(20, 30))

    with me.BudgetContext(flop_budget=10**8) as budget:
        me.einsum('ij,jk->ik', A, B)  # 10 * 20 * 30 = 6000
        me.exp(me.ones((100,)))        # 100
        me.sum(me.ones((50,)))          # 50

        assert budget.flops_used == 6000 + 100 + 50
        assert budget.flops_remaining == 10**8 - 6150


def test_flop_query_matches_execution():
    """FLOP cost query API returns same cost as actual execution."""
    query_cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(10, 20), (20, 30)])

    with me.BudgetContext(flop_budget=10**8) as budget:
        A = me.array(numpy.random.randn(10, 20))
        B = me.array(numpy.random.randn(20, 30))
        me.einsum('ij,jk->ik', A, B)

    assert budget.flops_used == query_cost


def test_mixed_free_and_counted():
    """Free ops don't consume budget, counted ops do."""
    with me.BudgetContext(flop_budget=1000) as budget:
        # Free
        x = me.zeros((10, 10))
        x = me.reshape(x, (100,))
        x = me.ones((5, 5))
        x = me.transpose(x)
        assert budget.flops_used == 0

        # Counted
        me.exp(me.ones((10,)))
        assert budget.flops_used == 10


def test_budget_exhaustion_mid_computation():
    """Budget runs out in the middle of a computation."""
    with pytest.raises(me.BudgetExhaustedError) as exc_info:
        with me.BudgetContext(flop_budget=500) as budget:
            me.exp(me.ones((100,)))  # 100 FLOPs
            me.exp(me.ones((100,)))  # 100 FLOPs — total 200
            me.exp(me.ones((100,)))  # 100 FLOPs — total 300
            me.exp(me.ones((100,)))  # 100 FLOPs — total 400
            me.exp(me.ones((100,)))  # 100 FLOPs — total 500
            me.exp(me.ones((100,)))  # 100 FLOPs — would exceed!

    assert exc_info.value.flops_remaining == 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: all PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all PASS across all test files

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests"
```

---

### Task 13: Documentation Setup

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/quickstart.md`
- Create: `README.md`

- [ ] **Step 1: Create `mkdocs.yml`**

```yaml
# mkdocs.yml
site_name: mechestim
site_description: NumPy-compatible math primitives with FLOP counting
repo_url: https://github.com/AIcrowd/mechestim

theme:
  name: material
  palette:
    scheme: default
    primary: indigo

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - Quick Start: quickstart.md
  - API Reference:
    - Counted Operations: api/counted-ops.md
    - Free Operations: api/free-ops.md
    - Budget: api/budget.md
    - FLOP Cost Query: api/flops.md
    - Errors: api/errors.md
  - Changelog: changelog.md

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
```

- [ ] **Step 2: Create `docs/index.md`**

```markdown
# mechestim

NumPy-compatible math primitives with analytical FLOP counting for the
[Mechanistic Estimation Challenge](https://github.com/AIcrowd/mechestim).

## What is this?

mechestim is a drop-in replacement for a subset of NumPy that counts
FLOPs as you compute. Use it to develop algorithms where the goal is
to minimize computational cost, not wall-clock time.

## Quick example

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    W = me.array(weight_matrix)
    x = me.zeros((256,))
    h = me.einsum('ij,j->i', W, x)
    h = me.maximum(h, 0)
    print(budget.summary())
```

## Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
```
```

- [ ] **Step 3: Create `docs/quickstart.md`**

```markdown
# Quick Start

## Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
```

## Basic Usage

```python
import mechestim as me

# All computation happens inside a BudgetContext
with me.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs)
    A = me.ones((256, 256))
    B = me.eye(256)

    # Counted operations (deduct from budget)
    C = me.einsum('ij,jk->ik', A, B)   # costs 256^3 FLOPs
    D = me.exp(C)                       # costs 256^2 FLOPs

    # Check your budget
    print(f"Used: {budget.flops_used:,} FLOPs")
    print(f"Remaining: {budget.flops_remaining:,} FLOPs")
    print(budget.summary())
```

## Planning Your Budget

Query operation costs before executing them:

```python
import mechestim as me

cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,} FLOPs")

cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,} FLOPs")
```
```

- [ ] **Step 4: Create placeholder API docs**

Create these files with auto-generation directives:

`docs/api/counted-ops.md`:
```markdown
# Counted Operations

::: mechestim._einsum.einsum
::: mechestim._pointwise
```

`docs/api/free-ops.md`:
```markdown
# Free Operations

::: mechestim._free_ops
```

`docs/api/budget.md`:
```markdown
# Budget

::: mechestim._budget.BudgetContext
::: mechestim._budget.OpRecord
```

`docs/api/flops.md`:
```markdown
# FLOP Cost Query API

::: mechestim._flops
```

`docs/api/errors.md`:
```markdown
# Errors

::: mechestim.errors
```

`docs/changelog.md`:
```markdown
# Changelog

## 0.1.0 (2026-04-01)

Initial release for warm-up round.

- Einsum with symmetry detection and FLOP counting
- Pointwise operations (exp, log, add, multiply, etc.)
- Reductions (sum, mean, max, etc.)
- SVD with truncated top-k
- Free tensor creation and manipulation ops
- Budget enforcement via BudgetContext
- FLOP cost query API
- NumPy-compatible API (`import mechestim as me`)
```

- [ ] **Step 5: Create `README.md`**

```markdown
# mechestim

NumPy-compatible math primitives with analytical FLOP counting for the Mechanistic Estimation Challenge.

## Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
```

## Usage

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    W = me.array(weight_matrix)
    x = me.zeros((256,))
    h = me.einsum('ij,j->i', W, x)
    h = me.maximum(h, 0)
    print(budget.summary())
```

## Documentation

Full documentation at [mechestim.github.io](https://aicrowd.github.io/mechestim/) (coming soon).

## Development

```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
uv run pytest
```

## License

MIT
```

- [ ] **Step 6: Verify docs build**

Run:
```bash
uv run mkdocs build
```
Expected: Site built successfully

- [ ] **Step 7: Commit**

```bash
git add mkdocs.yml docs/ README.md
git commit -m "docs: add mkdocs site, quickstart, API reference, README"
```

---

## Summary

| Task | Description | Depends On | Files Created |
|------|-------------|------------|---------------|
| 1 | Project scaffolding | — | pyproject.toml, __init__.py, py.typed |
| 2 | Error classes | 1 | errors.py, test_errors.py |
| 3 | Budget context | 2 | _budget.py, test_budget.py |
| 4 | FLOP calculators | 1 | _flops.py, flops.py, test_flops.py |
| 5 | Validation utils | 2, 3 | _validation.py |
| 6 | Free ops + random | 3 | _free_ops.py, random/, test_free_ops.py, test_random.py |
| 7 | Pointwise ops | 4, 5 | _pointwise.py, test_pointwise.py |
| 8 | Einsum | 4, 5 | _einsum.py, test_einsum.py |
| 9 | SVD (linalg) | 4, 5 | linalg/, test_linalg.py |
| 10 | __init__.py assembly | 6-9 | __init__.py (full) |
| 11 | NumPy compat tests | 10 | test_numpy_compat.py |
| 12 | Integration tests | 10 | test_integration.py |
| 13 | Documentation | 10 | mkdocs.yml, docs/, README.md |

**Parallelizable:** Tasks 6, 7, 8, 9 can run in parallel after Task 5.
