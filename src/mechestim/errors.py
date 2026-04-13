"""Exception and warning classes for mechestim."""

from __future__ import annotations

_DOCS_BASE = "https://aicrowd.github.io/mechestim/troubleshooting/common-errors"


class MechEstimError(Exception):
    """Base exception for all mechestim errors."""


class BudgetExhaustedError(MechEstimError):
    """Raised when an operation would exceed the FLOP budget."""

    def __init__(self, op_name: str, *, flop_cost: int, flops_remaining: int):
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.flops_remaining = flops_remaining
        super().__init__(
            f"{op_name} would cost {flop_cost:,} FLOPs but only "
            f"{flops_remaining:,} remain. "
            f"See: {_DOCS_BASE}/#budgetexhaustederror"
        )


class NoBudgetContextError(MechEstimError):
    """Raised when a counted operation is called outside a BudgetContext."""

    def __init__(self):
        super().__init__(
            "No active BudgetContext. "
            "Wrap your code in `with mechestim.BudgetContext(...):`  "
            f"See: {_DOCS_BASE}/#nobudgetcontexterror"
        )


class SymmetryError(MechEstimError):
    """Raised when a claimed tensor symmetry does not hold."""

    def __init__(
        self,
        axes: tuple[int, ...],
        max_deviation: float,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ):
        self.axes = axes
        self.max_deviation = max_deviation
        self.atol = atol
        self.rtol = rtol
        super().__init__(
            f"Tensor not symmetric along axes ({', '.join(str(d) for d in axes)}): "
            f"max deviation = {max_deviation} "
            f"(tolerance: atol={atol}, rtol={rtol}). "
            f"See: {_DOCS_BASE}/#symmetryerror"
        )


class UnsupportedFunctionError(MechEstimError):
    """Raised when calling a function not available in the installed NumPy."""

    def __init__(self, func_name: str, *, min_version: str):
        import numpy as _np

        self.func_name = func_name
        self.min_version = min_version
        super().__init__(
            f"numpy.{func_name} requires numpy >= {min_version} "
            f"(you have numpy {_np.__version__}). "
            f"To use it: uv pip install 'numpy>={min_version}'"
        )


class MechEstimWarning(UserWarning):
    """Warning issued when mechestim detects potential numerical issues."""


class SymmetryLossWarning(MechEstimWarning):
    """Warning issued when an operation causes loss of symmetry metadata."""
