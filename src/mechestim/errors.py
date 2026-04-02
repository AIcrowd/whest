"""Exception and warning classes for mechestim."""
from __future__ import annotations


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
    """Raised when a claimed tensor symmetry does not hold."""

    def __init__(self, dims: tuple[int, ...], max_deviation: float,
                 atol: float = 1e-6, rtol: float = 1e-5):
        self.dims = dims
        self.max_deviation = max_deviation
        self.atol = atol
        self.rtol = rtol
        super().__init__(
            f"Tensor not symmetric along dims ({', '.join(str(d) for d in dims)}): "
            f"max deviation = {max_deviation} "
            f"(tolerance: atol={atol}, rtol={rtol})"
        )


class MechEstimWarning(UserWarning):
    """Warning issued when mechestim detects potential numerical issues."""
