"""Custom exceptions and warnings for the mechestim client."""
from __future__ import annotations


class MechEstimError(Exception):
    """Base exception for all mechestim errors."""


class BudgetExhaustedError(MechEstimError):
    """Raised when the FLOP budget has been exhausted."""

    def __init__(
        self,
        op_name: str = "",
        flop_cost: int = 0,
        flops_remaining: int = 0,
        message: str = "",
    ) -> None:
        if message:
            text = message
        else:
            text = (
                f"Budget exhausted: operation '{op_name}' requires {flop_cost} FLOPs "
                f"but only {flops_remaining} FLOPs remain."
            )
        super().__init__(text)
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.flops_remaining = flops_remaining


class NoBudgetContextError(MechEstimError):
    """Raised when a tracked operation is called outside a BudgetContext."""

    _DEFAULT = (
        "No active budget context. Wrap your computation in a BudgetContext."
    )

    def __init__(self, message: str = "") -> None:
        super().__init__(message if message else self._DEFAULT)


class SymmetryError(MechEstimError):
    """Raised when a matrix that must be symmetric is not."""

    def __init__(
        self,
        dims: tuple = (),
        max_deviation: float = 0.0,
        message: str = "",
    ) -> None:
        if message:
            text = message
        else:
            text = (
                f"Matrix with dims {dims} is not symmetric "
                f"(max deviation: {max_deviation})."
            )
        super().__init__(text)
        self.dims = dims
        self.max_deviation = max_deviation


class MechEstimWarning(UserWarning):
    """Warning for NaN/Inf values or other numeric anomalies."""


class MechEstimServerError(MechEstimError):
    """Server-side error that does not map to a more specific exception."""


# ---------------------------------------------------------------------------
# Error map and dispatcher
# ---------------------------------------------------------------------------

_MECHESTIM_ERRORS = frozenset({
    "BudgetExhaustedError",
    "NoBudgetContextError",
    "SymmetryError",
})

_ERROR_MAP: dict[str, type[Exception]] = {
    "BudgetExhaustedError": BudgetExhaustedError,
    "NoBudgetContextError": NoBudgetContextError,
    "SymmetryError": SymmetryError,
    "MechEstimServerError": MechEstimServerError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "InvalidRequestError": MechEstimServerError,
}


def raise_from_response(error_type: str, message: str) -> None:
    """Map *error_type* string to an exception class and raise it.

    Unknown error types are treated as :class:`MechEstimServerError`.
    """
    # Ensure message is a str (server may send bytes if it contains non-ASCII)
    if isinstance(message, bytes):
        message = message.decode("utf-8", errors="replace")
    if isinstance(error_type, bytes):
        error_type = error_type.decode("utf-8", errors="replace")

    exc_cls = _ERROR_MAP.get(error_type, MechEstimServerError)
    if error_type in _MECHESTIM_ERRORS:
        raise exc_cls(message=message)
    raise exc_cls(message)
