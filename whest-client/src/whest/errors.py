"""Custom exceptions and warnings for the whest client."""

from __future__ import annotations


class WhestError(Exception):
    """Base exception for all whest errors."""


class BudgetExhaustedError(WhestError):
    """Raised when the FLOP budget has been exhausted."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class NoBudgetContextError(WhestError):
    """Raised when a tracked operation is called outside a BudgetContext."""

    _DEFAULT = "No active budget context. Wrap your computation in a BudgetContext."

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class SymmetryError(WhestError):
    """Raised when a matrix that must be symmetric is not."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class UnsupportedFunctionError(WhestError):
    """Raised when calling a function not available in the installed NumPy."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class WhestWarning(UserWarning):
    """Warning for NaN/Inf values or other numeric anomalies."""


class SymmetryLossWarning(WhestWarning):
    """Warning issued when an operation causes loss of symmetry metadata."""


class WhestServerError(WhestError):
    """Server-side error that does not map to a more specific exception."""


# ---------------------------------------------------------------------------
# Error map and dispatcher
# ---------------------------------------------------------------------------

_WHEST_ERRORS = frozenset(
    {
        "BudgetExhaustedError",
        "NoBudgetContextError",
        "SymmetryError",
        "UnsupportedFunctionError",
    }
)

_ERROR_MAP: dict[str, type[Exception]] = {
    "BudgetExhaustedError": BudgetExhaustedError,
    "NoBudgetContextError": NoBudgetContextError,
    "SymmetryError": SymmetryError,
    "UnsupportedFunctionError": UnsupportedFunctionError,
    "WhestServerError": WhestServerError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "InvalidRequestError": WhestServerError,
}


def raise_from_response(error_type: str, message: str) -> None:
    """Map *error_type* string to an exception class and raise it.

    Unknown error types are treated as :class:`WhestServerError`.
    """
    # Ensure message is a str (server may send bytes if it contains non-ASCII)
    if isinstance(message, bytes):
        message = message.decode("utf-8", errors="replace")
    if isinstance(error_type, bytes):
        error_type = error_type.decode("utf-8", errors="replace")

    exc_cls = _ERROR_MAP.get(error_type, WhestServerError)
    if error_type in _WHEST_ERRORS:
        raise exc_cls(message=message)
    raise exc_cls(message)
