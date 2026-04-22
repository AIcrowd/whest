"""Exception and warning classes for whest."""

from __future__ import annotations

import os

_DEFAULT_DOCS_ROOT = "https://aicrowd.github.io/whest/docs"
_BUDGET_DOCS_PATH = "/guides/budget-planning"
_COMPETITION_DOCS_PATH = "/getting-started/competition"
_SYMMETRY_DOCS_PATH = "/guides/symmetry"


def _docs_url(path: str) -> str:
    root = os.environ.get("WHEST_DOCS_ROOT", "").strip() or _DEFAULT_DOCS_ROOT
    return f"{root.rstrip('/')}{path}"


class WhestError(Exception):
    """Base exception for all whest errors."""


class BudgetExhaustedError(WhestError):
    """Raised when an operation would exceed the FLOP budget."""

    def __init__(self, op_name: str, *, flop_cost: int, flops_remaining: int):
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.flops_remaining = flops_remaining
        super().__init__(
            f"{op_name} would cost {flop_cost:,} FLOPs but only "
            f"{flops_remaining:,} remain. "
            f"See: {_docs_url(_BUDGET_DOCS_PATH)}"
        )


class TimeExhaustedError(WhestError):
    """Raised when an operation exceeds the wall-clock time limit."""

    def __init__(self, op_name: str, *, elapsed_s: float, limit_s: float):
        self.op_name = op_name
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s
        super().__init__(
            f"{op_name}: wall-clock time {elapsed_s:.3f}s exceeds "
            f"limit {limit_s:.3f}s. "
            f"See: {_docs_url(_BUDGET_DOCS_PATH)}"
        )


class NoBudgetContextError(WhestError):
    """Raised when a counted operation is called outside a BudgetContext."""

    def __init__(self):
        super().__init__(
            "No active BudgetContext. "
            "Wrap your code in `with whest.BudgetContext(...):`  "
            f"See: {_docs_url(_COMPETITION_DOCS_PATH)}"
        )


class SymmetryError(WhestError):
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
            f"See: {_docs_url(_SYMMETRY_DOCS_PATH)}"
        )


class UnsupportedFunctionError(WhestError):
    """Raised when calling a function not available in the installed NumPy.

    Use ``min_version`` for functions that require a newer numpy than installed
    (e.g. ``bitwise_count`` requires numpy >= 2.1). Use ``max_version`` for
    functions that have been removed in a newer numpy than installed
    (e.g. ``in1d`` was removed in numpy 2.4). ``replacement`` optionally names
    the function users should call instead.
    """

    def __init__(
        self,
        func_name: str,
        *,
        min_version: str | None = None,
        max_version: str | None = None,
        replacement: str | None = None,
    ):
        import numpy as _np

        self.func_name = func_name
        self.min_version = min_version
        self.max_version = max_version
        self.replacement = replacement

        if max_version is not None:
            if replacement is not None:
                msg = (
                    f"numpy.{func_name} was removed in numpy {max_version}; "
                    f"use `{replacement}` instead "
                    f"(you have numpy {_np.__version__})."
                )
            else:
                msg = (
                    f"numpy.{func_name} was removed in numpy {max_version} "
                    f"(you have numpy {_np.__version__})."
                )
        elif min_version is not None:
            msg = (
                f"numpy.{func_name} requires numpy >= {min_version} "
                f"(you have numpy {_np.__version__}). "
                f"To use it: uv pip install 'numpy>={min_version}'"
            )
        else:
            msg = f"numpy.{func_name} is not supported by whest."

        super().__init__(msg)


class WhestWarning(UserWarning):
    """Warning issued when whest detects potential numerical issues."""


class SymmetryLossWarning(WhestWarning):
    """Warning issued when an operation causes loss of symmetry metadata."""
