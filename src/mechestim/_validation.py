"""Input validation and NaN/Inf checking for mechestim operations."""

from __future__ import annotations

import warnings

import numpy as np

from mechestim._budget import get_active_budget
from mechestim.errors import MechEstimWarning, NoBudgetContextError


def require_budget():
    """Return the active budget or raise NoBudgetContextError."""
    budget = get_active_budget()
    if budget is None:
        raise NoBudgetContextError()
    return budget


def validate_ndarray(*arrays: object) -> None:
    """Validate that all arguments are numpy ndarrays."""
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}")


def check_nan_inf(result: np.ndarray, op_name: str) -> None:
    """Issue a warning if result contains NaN or Inf values."""
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
