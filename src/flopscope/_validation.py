"""Input validation and NaN/Inf checking for flopscope operations."""

from __future__ import annotations

import warnings

import numpy as np

from flopscope._budget import get_active_budget
from flopscope.errors import FlopscopeWarning


def require_budget():
    """Return the active budget, auto-activating the global default if needed."""
    budget = get_active_budget()
    if budget is not None:
        return budget
    from flopscope._budget import _get_global_default

    return _get_global_default()


def validate_ndarray(*arrays: object) -> None:
    """Validate that all arguments are numpy ndarrays."""
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}")


def coerce_array(x):
    """Convert input to ndarray if not already one, matching NumPy's behavior."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def coerce_arrays(*arrays):
    """Convert multiple inputs to ndarrays."""
    return tuple(coerce_array(a) for a in arrays)


def check_nan_inf(result: np.ndarray, op_name: str) -> None:
    """Issue a warning if result contains NaN or Inf values.

    Skips dtypes that don't support `np.isnan`/`np.isinf` (e.g. object,
    integer, complex with object content) — these can never contain NaN
    or Inf as ndarray values, so the check is a no-op.
    """
    if not isinstance(result, np.ndarray):
        return
    # np.isnan/np.isinf only support float and complex dtypes. For other
    # dtypes (object, integer, bool, structured), there are no NaN/Inf
    # values to detect, so skip the check.
    if result.dtype.kind not in ("f", "c"):
        return
    # Strip flopscope subclasses so np.isnan / np.isinf do not re-dispatch
    # through __array_ufunc__ and recurse into me.isnan / me.isfinite,
    # which in turn would call check_nan_inf again.
    plain = result.view(np.ndarray) if type(result) is not np.ndarray else result
    nan_count = int(np.isnan(plain).sum())
    inf_count = int(np.isinf(plain).sum())
    if nan_count > 0 or inf_count > 0:
        warnings.warn(
            f"{op_name} produced {nan_count} NaN and {inf_count} Inf values "
            f"in output of shape {result.shape}",
            FlopscopeWarning,
            stacklevel=3,
        )


def maybe_check_nan_inf(result: object, op_name: str) -> None:
    """Run :func:`check_nan_inf` only if the global ``check_nan_inf`` setting is on.

    Production scoring runs with the setting off (default) and pays no
    per-op O(n) scan cost.  Debug callers opt in via
    ``flopscope.configure(check_nan_inf=True)``.
    """
    from flopscope._config import get_setting

    if not get_setting("check_nan_inf"):
        return
    if isinstance(result, np.ndarray):
        check_nan_inf(result, op_name)
