"""NumPy version checking for mechestim."""
from __future__ import annotations

import warnings

import numpy as _np

from mechestim.errors import MechEstimWarning


def check_numpy_version(pinned: str) -> None:
    """Warn if the installed numpy version doesn't match the pinned version."""
    installed = _np.__version__
    pinned_parts = pinned.split(".")[:2]
    installed_parts = installed.split(".")[:2]
    if pinned_parts != installed_parts:
        warnings.warn(
            f"mechestim registry was built for numpy {pinned} but numpy "
            f"{installed} is installed. Some functions may be missing or "
            f"behave differently.",
            MechEstimWarning,
            stacklevel=3,
        )
