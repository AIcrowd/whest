"""NumPy version checking for mechestim."""

from __future__ import annotations

import warnings

import numpy as _np

from mechestim.errors import MechEstimWarning


def check_numpy_version(supported_range: str) -> None:
    """Warn if the installed numpy version is outside the supported range.

    Parameters
    ----------
    supported_range : str
        A PEP 440 version specifier like ``">=2.0.0,<2.3.0"``.
    """
    installed = _np.__version__
    installed_parts = tuple(int(x) for x in installed.split(".")[:2])

    # Parse the supported range: ">=2.0.0,<2.3.0"
    min_version = None
    max_version = None
    for spec in supported_range.split(","):
        spec = spec.strip()
        if spec.startswith(">="):
            min_version = tuple(int(x) for x in spec[2:].split(".")[:2])
        elif spec.startswith("<"):
            max_version = tuple(int(x) for x in spec[1:].split(".")[:2])

    in_range = True
    if min_version and installed_parts < min_version:
        in_range = False
    if max_version and installed_parts >= max_version:
        in_range = False

    if not in_range:
        warnings.warn(
            f"mechestim supports numpy {supported_range} but numpy "
            f"{installed} is installed. Some functions may be missing or "
            f"behave differently.",
            MechEstimWarning,
            stacklevel=3,
        )
