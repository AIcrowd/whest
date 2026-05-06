# src/flopscope/fft/_free.py
"""Zero-FLOP FFT utility operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as _np
from numpy.typing import ArrayLike

from flopscope._docstrings import attach_docstring
from flopscope._ndarray import FlopscopeArray, _to_base_ndarray


def fftfreq(n: int, d: float = 1.0, device: Any = None) -> FlopscopeArray:
    """FFT sample frequencies. Cost: 0 FLOPs."""
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    return _np.fft.fftfreq(n, d=d, **kwargs)  # type: ignore[reportReturnType]


attach_docstring(fftfreq, _np.fft.fftfreq, "free", "0 FLOPs")


def rfftfreq(n: int, d: float = 1.0, device: Any = None) -> FlopscopeArray:
    """Real FFT sample frequencies. Cost: 0 FLOPs."""
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    return _np.fft.rfftfreq(n, d=d, **kwargs)  # type: ignore[reportReturnType]


attach_docstring(rfftfreq, _np.fft.rfftfreq, "free", "0 FLOPs")


def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> FlopscopeArray:
    """Shift zero-frequency component to center. Cost: 0 FLOPs."""
    return _np.fft.fftshift(_to_base_ndarray(x), axes=axes)  # type: ignore[reportReturnType]


attach_docstring(fftshift, _np.fft.fftshift, "free", "0 FLOPs")


def ifftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> FlopscopeArray:
    """Inverse of fftshift. Cost: 0 FLOPs."""
    return _np.fft.ifftshift(_to_base_ndarray(x), axes=axes)  # type: ignore[reportReturnType]


attach_docstring(ifftshift, _np.fft.ifftshift, "free", "0 FLOPs")
