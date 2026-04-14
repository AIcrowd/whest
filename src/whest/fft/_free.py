# src/whest/fft/_free.py
"""Zero-FLOP FFT utility operations."""

from __future__ import annotations

import numpy as _np

from whest._docstrings import attach_docstring


def fftfreq(n, d=1.0, device=None):
    """FFT sample frequencies. Cost: 0 FLOPs."""
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    return _np.fft.fftfreq(n, d=d, **kwargs)


attach_docstring(fftfreq, _np.fft.fftfreq, "free", "0 FLOPs")


def rfftfreq(n, d=1.0, device=None):
    """Real FFT sample frequencies. Cost: 0 FLOPs."""
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    return _np.fft.rfftfreq(n, d=d, **kwargs)


attach_docstring(rfftfreq, _np.fft.rfftfreq, "free", "0 FLOPs")


def fftshift(x, axes=None):
    """Shift zero-frequency component to center. Cost: 0 FLOPs."""
    return _np.fft.fftshift(x, axes=axes)


attach_docstring(fftshift, _np.fft.fftshift, "free", "0 FLOPs")


def ifftshift(x, axes=None):
    """Inverse of fftshift. Cost: 0 FLOPs."""
    return _np.fft.ifftshift(x, axes=axes)


attach_docstring(ifftshift, _np.fft.ifftshift, "free", "0 FLOPs")
