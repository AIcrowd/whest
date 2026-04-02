# src/mechestim/fft/_free.py
"""Zero-FLOP FFT utility operations."""
from __future__ import annotations
import numpy as _np

def fftfreq(n, d=1.0):
    """FFT sample frequencies. Cost: 0 FLOPs."""
    return _np.fft.fftfreq(n, d=d)

def rfftfreq(n, d=1.0):
    """Real FFT sample frequencies. Cost: 0 FLOPs."""
    return _np.fft.rfftfreq(n, d=d)

def fftshift(x, axes=None):
    """Shift zero-frequency component to center. Cost: 0 FLOPs."""
    return _np.fft.fftshift(x, axes=axes)

def ifftshift(x, axes=None):
    """Inverse of fftshift. Cost: 0 FLOPs."""
    return _np.fft.ifftshift(x, axes=axes)
