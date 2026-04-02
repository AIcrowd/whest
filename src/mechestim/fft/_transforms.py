# src/mechestim/fft/_transforms.py
"""FFT transform wrappers with FLOP counting.

Cost model: 5 * N * log2(N) for complex DFT of length N (Cooley-Tukey radix-2).
Source: Cooley & Tukey (1965); Van Loan, "Computational Frameworks for the FFT" (1992), §1.4.
"""
from __future__ import annotations
import math
import numpy as _np
from mechestim._validation import require_budget, validate_ndarray

def fft_cost(n: int) -> int:
    """FLOP cost of 1-D complex FFT. Formula: 5 * n * ceil(log2(n))."""
    if n <= 1: return 0
    return 5 * n * math.ceil(math.log2(n))

def rfft_cost(n: int) -> int:
    """FLOP cost of 1-D real FFT. Formula: 5 * (n//2) * ceil(log2(n))."""
    if n <= 1: return 0
    return 5 * (n // 2) * math.ceil(math.log2(n))

def fftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of N-D complex FFT. Formula: 5 * N * ceil(log2(N)), N=prod(shape)."""
    N = 1
    for d in shape: N *= d
    if N <= 1: return 0
    return 5 * N * math.ceil(math.log2(N))

def rfftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of N-D real FFT. Formula: 5 * (N//2) * ceil(log2(N))."""
    N = 1
    for d in shape: N *= d
    if N <= 1: return 0
    return 5 * (N // 2) * math.ceil(math.log2(N))

def hfft_cost(n_out: int) -> int:
    """FLOP cost of Hermitian FFT. Formula: 5 * n_out * ceil(log2(n_out))."""
    if n_out <= 1: return 0
    return 5 * n_out * math.ceil(math.log2(n_out))

# 1-D transforms
def fft(a, n=None, axis=-1, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if n is None: n = a.shape[axis]
    cost = fft_cost(n)
    budget.deduct("fft.fft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.fft(a, n=n, axis=axis, norm=norm)

def ifft(a, n=None, axis=-1, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if n is None: n = a.shape[axis]
    cost = fft_cost(n)
    budget.deduct("fft.ifft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ifft(a, n=n, axis=axis, norm=norm)

def rfft(a, n=None, axis=-1, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if n is None: n = a.shape[axis]
    cost = rfft_cost(n)
    budget.deduct("fft.rfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.rfft(a, n=n, axis=axis, norm=norm)

def irfft(a, n=None, axis=-1, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if n is None: n = 2 * (a.shape[axis] - 1)
    cost = rfft_cost(n)
    budget.deduct("fft.irfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.irfft(a, n=n, axis=axis, norm=norm)

# 2-D transforms
def fft2(a, s=None, axes=(-2, -1), norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None: s = tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.fft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.fft2(a, s=s, axes=axes, norm=norm)

def ifft2(a, s=None, axes=(-2, -1), norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None: s = tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.ifft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ifft2(a, s=s, axes=axes, norm=norm)

def rfft2(a, s=None, axes=(-2, -1), norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None: s = tuple(a.shape[ax] for ax in axes)
    cost = rfftn_cost(s)
    budget.deduct("fft.rfft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.rfft2(a, s=s, axes=axes, norm=norm)

def irfft2(a, s=None, axes=(-2, -1), norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None: s = (a.shape[axes[0]], 2 * (a.shape[axes[1]] - 1))
    cost = rfftn_cost(s)
    budget.deduct("fft.irfft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.irfft2(a, s=s, axes=axes, norm=norm)

# N-D transforms
def fftn(a, s=None, axes=None, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = a.shape if axes is None else tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.fftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.fftn(a, s=s, axes=axes, norm=norm)

def ifftn(a, s=None, axes=None, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = a.shape if axes is None else tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.ifftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ifftn(a, s=s, axes=axes, norm=norm)

def rfftn(a, s=None, axes=None, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = a.shape if axes is None else tuple(a.shape[ax] for ax in axes)
    cost = rfftn_cost(s)
    budget.deduct("fft.rfftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.rfftn(a, s=s, axes=axes, norm=norm)

def irfftn(a, s=None, axes=None, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        if axes is None:
            s = tuple(d if i < len(a.shape) - 1 else 2 * (d - 1) for i, d in enumerate(a.shape))
        else:
            s = tuple(a.shape[ax] if i < len(axes) - 1 else 2 * (a.shape[ax] - 1) for i, ax in enumerate(axes))
    cost = rfftn_cost(s)
    budget.deduct("fft.irfftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.irfftn(a, s=s, axes=axes, norm=norm)

# Hermitian transforms
def hfft(a, n=None, axis=-1, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if n is None: n = 2 * (a.shape[axis] - 1)
    cost = hfft_cost(n)
    budget.deduct("fft.hfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.hfft(a, n=n, axis=axis, norm=norm)

def ihfft(a, n=None, axis=-1, norm=None):
    budget = require_budget()
    validate_ndarray(a)
    if n is None: n = a.shape[axis]
    cost = hfft_cost(n)
    budget.deduct("fft.ihfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ihfft(a, n=n, axis=axis, norm=norm)
