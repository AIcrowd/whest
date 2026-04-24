# src/flopscope/fft/_transforms.py
"""FFT transform wrappers with FLOP counting.

Cost model: 5 * N * log2(N) for complex DFT of length N (Cooley-Tukey radix-2).
Source: Cooley & Tukey (1965); Van Loan, "Computational Frameworks for the FFT" (1992), §1.4.
"""

from __future__ import annotations

import math

import numpy as _np

from flopscope._docstrings import attach_docstring
from flopscope._validation import require_budget


def fft_cost(n: int) -> int:
    """FLOP cost of a 1-D complex FFT.

    Parameters
    ----------
    n : int
        Transform length.

    Returns
    -------
    int
        Estimated FLOP count: 5 * n * ceil(log2(n)).

    Notes
    -----
    Source: Cooley & Tukey (1965); Van Loan, *Computational Frameworks
    for the FFT* (1992), §1.4. Assumes radix-2 Cooley-Tukey algorithm.
    """
    if n <= 1:
        return 0
    return 5 * n * math.ceil(math.log2(n))


def rfft_cost(n: int) -> int:
    """FLOP cost of a 1-D real FFT.

    Parameters
    ----------
    n : int
        Input length (real-valued).

    Returns
    -------
    int
        Estimated FLOP count: 5 * (n // 2) * ceil(log2(n)).

    Notes
    -----
    The real FFT exploits conjugate symmetry, roughly halving the work
    compared to a full complex FFT.
    """
    if n <= 1:
        return 0
    return 5 * (n // 2) * math.ceil(math.log2(n))


def fftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of an N-D complex FFT.

    Parameters
    ----------
    shape : tuple of int
        Transform shape along each axis.

    Returns
    -------
    int
        Estimated FLOP count: 5 * N * sum(ceil(log2(d_i))), where N = prod(shape).

    Notes
    -----
    The multi-dimensional FFT is computed as successive 1-D FFTs along each
    axis. The cost along axis *i* is 5 * (N / d_i) * d_i * ceil(log2(d_i))
    = 5 * N * ceil(log2(d_i)). Summing over axes gives
    5 * N * sum(ceil(log2(d_i))).
    """
    N = 1
    for d in shape:
        N *= d
    if N <= 1:
        return 0
    log_sum = sum(math.ceil(math.log2(d)) for d in shape if d > 1)
    return 5 * N * log_sum


def rfftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of an N-D real FFT.

    Parameters
    ----------
    shape : tuple of int
        Transform shape along each axis.

    Returns
    -------
    int
        Estimated FLOP count: 5 * (N // 2) * sum(ceil(log2(d_i))), where N = prod(shape).

    Notes
    -----
    Exploits conjugate symmetry along the last axis, roughly halving
    the work compared to a full complex N-D FFT.
    """
    N = 1
    for d in shape:
        N *= d
    if N <= 1:
        return 0
    log_sum = sum(math.ceil(math.log2(d)) for d in shape if d > 1)
    return 5 * (N // 2) * log_sum


def hfft_cost(n_out: int) -> int:
    """FLOP cost of a Hermitian FFT.

    Parameters
    ----------
    n_out : int
        Output length.

    Returns
    -------
    int
        Estimated FLOP count: 5 * n_out * ceil(log2(n_out)).

    Notes
    -----
    The Hermitian FFT computes the FFT of a signal with Hermitian symmetry,
    producing real-valued output.
    """
    if n_out <= 1:
        return 0
    return 5 * n_out * math.ceil(math.log2(n_out))


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _batch_count_1d(a: _np.ndarray, axis: int) -> int:
    """Number of independent 1-D transforms along *axis*."""
    if a.ndim == 0 or a.shape[axis] == 0:
        return 1
    return a.size // a.shape[axis]


def _batch_count_nd(a: _np.ndarray, axes: tuple[int, ...] | None) -> int:
    """Number of independent N-D transforms over *axes*."""
    if axes is None:
        return 1  # all axes are transform axes
    batch = a.size
    for ax in axes:
        batch //= a.shape[ax]
    return max(batch, 1)


# 1-D transforms
def fft(a, n=None, axis=-1, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * fft_cost(n)
    with budget.deduct("fft.fft", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.fft(a, n=n, axis=axis, norm=norm, out=out)
    return result


attach_docstring(
    fft, _np.fft.fft, "fft", "5 \u00d7 n \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * fft_cost(n)
    with budget.deduct("fft.ifft", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.ifft(a, n=n, axis=axis, norm=norm, out=out)
    return result


attach_docstring(
    ifft, _np.fft.ifft, "fft", "5 \u00d7 n \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


def rfft(a, n=None, axis=-1, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * rfft_cost(n)
    with budget.deduct("fft.rfft", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.rfft(a, n=n, axis=axis, norm=norm, out=out)
    return result


attach_docstring(
    rfft, _np.fft.rfft, "fft", "5 \u00d7 (n/2) \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


def irfft(a, n=None, axis=-1, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = 2 * (a.shape[axis] - 1)
    cost = _batch_count_1d(a, axis) * rfft_cost(n)
    with budget.deduct("fft.irfft", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.irfft(a, n=n, axis=axis, norm=norm, out=out)
    return result


attach_docstring(
    irfft, _np.fft.irfft, "fft", "5 \u00d7 (n/2) \u00d7 \u2308log\u2082(n)\u2309 FLOPs"
)


# 2-D transforms
def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = tuple(a.shape[ax] for ax in axes)
    else:
        s_for_cost = tuple(
            a.shape[axes[i]] if si is None else si for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * fftn_cost(s_for_cost)
    with budget.deduct("fft.fft2", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.fft2(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    fft2,
    _np.fft.fft2,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = tuple(a.shape[ax] for ax in axes)
    else:
        s_for_cost = tuple(
            a.shape[axes[i]] if si is None else si for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * fftn_cost(s_for_cost)
    with budget.deduct("fft.ifft2", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.ifft2(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    ifft2,
    _np.fft.ifft2,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = tuple(a.shape[ax] for ax in axes)
    else:
        s_for_cost = tuple(
            a.shape[axes[i]] if si is None else si for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * rfftn_cost(s_for_cost)
    with budget.deduct("fft.rfft2", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.rfft2(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    rfft2,
    _np.fft.rfft2,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = (a.shape[axes[0]], 2 * (a.shape[axes[1]] - 1))
    else:
        s_for_cost = tuple(
            (a.shape[axes[i]] if i < len(s) - 1 else 2 * (a.shape[axes[i]] - 1))
            if si is None
            else si
            for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * rfftn_cost(s_for_cost)
    with budget.deduct(
        "fft.irfft2", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.fft.irfft2(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    irfft2,
    _np.fft.irfft2,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


# N-D transforms
def fftn(a, s=None, axes=None, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = a.shape if axes is None else tuple(a.shape[ax] for ax in axes)
    else:
        eff_axes = tuple(range(a.ndim)) if axes is None else tuple(axes)
        s_for_cost = tuple(
            a.shape[eff_axes[i]] if si is None else si for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * fftn_cost(s_for_cost)
    with budget.deduct("fft.fftn", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.fftn(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    fftn,
    _np.fft.fftn,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


def ifftn(a, s=None, axes=None, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = a.shape if axes is None else tuple(a.shape[ax] for ax in axes)
    else:
        eff_axes = tuple(range(a.ndim)) if axes is None else tuple(axes)
        s_for_cost = tuple(
            a.shape[eff_axes[i]] if si is None else si for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * fftn_cost(s_for_cost)
    with budget.deduct("fft.ifftn", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.ifftn(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    ifftn,
    _np.fft.ifftn,
    "fft",
    "5 \u00d7 N \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


def rfftn(a, s=None, axes=None, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        s_for_cost = a.shape if axes is None else tuple(a.shape[ax] for ax in axes)
    else:
        eff_axes = tuple(range(a.ndim)) if axes is None else tuple(axes)
        s_for_cost = tuple(
            a.shape[eff_axes[i]] if si is None else si for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * rfftn_cost(s_for_cost)
    with budget.deduct("fft.rfftn", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.rfftn(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    rfftn,
    _np.fft.rfftn,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


def irfftn(a, s=None, axes=None, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if s is None:
        if axes is None:
            s_for_cost = tuple(
                d if i < len(a.shape) - 1 else 2 * (d - 1)
                for i, d in enumerate(a.shape)
            )
        else:
            s_for_cost = tuple(
                a.shape[ax] if i < len(axes) - 1 else 2 * (a.shape[ax] - 1)
                for i, ax in enumerate(axes)
            )
    else:
        eff_axes = tuple(range(a.ndim)) if axes is None else tuple(axes)
        s_for_cost = tuple(
            (a.shape[eff_axes[i]] if i < len(s) - 1 else 2 * (a.shape[eff_axes[i]] - 1))
            if si is None
            else si
            for i, si in enumerate(s)
        )
    cost = _batch_count_nd(a, axes) * rfftn_cost(s_for_cost)
    with budget.deduct(
        "fft.irfftn", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.fft.irfftn(a, s=s, axes=axes, norm=norm, out=out)
    return result


attach_docstring(
    irfftn,
    _np.fft.irfftn,
    "fft",
    "5 \u00d7 (N/2) \u00d7 \u2308log\u2082(N)\u2309 FLOPs, N = product of transform shape",
)


# Hermitian transforms
def hfft(a, n=None, axis=-1, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = 2 * (a.shape[axis] - 1)
    cost = _batch_count_1d(a, axis) * hfft_cost(n)
    with budget.deduct("fft.hfft", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.hfft(a, n=n, axis=axis, norm=norm, out=out)
    return result


attach_docstring(
    hfft,
    _np.fft.hfft,
    "fft",
    "5 \u00d7 n_out \u00d7 \u2308log\u2082(n_out)\u2309 FLOPs",
)


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if n is None:
        n = a.shape[axis]
    cost = _batch_count_1d(a, axis) * hfft_cost(n)
    with budget.deduct("fft.ihfft", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.fft.ihfft(a, n=n, axis=axis, norm=norm, out=out)
    return result


attach_docstring(
    ihfft,
    _np.fft.ihfft,
    "fft",
    "5 \u00d7 n_out \u00d7 \u2308log\u2082(n_out)\u2309 FLOPs",
)

import sys as _sys  # noqa: E402

from flopscope._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
