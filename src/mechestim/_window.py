# src/mechestim/_window.py
"""Window function wrappers with FLOP counting."""

from __future__ import annotations

import numpy as _np

from mechestim._docstrings import attach_docstring
from mechestim._validation import require_budget


def bartlett_cost(n: int) -> int:
    """FLOP cost of Bartlett window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: n.

    Notes
    -----
    One linear evaluation per sample.
    """
    return max(n, 1)


def bartlett(M):
    budget = require_budget()
    cost = bartlett_cost(M)
    budget.deduct("bartlett", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.bartlett(M)


attach_docstring(bartlett, _np.bartlett, "counted_custom", "n FLOPs")


def blackman_cost(n: int) -> int:
    """FLOP cost of Blackman window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 3n.

    Notes
    -----
    Three cosine terms per sample.
    """
    return max(3 * n, 1)


def blackman(M):
    budget = require_budget()
    cost = blackman_cost(M)
    budget.deduct("blackman", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.blackman(M)


attach_docstring(blackman, _np.blackman, "counted_custom", "3n FLOPs")


def hamming_cost(n: int) -> int:
    """FLOP cost of Hamming window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: n.

    Notes
    -----
    One FMA (cosine + scale) per sample, counted as 1 op under FMA=1.
    """
    return max(n, 1)


def hamming(M):
    budget = require_budget()
    cost = hamming_cost(M)
    budget.deduct("hamming", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.hamming(M)


attach_docstring(hamming, _np.hamming, "counted_custom", "n FLOPs")


def hanning_cost(n: int) -> int:
    """FLOP cost of Hanning window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: n.

    Notes
    -----
    One FMA (cosine + scale) per sample, counted as 1 op under FMA=1.
    """
    return max(n, 1)


def hanning(M):
    budget = require_budget()
    cost = hanning_cost(M)
    budget.deduct("hanning", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.hanning(M)


attach_docstring(hanning, _np.hanning, "counted_custom", "n FLOPs")


def kaiser_cost(n: int) -> int:
    """FLOP cost of Kaiser window generation.

    Parameters
    ----------
    n : int
        Window length.

    Returns
    -------
    int
        Estimated FLOP count: 3n.

    Notes
    -----
    Bessel function evaluation per sample.
    """
    return max(3 * n, 1)


def kaiser(M, beta):
    budget = require_budget()
    cost = kaiser_cost(M)
    budget.deduct("kaiser", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.kaiser(M, beta)


attach_docstring(kaiser, _np.kaiser, "counted_custom", "3n FLOPs")

import sys as _sys  # noqa: E402

from mechestim._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
