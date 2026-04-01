# src/mechestim/_window.py
"""Window function wrappers with FLOP counting."""
from __future__ import annotations
import numpy as _np
from mechestim._validation import require_budget

def bartlett_cost(n: int) -> int:
    """FLOP cost of Bartlett window. Formula: n. Source: One linear eval per sample."""
    return max(n, 1)

def bartlett(M):
    budget = require_budget()
    cost = bartlett_cost(M)
    budget.deduct("bartlett", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.bartlett(M)

def blackman_cost(n: int) -> int:
    """FLOP cost of Blackman window. Formula: 3*n. Source: Three cosine terms per sample."""
    return max(3 * n, 1)

def blackman(M):
    budget = require_budget()
    cost = blackman_cost(M)
    budget.deduct("blackman", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.blackman(M)

def hamming_cost(n: int) -> int:
    """FLOP cost of Hamming window. Formula: n. Source: One cosine per sample."""
    return max(n, 1)

def hamming(M):
    budget = require_budget()
    cost = hamming_cost(M)
    budget.deduct("hamming", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.hamming(M)

def hanning_cost(n: int) -> int:
    """FLOP cost of Hanning window. Formula: n. Source: One cosine per sample."""
    return max(n, 1)

def hanning(M):
    budget = require_budget()
    cost = hanning_cost(M)
    budget.deduct("hanning", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.hanning(M)

def kaiser_cost(n: int) -> int:
    """FLOP cost of Kaiser window. Formula: 3*n. Source: Bessel function eval per sample."""
    return max(3 * n, 1)

def kaiser(M, beta):
    budget = require_budget()
    cost = kaiser_cost(M)
    budget.deduct("kaiser", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.kaiser(M, beta)
