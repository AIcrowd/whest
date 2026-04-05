"""Counted wrappers for ``numpy.random``.

Most samplers deduct ``numel(output)`` FLOPs from the active budget.
Shuffle-like operations (``permutation``, ``shuffle``, ``choice`` without
replacement) deduct ``n * ceil(log2(n))`` FLOPs.

Configuration helpers (``seed``, ``get_state``, ``set_state``,
``default_rng``, ``RandomState``) are free.

Any attribute not listed here is forwarded to ``numpy.random`` via
``__getattr__`` without budget deduction.
"""

from __future__ import annotations

import builtins as _builtins

import numpy as _np
import numpy.random as _npr

from mechestim._flops import _ceil_log2, sort_cost
from mechestim._validation import require_budget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_size(*dims, size=None):
    """Compute the total number of output elements."""
    if size is not None:
        if isinstance(size, (int, _np.integer)):
            return int(size)
        return int(_np.prod(size))
    if dims:
        return int(_np.prod(dims))
    return 1


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _counted_sampler(np_func, op_name):
    """Factory for simple samplers: cost = numel(output)."""

    def wrapper(*args, size=None, **kwargs):
        budget = require_budget()
        n = _output_size(size=size)
        cost = _builtins.max(n, 1)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=((n,),))
        return np_func(*args, size=size, **kwargs)

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    return wrapper


def _counted_dims_sampler(np_func, op_name):
    """Factory for rand/randn that take *dims instead of size=."""

    def wrapper(*dims):
        budget = require_budget()
        n = int(_np.prod(dims)) if dims else 1
        cost = _builtins.max(n, 1)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=((n,),))
        if dims:
            return np_func(*dims)
        return np_func()

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    return wrapper


# ---------------------------------------------------------------------------
# Free (configuration) functions
# ---------------------------------------------------------------------------

seed = _npr.seed
get_state = _npr.get_state
set_state = _npr.set_state
default_rng = _npr.default_rng
RandomState = _npr.RandomState


# ---------------------------------------------------------------------------
# Dims-based samplers (rand, randn)
# ---------------------------------------------------------------------------

rand = _counted_dims_sampler(_npr.rand, "random.rand")
randn = _counted_dims_sampler(_npr.randn, "random.randn")


# ---------------------------------------------------------------------------
# Simple samplers (cost = numel(output))
# ---------------------------------------------------------------------------

normal = _counted_sampler(_npr.normal, "random.normal")
uniform = _counted_sampler(_npr.uniform, "random.uniform")
standard_normal = _counted_sampler(_npr.standard_normal, "random.standard_normal")
standard_exponential = _counted_sampler(
    _npr.standard_exponential, "random.standard_exponential"
)
exponential = _counted_sampler(_npr.exponential, "random.exponential")
poisson = _counted_sampler(_npr.poisson, "random.poisson")
binomial = _counted_sampler(_npr.binomial, "random.binomial")
geometric = _counted_sampler(_npr.geometric, "random.geometric")
hypergeometric = _counted_sampler(_npr.hypergeometric, "random.hypergeometric")
negative_binomial = _counted_sampler(_npr.negative_binomial, "random.negative_binomial")
logseries = _counted_sampler(_npr.logseries, "random.logseries")
power = _counted_sampler(_npr.power, "random.power")
pareto = _counted_sampler(_npr.pareto, "random.pareto")
rayleigh = _counted_sampler(_npr.rayleigh, "random.rayleigh")
standard_cauchy = _counted_sampler(_npr.standard_cauchy, "random.standard_cauchy")
standard_t = _counted_sampler(_npr.standard_t, "random.standard_t")
standard_gamma = _counted_sampler(_npr.standard_gamma, "random.standard_gamma")
weibull = _counted_sampler(_npr.weibull, "random.weibull")
zipf = _counted_sampler(_npr.zipf, "random.zipf")
gumbel = _counted_sampler(_npr.gumbel, "random.gumbel")
laplace = _counted_sampler(_npr.laplace, "random.laplace")
logistic = _counted_sampler(_npr.logistic, "random.logistic")
lognormal = _counted_sampler(_npr.lognormal, "random.lognormal")
vonmises = _counted_sampler(_npr.vonmises, "random.vonmises")
wald = _counted_sampler(_npr.wald, "random.wald")
triangular = _counted_sampler(_npr.triangular, "random.triangular")
chisquare = _counted_sampler(_npr.chisquare, "random.chisquare")
noncentral_chisquare = _counted_sampler(
    _npr.noncentral_chisquare, "random.noncentral_chisquare"
)
noncentral_f = _counted_sampler(_npr.noncentral_f, "random.noncentral_f")
f = _counted_sampler(_npr.f, "random.f")
beta = _counted_sampler(_npr.beta, "random.beta")
gamma = _counted_sampler(_npr.gamma, "random.gamma")
multinomial = _counted_sampler(_npr.multinomial, "random.multinomial")
multivariate_normal = _counted_sampler(
    _npr.multivariate_normal, "random.multivariate_normal"
)
dirichlet = _counted_sampler(_npr.dirichlet, "random.dirichlet")
randint = _counted_sampler(_npr.randint, "random.randint")
random = _counted_sampler(_npr.random, "random.random")
random_sample = _counted_sampler(_npr.random_sample, "random.random_sample")
ranf = _counted_sampler(_npr.ranf, "random.ranf")
sample = _counted_sampler(_npr.sample, "random.sample")


# ---------------------------------------------------------------------------
# Hand-coded ops with special cost formulas
# ---------------------------------------------------------------------------


def permutation(x):
    """Counted version of ``numpy.random.permutation``.

    Cost: sort_cost(n) = n * ceil(log2(n)) FLOPs.
    """
    budget = require_budget()
    n = int(x) if isinstance(x, (int, _np.integer)) else x.shape[0]
    cost = sort_cost(n)
    budget.deduct("random.permutation", flop_cost=cost, subscripts=None, shapes=((n,),))
    return _npr.permutation(x)


def shuffle(x):
    """Counted version of ``numpy.random.shuffle``.

    Modifies ``x`` in-place. Cost: sort_cost(n) = n * ceil(log2(n)) FLOPs.
    """
    budget = require_budget()
    n = x.shape[0]
    cost = sort_cost(n)
    budget.deduct("random.shuffle", flop_cost=cost, subscripts=None, shapes=((n,),))
    _npr.shuffle(x)


def choice(a, size=None, replace=True, p=None):
    """Counted version of ``numpy.random.choice``.

    Cost: numel(output) FLOPs if ``replace=True``;
    sort_cost(n) = n * ceil(log2(n)) FLOPs if ``replace=False``.
    """
    budget = require_budget()
    if isinstance(a, (int, _np.integer)):
        n = int(a)
    else:
        a_arr = _np.asarray(a)
        n = a_arr.shape[0] if a_arr.ndim > 0 else 1
    if replace:
        out_size = _output_size(size=size)
        cost = _builtins.max(out_size, 1)
        budget.deduct(
            "random.choice", flop_cost=cost, subscripts=None, shapes=((out_size,),)
        )
    else:
        cost = sort_cost(n)
        budget.deduct("random.choice", flop_cost=cost, subscripts=None, shapes=((n,),))
    return _npr.choice(a, size=size, replace=replace, p=p)


def bytes(length):
    """Counted version of ``numpy.random.bytes``.

    Cost: ``length`` FLOPs.
    """
    budget = require_budget()
    cost = _builtins.max(int(length), 1)
    budget.deduct("random.bytes", flop_cost=cost, subscripts=None, shapes=((length,),))
    return _npr.bytes(length)


# ---------------------------------------------------------------------------
# Fallback __getattr__ for anything not explicitly listed
# ---------------------------------------------------------------------------


def __getattr__(name):
    if hasattr(_npr, name):
        return getattr(_npr, name)
    raise AttributeError(f"mechestim.random does not provide '{name}'")
