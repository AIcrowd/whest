"""Counted wrappers for ``numpy.random``.

Most samplers deduct ``numel(output)`` FLOPs from the active budget.
Shuffle-like operations (``permutation``, ``shuffle``, ``choice`` without
replacement) deduct ``n * ceil(log2(n))`` FLOPs.

Configuration helpers (``seed``, ``get_state``, ``set_state``,
``default_rng``, ``RandomState``, ``SeedSequence``) are free.

Any attribute not listed here is forwarded to ``numpy.random`` via
``__getattr__`` without budget deduction.
"""

from __future__ import annotations

import builtins as _builtins
import inspect as _inspect

import numpy as _np
import numpy.random as _npr

from whest._flops import _ceil_log2, sort_cost
from whest._perm_group import PermutationGroup
from whest._validation import require_budget

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
    """Factory for simple samplers: cost = numel(output).

    Passes all args/kwargs through transparently to numpy. Derives the
    output size from the actual result to correctly handle ``size`` passed
    as either a positional or keyword argument.
    """

    def wrapper(*args, **kwargs):
        budget = require_budget()
        result = np_func(*args, **kwargs)
        if isinstance(result, _np.ndarray):
            n = _builtins.max(result.size, 1)
        elif isinstance(result, (int, float, _np.integer, _np.floating)):
            n = 1
        else:
            n = 1
        with budget.deduct(op_name, flop_cost=n, subscripts=None, shapes=((n,),)):
            pass  # numpy already executed
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


def _counted_dims_sampler(np_func, op_name):
    """Factory for rand/randn that take *dims instead of size=."""

    def wrapper(*dims):
        budget = require_budget()
        n = int(_np.prod(dims)) if dims else 1
        cost = _builtins.max(n, 1)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=((n,),)):
            if dims:
                result = np_func(*dims)
            else:
                result = np_func()
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


# ---------------------------------------------------------------------------
# Free (configuration) functions
# ---------------------------------------------------------------------------

seed = _npr.seed
get_state = _npr.get_state
set_state = _npr.set_state
default_rng = _npr.default_rng
RandomState = _npr.RandomState
SeedSequence = _npr.SeedSequence


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


def _counted_size_only_sampler(np_func, op_name):
    """Factory for samplers where the only arg is ``size`` (positional or kw)."""

    def wrapper(size=None):
        budget = require_budget()
        n = _output_size(size=size)
        cost = _builtins.max(n, 1)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=((n,),)):
            result = np_func(size=size)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


random = _counted_size_only_sampler(_npr.random, "random.random")
random_sample = _counted_size_only_sampler(_npr.random_sample, "random.random_sample")
ranf = _counted_size_only_sampler(_npr.ranf, "random.ranf")
sample = _counted_size_only_sampler(_npr.sample, "random.sample")


# ---------------------------------------------------------------------------
# Hand-coded ops with special cost formulas
# ---------------------------------------------------------------------------


def permutation(x):
    """Counted version of ``numpy.random.permutation``.

    Cost: numel(output) FLOPs.
    """
    budget = require_budget()
    n = int(x) if isinstance(x, (int, _np.integer)) else x.shape[0]
    cost = _builtins.max(n, 1)
    with budget.deduct(
        "random.permutation", flop_cost=cost, subscripts=None, shapes=((n,),)
    ):
        result = _npr.permutation(x)
    return result


def shuffle(x):
    """Counted version of ``numpy.random.shuffle``.

    Modifies ``x`` in-place. Cost: numel(output) FLOPs.
    """
    budget = require_budget()
    if hasattr(x, "shape"):
        n = x.shape[0]
    else:
        n = len(x)
    cost = _builtins.max(n, 1)
    with budget.deduct(
        "random.shuffle", flop_cost=cost, subscripts=None, shapes=((n,),)
    ):
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
        with budget.deduct(
            "random.choice", flop_cost=cost, subscripts=None, shapes=((out_size,),)
        ):
            result = _npr.choice(a, size=size, replace=replace, p=p)
    else:
        cost = sort_cost(n)
        with budget.deduct(
            "random.choice", flop_cost=cost, subscripts=None, shapes=((n,),)
        ):
            result = _npr.choice(a, size=size, replace=replace, p=p)
    return result


def symmetric(
    shape: int | tuple[int, ...] | list[int],
    group: PermutationGroup,
    distribution: str | callable = "randn",
    **distribution_kwargs,
):
    """Sample random data and project it to a symmetry group.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the sampled array.
    group : PermutationGroup
        Symmetry group used for Reynolds averaging.
    distribution : str or callable, default ``\"randn\"``
        Name of a ``numpy.random`` distribution function (for example
        ``\"randn\"`` or ``\"normal\"``), or a callable that accepts either:

        - ``(*shape, **kwargs)``
        - ``size=shape``

        and returns an array.
    **distribution_kwargs
        Extra keyword arguments forwarded to the distribution function.

    Returns
    -------
    SymmetricTensor
        The symmetrized sample wrapped with :func:`we.as_symmetric`.

    Raises
    ------
    ValueError
        If ``shape`` is not an integer or a tuple/list of integers.
    TypeError
        If ``distribution`` is neither a NumPy random distribution name nor a
        callable.
    AttributeError
        If ``distribution`` is a string that is not present in NumPy random.
    SymmetryError
        If projected output does not satisfy the requested symmetry constraints.

    Notes
    -----
    This is equivalent to ``we.symmetrize( sampled_data, group)`` where
    ``sampled_data`` is drawn from ``distribution``.

    The implementation currently:

    1. Samples raw values from the selected distribution.
    2. Applies :func:`we.symmetrize` to project into the symmetry-invariant
       subspace.

    Estimated FLOP cost is approximately:

    ``C_dist(n_elem) + |G| * n_elem + n_elem`` (+ validation),

    where ``n_elem`` is the sampled array size, ``|G|`` is the group order, and
    ``C_dist(n_elem)`` is the cost of the chosen sampling distribution.
    The default ``distribution='randn'`` corresponds to ``C_dist(n_elem)≈n_elem``.

    For existing data, use :func:`we.symmetrize` directly.

    Examples
    --------
    >>> import whest as we
    >>> S = we.random.symmetric((4, 4), we.PermutationGroup.symmetric(2, axes=(0, 1)))
    >>> S.is_symmetric((0, 1))
    True

    >>> S = we.random.symmetric(
    ...     (3, 3, 3),
    ...     we.PermutationGroup.cyclic(3, axes=(0, 1, 2)),
    ...     distribution="normal",
    ...     loc=0.0,
    ...     scale=1.0,
    ... )
    >>> S.is_symmetric((0, 1, 2))
    True

    >>> import numpy as np
    >>> import whest as we
    >>> def shifted_uniform(shape, **kwargs):
    ...     return np.random.uniform(*shape, **kwargs)
    >>> S = we.random.symmetric((2, 2), we.PermutationGroup.symmetric(2, axes=(0, 1)), distribution=shifted_uniform)
    >>> S.is_symmetric((0, 1))
    True
    """
    if isinstance(shape, int):
        shape_tuple = (shape,)
    elif isinstance(shape, tuple):
        shape_tuple = shape
    elif isinstance(shape, list):
        shape_tuple = tuple(shape)
    else:
        try:
            shape_tuple = tuple(shape)
        except TypeError as exc:
            raise ValueError("shape must be an int or a tuple-like of ints") from exc

    shape_tuple = tuple(int(s) for s in shape_tuple)
    sample_size = _builtins.max(int(_np.prod(shape_tuple)), 1)

    if isinstance(distribution, str):
        if not hasattr(_npr, distribution):
            raise AttributeError(
                f"whest.random does not provide distribution '{distribution}'"
            )
        sampler = getattr(_npr, distribution)
        if distribution in {"rand", "randn"}:
            sample = sampler(*shape_tuple, **distribution_kwargs)
        else:
            sample = sampler(size=shape_tuple, **distribution_kwargs)
    elif callable(distribution):
        try:
            sample = distribution(*shape_tuple, **distribution_kwargs)
        except TypeError:
            sample = distribution(size=shape_tuple, **distribution_kwargs)
    else:
        raise TypeError(
            "distribution must be a numpy random function name or a callable"
        )

    budget = require_budget()
    sym_cost = _builtins.max(sample_size * _builtins.max(group.order(), 1), 1)
    sample_cost = sample_size
    with budget.deduct(
        "random.symmetric",
        flop_cost=_builtins.max(sample_cost + sym_cost, 1),
        subscripts=None,
        shapes=(shape_tuple,),
    ):
        from whest import symmetrize

        return symmetrize(sample, group)


def bytes(length):
    """Counted version of ``numpy.random.bytes``.

    Cost: ``length`` FLOPs.
    """
    budget = require_budget()
    cost = _builtins.max(int(length), 1)
    with budget.deduct(
        "random.bytes", flop_cost=cost, subscripts=None, shapes=((length,),)
    ):
        result = _npr.bytes(length)
    return result


# ---------------------------------------------------------------------------
# Fallback __getattr__ for anything not explicitly listed
# ---------------------------------------------------------------------------


def __getattr__(name):
    if hasattr(_npr, name):
        return getattr(_npr, name)
    raise AttributeError(f"whest.random does not provide '{name}'")


import sys as _sys  # noqa: E402

from whest._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(
    _sys.modules[__name__],
    skip_names={
        "default_rng",
        "RandomState",
        "SeedSequence",
        "seed",
        "get_state",
        "set_state",
    },
)
