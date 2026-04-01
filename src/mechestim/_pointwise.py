"""Counted pointwise operations and reductions for mechestim."""
from __future__ import annotations

import numpy as _np

from mechestim._budget import BudgetContext
from mechestim._flops import einsum_cost, pointwise_cost, reduction_cost
from mechestim._validation import check_nan_inf, require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _counted_unary(np_func, op_name: str):
    def wrapper(x):
        budget = require_budget()
        validate_ndarray(x)
        cost = pointwise_cost(x.shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,))
        result = np_func(x)
        check_nan_inf(result, op_name)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    return wrapper


def _counted_binary(np_func, op_name: str):
    def wrapper(x, y):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        cost = pointwise_cost(output_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape))
        result = np_func(x, y)
        check_nan_inf(result, op_name)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    return wrapper


def _counted_reduction(np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False):
    def wrapper(a, axis=None, **kwargs):
        budget = require_budget()
        validate_ndarray(a)
        cost = reduction_cost(a.shape, axis) * cost_multiplier
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    return wrapper


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------

exp = _counted_unary(_np.exp, "exp")
log = _counted_unary(_np.log, "log")
log2 = _counted_unary(_np.log2, "log2")
log10 = _counted_unary(_np.log10, "log10")
abs = _counted_unary(_np.abs, "abs")
negative = _counted_unary(_np.negative, "negative")
sqrt = _counted_unary(_np.sqrt, "sqrt")
square = _counted_unary(_np.square, "square")
sin = _counted_unary(_np.sin, "sin")
cos = _counted_unary(_np.cos, "cos")
tanh = _counted_unary(_np.tanh, "tanh")
sign = _counted_unary(_np.sign, "sign")
ceil = _counted_unary(_np.ceil, "ceil")
floor = _counted_unary(_np.floor, "floor")


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

add = _counted_binary(_np.add, "add")
subtract = _counted_binary(_np.subtract, "subtract")
multiply = _counted_binary(_np.multiply, "multiply")
divide = _counted_binary(_np.divide, "divide")
maximum = _counted_binary(_np.maximum, "maximum")
minimum = _counted_binary(_np.minimum, "minimum")
power = _counted_binary(_np.power, "power")
mod = _counted_binary(_np.mod, "mod")


# ---------------------------------------------------------------------------
# Special: clip
# ---------------------------------------------------------------------------

def clip(a, a_min, a_max):
    """Counted version of np.clip. Cost = numel(input)."""
    budget = require_budget()
    validate_ndarray(a)
    cost = pointwise_cost(a.shape)
    budget.deduct("clip", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    result = _np.clip(a, a_min, a_max)
    check_nan_inf(result, "clip")
    return result


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

sum = _counted_reduction(_np.sum, "sum")
max = _counted_reduction(_np.max, "max")
min = _counted_reduction(_np.min, "min")
prod = _counted_reduction(_np.prod, "prod")
mean = _counted_reduction(_np.mean, "mean", extra_output=True)
std = _counted_reduction(_np.std, "std", cost_multiplier=2, extra_output=True)
var = _counted_reduction(_np.var, "var", cost_multiplier=2, extra_output=True)
argmax = _counted_reduction(_np.argmax, "argmax")
argmin = _counted_reduction(_np.argmin, "argmin")
cumsum = _counted_reduction(_np.cumsum, "cumsum")
cumprod = _counted_reduction(_np.cumprod, "cumprod")


# ---------------------------------------------------------------------------
# dot and matmul
# ---------------------------------------------------------------------------

def dot(a, b):
    """Counted version of np.dot."""
    budget = require_budget()
    validate_ndarray(a, b)
    if a.ndim == 2 and b.ndim == 2:
        cost = einsum_cost("ij,jk->ik", shapes=[a.shape, b.shape])
    elif a.ndim == 1 and b.ndim == 1:
        cost = einsum_cost("i,i->", shapes=[a.shape, b.shape])
    else:
        cost = a.size * b.size
    budget.deduct("dot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.dot(a, b)
    check_nan_inf(result, "dot")
    return result


def matmul(a, b):
    """Counted version of np.matmul."""
    budget = require_budget()
    validate_ndarray(a, b)
    if a.ndim == 2 and b.ndim == 2:
        cost = einsum_cost("ij,jk->ik", shapes=[a.shape, b.shape])
    elif a.ndim == 1 and b.ndim == 1:
        cost = einsum_cost("i,i->", shapes=[a.shape, b.shape])
    else:
        cost = a.size * b.size
    budget.deduct("matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.matmul(a, b)
    check_nan_inf(result, "matmul")
    return result
