"""Counted pointwise operations and reductions for mechestim."""

from __future__ import annotations

import builtins as _builtins

import numpy as _np

from mechestim._docstrings import attach_docstring
from mechestim._flops import einsum_cost, pointwise_cost, reduction_cost
from mechestim._symmetric import SymmetricTensor, SymmetryInfo
from mechestim._validation import check_nan_inf, require_budget, validate_ndarray

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _counted_unary(np_func, op_name: str):
    def wrapper(x):
        budget = require_budget()
        validate_ndarray(x)
        sym_info = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        cost = pointwise_cost(x.shape, symmetry_info=sym_info)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,))
        result = np_func(x)
        check_nan_inf(result, op_name)
        if sym_info is not None:
            result = SymmetricTensor(result, symmetric_dims=sym_info.symmetric_dims)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    return wrapper


def _counted_unary_multi(np_func, op_name: str):
    """Factory for unary functions that return multiple arrays (e.g., modf, frexp)."""

    def wrapper(x):
        budget = require_budget()
        validate_ndarray(x)
        cost = pointwise_cost(x.shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,))
        result = np_func(x)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(input) FLOPs")
    return wrapper


def _counted_binary(np_func, op_name: str):
    def wrapper(x, y):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        x_sym = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        y_sym = y.symmetry_info if isinstance(y, SymmetricTensor) else None
        x_is_scalar = x.ndim == 0
        y_is_scalar = y.ndim == 0
        if x_sym and y_sym and x_sym.symmetric_dims == y_sym.symmetric_dims:
            out_sym_info = SymmetryInfo(
                symmetric_dims=x_sym.symmetric_dims, shape=output_shape
            )
            out_sym_dims = x_sym.symmetric_dims
        elif x_sym and y_is_scalar:
            out_sym_info = SymmetryInfo(
                symmetric_dims=x_sym.symmetric_dims, shape=output_shape
            )
            out_sym_dims = x_sym.symmetric_dims
        elif y_sym and x_is_scalar:
            out_sym_info = SymmetryInfo(
                symmetric_dims=y_sym.symmetric_dims, shape=output_shape
            )
            out_sym_dims = y_sym.symmetric_dims
        else:
            out_sym_info = None
            out_sym_dims = None
        cost = pointwise_cost(output_shape, symmetry_info=out_sym_info)
        budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        )
        result = np_func(x, y)
        check_nan_inf(result, op_name)
        if out_sym_dims is not None:
            result = SymmetricTensor(result, symmetric_dims=out_sym_dims)
        elif isinstance(result, SymmetricTensor):
            result = _np.asarray(result)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    return wrapper


def _counted_binary_multi(np_func, op_name: str):
    """Factory for binary functions that return multiple arrays (e.g., divmod)."""

    def wrapper(x, y):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        cost = pointwise_cost(output_shape)
        budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        )
        result = np_func(x, y)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    return wrapper


def _counted_reduction(
    np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False
):
    def wrapper(a, axis=None, **kwargs):
        budget = require_budget()
        validate_ndarray(a)
        sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
        cost = reduction_cost(a.shape, axis, symmetry_info=sym_info) * cost_multiplier
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
        if isinstance(result, SymmetricTensor):
            result = _np.asarray(result)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    cost_desc = (
        f"numel(input) * {cost_multiplier} FLOPs"
        if cost_multiplier > 1
        else "numel(input) FLOPs"
    )
    if extra_output:
        cost_desc += " + numel(output)"
    attach_docstring(wrapper, np_func, "counted_reduction", cost_desc)
    return wrapper


# ---------------------------------------------------------------------------
# Unary ops (original)
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
# Unary ops (new)
# ---------------------------------------------------------------------------

absolute = _counted_unary(_np.absolute, "absolute")
acos = _counted_unary(_np.acos, "acos")
acosh = _counted_unary(_np.acosh, "acosh")
angle = _counted_unary(_np.angle, "angle")
arccos = _counted_unary(_np.arccos, "arccos")
arccosh = _counted_unary(_np.arccosh, "arccosh")
arcsin = _counted_unary(_np.arcsin, "arcsin")
arcsinh = _counted_unary(_np.arcsinh, "arcsinh")
arctan = _counted_unary(_np.arctan, "arctan")
arctanh = _counted_unary(_np.arctanh, "arctanh")
around = _counted_unary(_np.around, "around")
asin = _counted_unary(_np.asin, "asin")
asinh = _counted_unary(_np.asinh, "asinh")
atan = _counted_unary(_np.atan, "atan")
atanh = _counted_unary(_np.atanh, "atanh")
bitwise_count = _counted_unary(_np.bitwise_count, "bitwise_count")
bitwise_invert = _counted_unary(_np.bitwise_invert, "bitwise_invert")
bitwise_not = _counted_unary(_np.bitwise_not, "bitwise_not")
cbrt = _counted_unary(_np.cbrt, "cbrt")
conj = _counted_unary(_np.conj, "conj")
conjugate = _counted_unary(_np.conjugate, "conjugate")
cosh = _counted_unary(_np.cosh, "cosh")
deg2rad = _counted_unary(_np.deg2rad, "deg2rad")
degrees = _counted_unary(_np.degrees, "degrees")
exp2 = _counted_unary(_np.exp2, "exp2")
expm1 = _counted_unary(_np.expm1, "expm1")
fabs = _counted_unary(_np.fabs, "fabs")
fix = _counted_unary(_np.fix, "fix")
i0 = _counted_unary(_np.i0, "i0")
imag = _counted_unary(_np.imag, "imag")
invert = _counted_unary(_np.invert, "invert")
iscomplex = _counted_unary(_np.iscomplex, "iscomplex")
iscomplexobj = _counted_unary(_np.iscomplexobj, "iscomplexobj")
isnat = _counted_unary(_np.isnat, "isnat")
isneginf = _counted_unary(_np.isneginf, "isneginf")
isposinf = _counted_unary(_np.isposinf, "isposinf")
isreal = _counted_unary(_np.isreal, "isreal")
isrealobj = _counted_unary(_np.isrealobj, "isrealobj")
log1p = _counted_unary(_np.log1p, "log1p")
logical_not = _counted_unary(_np.logical_not, "logical_not")
nan_to_num = _counted_unary(_np.nan_to_num, "nan_to_num")
positive = _counted_unary(_np.positive, "positive")
rad2deg = _counted_unary(_np.rad2deg, "rad2deg")
radians = _counted_unary(_np.radians, "radians")
real = _counted_unary(_np.real, "real")
real_if_close = _counted_unary(_np.real_if_close, "real_if_close")
reciprocal = _counted_unary(_np.reciprocal, "reciprocal")
rint = _counted_unary(_np.rint, "rint")
round = _counted_unary(_np.round, "round")
signbit = _counted_unary(_np.signbit, "signbit")
sinc = _counted_unary(_np.sinc, "sinc")
sinh = _counted_unary(_np.sinh, "sinh")
sort_complex = _counted_unary(_np.sort_complex, "sort_complex")
spacing = _counted_unary(_np.spacing, "spacing")
tan = _counted_unary(_np.tan, "tan")
trunc = _counted_unary(_np.trunc, "trunc")

# Multi-output unary ops
modf = _counted_unary_multi(_np.modf, "modf")
frexp = _counted_unary_multi(_np.frexp, "frexp")


# isclose is binary (takes 2 args) but classified as unary in registry
def isclose(a, b, **kwargs):
    """Counted version of np.isclose. Cost = numel(output) FLOPs."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    output_shape = _np.broadcast_shapes(a.shape, b.shape)
    cost = pointwise_cost(output_shape)
    budget.deduct("isclose", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    return _np.isclose(a, b, **kwargs)


attach_docstring(isclose, _np.isclose, "counted_unary", "numel(output) FLOPs")


# ---------------------------------------------------------------------------
# Binary ops (original)
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
# Binary ops (new)
# ---------------------------------------------------------------------------

arctan2 = _counted_binary(_np.arctan2, "arctan2")
atan2 = _counted_binary(_np.atan2, "atan2")
bitwise_and = _counted_binary(_np.bitwise_and, "bitwise_and")
bitwise_left_shift = _counted_binary(_np.bitwise_left_shift, "bitwise_left_shift")
bitwise_or = _counted_binary(_np.bitwise_or, "bitwise_or")
bitwise_right_shift = _counted_binary(_np.bitwise_right_shift, "bitwise_right_shift")
bitwise_xor = _counted_binary(_np.bitwise_xor, "bitwise_xor")
copysign = _counted_binary(_np.copysign, "copysign")
equal = _counted_binary(_np.equal, "equal")
float_power = _counted_binary(_np.float_power, "float_power")
floor_divide = _counted_binary(_np.floor_divide, "floor_divide")
fmax = _counted_binary(_np.fmax, "fmax")
fmin = _counted_binary(_np.fmin, "fmin")
fmod = _counted_binary(_np.fmod, "fmod")
gcd = _counted_binary(_np.gcd, "gcd")
greater = _counted_binary(_np.greater, "greater")
greater_equal = _counted_binary(_np.greater_equal, "greater_equal")
heaviside = _counted_binary(_np.heaviside, "heaviside")
hypot = _counted_binary(_np.hypot, "hypot")
lcm = _counted_binary(_np.lcm, "lcm")
ldexp = _counted_binary(_np.ldexp, "ldexp")
left_shift = _counted_binary(_np.left_shift, "left_shift")
less = _counted_binary(_np.less, "less")
less_equal = _counted_binary(_np.less_equal, "less_equal")
logaddexp = _counted_binary(_np.logaddexp, "logaddexp")
logaddexp2 = _counted_binary(_np.logaddexp2, "logaddexp2")
logical_and = _counted_binary(_np.logical_and, "logical_and")
logical_or = _counted_binary(_np.logical_or, "logical_or")
logical_xor = _counted_binary(_np.logical_xor, "logical_xor")
nextafter = _counted_binary(_np.nextafter, "nextafter")
not_equal = _counted_binary(_np.not_equal, "not_equal")
pow = _counted_binary(_np.pow, "pow")
remainder = _counted_binary(_np.remainder, "remainder")
right_shift = _counted_binary(_np.right_shift, "right_shift")
true_divide = _counted_binary(_np.true_divide, "true_divide")
vecdot = _counted_binary(_np.vecdot, "vecdot")

# Multi-output binary ops
divmod = _counted_binary_multi(_np.divmod, "divmod")


# ---------------------------------------------------------------------------
# Special: clip
# ---------------------------------------------------------------------------


def clip(a, a_min, a_max):
    """Counted version of np.clip. Cost = numel(input) or unique_elements if symmetric."""
    budget = require_budget()
    validate_ndarray(a)
    sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
    cost = pointwise_cost(a.shape, symmetry_info=sym_info)
    budget.deduct("clip", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    result = _np.clip(a, a_min, a_max)
    check_nan_inf(result, "clip")
    if sym_info is not None:
        result = SymmetricTensor(result, symmetric_dims=sym_info.symmetric_dims)
    return result


attach_docstring(clip, _np.clip, "counted_custom", "numel(input) FLOPs")


# ---------------------------------------------------------------------------
# Reductions (original)
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
# Reductions (new)
# ---------------------------------------------------------------------------

all = _counted_reduction(_np.all, "all")
amax = _counted_reduction(_np.amax, "amax")
amin = _counted_reduction(_np.amin, "amin")
any = _counted_reduction(_np.any, "any")
average = _counted_reduction(_np.average, "average", extra_output=True)
count_nonzero = _counted_reduction(_np.count_nonzero, "count_nonzero")
cumulative_prod = _counted_reduction(_np.cumulative_prod, "cumulative_prod")
cumulative_sum = _counted_reduction(_np.cumulative_sum, "cumulative_sum")
median = _counted_reduction(_np.median, "median")
nanargmax = _counted_reduction(_np.nanargmax, "nanargmax")
nanargmin = _counted_reduction(_np.nanargmin, "nanargmin")
nancumprod = _counted_reduction(_np.nancumprod, "nancumprod")
nancumsum = _counted_reduction(_np.nancumsum, "nancumsum")
nanmax = _counted_reduction(_np.nanmax, "nanmax")
nanmean = _counted_reduction(_np.nanmean, "nanmean", extra_output=True)
nanmedian = _counted_reduction(_np.nanmedian, "nanmedian")
nanmin = _counted_reduction(_np.nanmin, "nanmin")
nanpercentile = _counted_reduction(_np.nanpercentile, "nanpercentile")
nanprod = _counted_reduction(_np.nanprod, "nanprod")
nanquantile = _counted_reduction(_np.nanquantile, "nanquantile")
nanstd = _counted_reduction(_np.nanstd, "nanstd", cost_multiplier=2, extra_output=True)
nansum = _counted_reduction(_np.nansum, "nansum")
nanvar = _counted_reduction(_np.nanvar, "nanvar", cost_multiplier=2, extra_output=True)
percentile = _counted_reduction(_np.percentile, "percentile")
quantile = _counted_reduction(_np.quantile, "quantile")

# ptp: numpy 2.0 removed it from ndarray but np.ptp still exists
if hasattr(_np, "ptp"):
    ptp = _counted_reduction(_np.ptp, "ptp")
else:

    def ptp(a, axis=None, **kwargs):
        """Peak-to-peak range. Cost = numel(input) FLOPs."""
        budget = require_budget()
        validate_ndarray(a)
        cost = reduction_cost(a.shape, axis)
        budget.deduct("ptp", flop_cost=cost, subscripts=None, shapes=(a.shape,))
        return _np.max(a, axis=axis, **kwargs) - _np.min(a, axis=axis, **kwargs)

    attach_docstring(ptp, _np.max, "counted_reduction", "numel(input) FLOPs")


# ---------------------------------------------------------------------------
# dot and matmul
# ---------------------------------------------------------------------------


def dot(a, b):
    """Counted version of np.dot."""
    budget = require_budget()
    validate_ndarray(a, b)
    # Extract symmetry info for cost calculation
    operand_symmetries = [
        a.symmetry_info if isinstance(a, SymmetricTensor) else None,
        b.symmetry_info if isinstance(b, SymmetricTensor) else None,
    ]
    has_sym = _builtins.any(s is not None for s in operand_symmetries)
    if a.ndim == 2 and b.ndim == 2:
        cost = einsum_cost(
            "ij,jk->ik",
            shapes=[a.shape, b.shape],
            operand_symmetries=operand_symmetries if has_sym else None,
        )
    elif a.ndim == 1 and b.ndim == 1:
        cost = einsum_cost(
            "i,i->",
            shapes=[a.shape, b.shape],
            operand_symmetries=operand_symmetries if has_sym else None,
        )
    else:
        cost = a.size * b.size
    budget.deduct("dot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.dot(a, b)
    check_nan_inf(result, "dot")
    return result


attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")


def matmul(a, b):
    """Counted version of np.matmul."""
    budget = require_budget()
    validate_ndarray(a, b)
    # Extract symmetry info for cost calculation
    operand_symmetries = [
        a.symmetry_info if isinstance(a, SymmetricTensor) else None,
        b.symmetry_info if isinstance(b, SymmetricTensor) else None,
    ]
    has_sym = _builtins.any(s is not None for s in operand_symmetries)
    if a.ndim == 2 and b.ndim == 2:
        cost = einsum_cost(
            "ij,jk->ik",
            shapes=[a.shape, b.shape],
            operand_symmetries=operand_symmetries if has_sym else None,
        )
    elif a.ndim == 1 and b.ndim == 1:
        cost = einsum_cost(
            "i,i->",
            shapes=[a.shape, b.shape],
            operand_symmetries=operand_symmetries if has_sym else None,
        )
    else:
        cost = a.size * b.size
    budget.deduct("matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.matmul(a, b)
    check_nan_inf(result, "matmul")
    return result


attach_docstring(matmul, _np.matmul, "counted_custom", "depends on operand dimensions")


# ---------------------------------------------------------------------------
# Custom ops (new)
# ---------------------------------------------------------------------------


def inner(a, b):
    """Counted version of np.inner."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    cost = (
        a.size
        if (a.ndim <= 1 and b.ndim <= 1)
        else a.size * (b.shape[-1] if b.ndim > 1 else 1)
    )
    budget.deduct("inner", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.inner(a, b)
    return result


attach_docstring(inner, _np.inner, "counted_custom", "product of matching dims")


def outer(a, b):
    """Counted version of np.outer."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    cost = a.size * b.size
    budget.deduct("outer", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    return _np.outer(a, b)


attach_docstring(outer, _np.outer, "counted_custom", "m * n FLOPs")


def tensordot(a, b, axes=2):
    """Counted version of np.tensordot."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    result = _np.tensordot(a, b, axes=axes)
    if isinstance(axes, int):
        contracted = 1
        for i in range(axes):
            contracted *= a.shape[a.ndim - axes + i]
        cost = _builtins.max(result.size * contracted, 1)
    else:
        contracted = 1
        for i in axes[0]:
            contracted *= a.shape[i]
        cost = _builtins.max(result.size * contracted, 1)
    budget.deduct(
        "tensordot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    )
    return result


attach_docstring(tensordot, _np.tensordot, "counted_custom", "product of all dims")


def vdot(a, b):
    """Counted version of np.vdot."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    cost = a.size
    budget.deduct("vdot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    return _np.vdot(a, b)


attach_docstring(vdot, _np.vdot, "counted_custom", "size of input FLOPs")


def kron(a, b):
    """Counted version of np.kron."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    result = _np.kron(a, b)
    budget.deduct(
        "kron", flop_cost=result.size, subscripts=None, shapes=(a.shape, b.shape)
    )
    return result


attach_docstring(kron, _np.kron, "counted_custom", "output size FLOPs")


def cross(a, b, **kwargs):
    """Counted version of np.cross."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    result = _np.cross(a, b, **kwargs)
    r = _np.asarray(result)
    budget.deduct(
        "cross",
        flop_cost=_builtins.max(r.size * 3, 1),
        subscripts=None,
        shapes=(a.shape, b.shape),
    )
    return result


attach_docstring(cross, _np.cross, "counted_custom", "output_size * 3 FLOPs")


def diff(a, n=1, axis=-1, **kwargs):
    """Counted version of np.diff."""
    budget = require_budget()
    validate_ndarray(a)
    result = _np.diff(a, n=n, axis=axis, **kwargs)
    budget.deduct(
        "diff",
        flop_cost=_builtins.max(result.size, 1),
        subscripts=None,
        shapes=(a.shape,),
    )
    return result


attach_docstring(diff, _np.diff, "counted_custom", "numel(output) FLOPs")


def gradient(f, *varargs, **kwargs):
    """Counted version of np.gradient."""
    budget = require_budget()
    validate_ndarray(f)
    budget.deduct("gradient", flop_cost=f.size, subscripts=None, shapes=(f.shape,))
    return _np.gradient(f, *varargs, **kwargs)


attach_docstring(gradient, _np.gradient, "counted_custom", "numel(input) FLOPs")


def ediff1d(ary, **kwargs):
    """Counted version of np.ediff1d."""
    budget = require_budget()
    if not isinstance(ary, _np.ndarray):
        ary = _np.asarray(ary)
    result = _np.ediff1d(ary, **kwargs)
    budget.deduct(
        "ediff1d",
        flop_cost=_builtins.max(result.size, 1),
        subscripts=None,
        shapes=(ary.shape,),
    )
    return result


attach_docstring(ediff1d, _np.ediff1d, "counted_custom", "numel(output) FLOPs")


def convolve(a, v, mode="full"):
    """Counted version of np.convolve."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    result = _np.convolve(a, v, mode=mode)
    budget.deduct(
        "convolve",
        flop_cost=_builtins.max(a.size * v.size, 1),
        subscripts=None,
        shapes=(a.shape, v.shape),
    )
    return result


attach_docstring(convolve, _np.convolve, "counted_custom", "n * m FLOPs")


def correlate(a, v, mode="valid"):
    """Counted version of np.correlate."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    result = _np.correlate(a, v, mode=mode)
    budget.deduct(
        "correlate",
        flop_cost=_builtins.max(a.size * v.size, 1),
        subscripts=None,
        shapes=(a.shape, v.shape),
    )
    return result


attach_docstring(correlate, _np.correlate, "counted_custom", "n * m FLOPs")


def corrcoef(x, y=None, **kwargs):
    """Counted version of np.corrcoef."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = x.size * x.size if y is None else x.size * _np.asarray(y).size
    budget.deduct(
        "corrcoef", flop_cost=_builtins.max(cost, 1), subscripts=None, shapes=(x.shape,)
    )
    return _np.corrcoef(x, y=y, **kwargs)


attach_docstring(corrcoef, _np.corrcoef, "counted_custom", "n^2 FLOPs")


def cov(m, y=None, **kwargs):
    """Counted version of np.cov."""
    budget = require_budget()
    if not isinstance(m, _np.ndarray):
        m = _np.asarray(m)
    cost = m.size * m.size if y is None else m.size * _np.asarray(y).size
    budget.deduct(
        "cov", flop_cost=_builtins.max(cost, 1), subscripts=None, shapes=(m.shape,)
    )
    return _np.cov(m, y=y, **kwargs)


attach_docstring(cov, _np.cov, "counted_custom", "n^2 FLOPs")


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Counted version of np.trapezoid."""
    budget = require_budget()
    if not isinstance(y, _np.ndarray):
        y = _np.asarray(y)
    budget.deduct("trapezoid", flop_cost=y.size, subscripts=None, shapes=(y.shape,))
    return _np.trapezoid(y, x=x, dx=dx, axis=axis)


attach_docstring(trapezoid, _np.trapezoid, "counted_custom", "numel(input) FLOPs")


def trapz(y, x=None, dx=1.0, axis=-1):
    """Counted version of np.trapz (deprecated alias for trapezoid)."""
    budget = require_budget()
    if not isinstance(y, _np.ndarray):
        y = _np.asarray(y)
    budget.deduct("trapz", flop_cost=y.size, subscripts=None, shapes=(y.shape,))
    return _np.trapz(y, x=x, dx=dx, axis=axis)


attach_docstring(trapz, _np.trapz, "counted_custom", "numel(input) FLOPs")


def interp(x, xp, fp, **kwargs):
    """Counted version of np.interp."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    budget.deduct(
        "interp", flop_cost=_builtins.max(x.size, 1), subscripts=None, shapes=(x.shape,)
    )
    return _np.interp(x, xp, fp, **kwargs)


attach_docstring(interp, _np.interp, "counted_custom", "numel(x) FLOPs")


def einsum_path(subscripts, *operands, **kwargs):
    """Free passthrough for np.einsum_path (planning only, 0 FLOPs)."""
    return _np.einsum_path(subscripts, *operands, **kwargs)


attach_docstring(
    einsum_path, _np.einsum_path, "counted_custom", "0 FLOPs (planning only)"
)
