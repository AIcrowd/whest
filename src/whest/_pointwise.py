"""Counted pointwise operations and reductions for whest."""

from __future__ import annotations

import builtins as _builtins
import inspect as _inspect

import numpy as _np

from whest._docstrings import attach_docstring
from whest._flops import _ceil_log2, einsum_cost, pointwise_cost, reduction_cost
from whest._ndarray import _aswhest
from whest._symmetric import (
    SymmetricTensor,
    SymmetryInfo,
    _warn_symmetry_loss,
    intersect_symmetry,
    propagate_symmetry_reduce,
)
from whest._validation import check_nan_inf, require_budget
from whest.errors import UnsupportedFunctionError

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _counted_unary(np_func, op_name: str):
    def wrapper(x):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        sym_info = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        cost = pointwise_cost(x.shape, symmetry_info=sym_info)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = np_func(x)
        check_nan_inf(result, op_name)
        if sym_info is not None:
            result = SymmetricTensor(result, symmetric_axes=sym_info.symmetric_axes)
        if sym_info is None:
            result = _aswhest(result)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    return wrapper


def _counted_unary_multi(np_func, op_name: str):
    """Factory for unary functions that return multiple arrays (e.g., modf, frexp)."""

    def wrapper(x):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        cost = pointwise_cost(x.shape)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = np_func(x)
        if isinstance(result, tuple):
            result = tuple(_aswhest(r) for r in result)
        else:
            result = _aswhest(result)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(input) FLOPs")
    return wrapper


def _counted_binary(np_func, op_name: str):
    def wrapper(x, y):
        budget = require_budget()
        # Preserve original (possibly Python-scalar) values for the actual
        # numpy call so that NEP 50 weak-typing rules apply correctly. We
        # only need ndarray views for shape and symmetry inspection below.
        x_orig, y_orig = x, y
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        x_sym = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        y_sym = y.symmetry_info if isinstance(y, SymmetricTensor) else None
        x_is_scalar = x.ndim == 0
        y_is_scalar = y.ndim == 0

        # Determine output symmetry.
        out_groups: list | None = None
        out_sym_axes: list | None = None
        if x_sym and y_is_scalar:
            out_groups = x_sym.groups if x_sym.groups else None
            out_sym_axes = x_sym.symmetric_axes
        elif y_sym and x_is_scalar:
            out_groups = y_sym.groups if y_sym.groups else None
            out_sym_axes = y_sym.symmetric_axes
        elif x_sym or y_sym:
            x_groups = x_sym.groups if x_sym else []
            y_groups = y_sym.groups if y_sym else []
            out_groups = intersect_symmetry(
                x_groups if x_groups else None,
                y_groups if y_groups else None,
                x.shape,
                y.shape,
                output_shape,
            )
            out_sym_axes = (
                [g.axes for g in out_groups if g.axes is not None]
                if out_groups
                else None
            )

        out_sym_info = (
            SymmetryInfo(symmetric_axes=out_sym_axes, shape=output_shape)
            if out_sym_axes
            else None
        )
        cost = pointwise_cost(output_shape, symmetry_info=out_sym_info)
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            # Call the underlying ufunc with the ORIGINAL inputs so that
            # Python-scalar dtype promotion (NEP 50) and FloatingPointError
            # propagation (np.errstate) work exactly as in plain numpy.
            result = np_func(x_orig, y_orig)
        check_nan_inf(result, op_name)
        if out_sym_axes:
            result = SymmetricTensor(
                result, symmetric_axes=out_sym_axes, perm_groups=out_groups
            )
            # Warn if either input had more symmetry.
            input_axes = set()
            if x_sym:
                input_axes.update(x_sym.symmetric_axes)
            if y_sym:
                input_axes.update(y_sym.symmetric_axes)
            out_set = set(out_sym_axes)
            lost = [g for g in input_axes if g not in out_set]
            if lost:
                _warn_symmetry_loss(
                    lost, f"{op_name} — groups not shared by both operands"
                )
        elif isinstance(result, SymmetricTensor):
            result = _np.asarray(result)
            # Warn about total loss.
            input_groups_list = []
            if x_sym:
                input_groups_list.extend(x_sym.symmetric_axes)
            if y_sym:
                input_groups_list.extend(y_sym.symmetric_axes)
            if input_groups_list:
                _warn_symmetry_loss(
                    input_groups_list,
                    f"{op_name} — no symmetry groups shared by both operands",
                )
        if not isinstance(result, SymmetricTensor):
            result = _aswhest(result)
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
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            result = np_func(x, y)
        if isinstance(result, tuple):
            result = tuple(_aswhest(r) for r in result)
        else:
            result = _aswhest(result)
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
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
        cost = reduction_cost(a.shape, axis, symmetry_info=sym_info) * cost_multiplier
        if extra_output:
            # Pre-compute extra cost from output shape without running numpy yet
            if axis is None:
                extra_cost = 1  # scalar output
            else:
                ax = axis if axis >= 0 else axis + a.ndim
                keepdims = kwargs.get("keepdims", False)
                if keepdims:
                    out_shape = a.shape[:ax] + (1,) + a.shape[ax + 1 :]
                else:
                    out_shape = a.shape[:ax] + a.shape[ax + 1 :]
                extra_cost = pointwise_cost(out_shape)
            cost += extra_cost
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            result = np_func(a, axis=axis, **kwargs)

        # If caller passed out=, honor numpy's contract: the returned object
        # must be the exact same object (identity). Skip WhestArray wrapping
        # and symmetry tagging when out= is supplied — callers relying on
        # out= are opting out of whest's subclass semantics.
        if kwargs.get("out") is not None:
            return kwargs["out"]

        # Propagate symmetry through reduction.
        if sym_info is not None:
            keepdims = kwargs.get("keepdims", False)
            perm_groups = sym_info.groups if sym_info.groups else []
            new_groups = propagate_symmetry_reduce(
                perm_groups, len(a.shape), axis, keepdims=keepdims
            )
            if new_groups is not None:
                result = _np.asarray(result).view(SymmetricTensor)
                result._symmetry_groups = new_groups
                result._symmetric_axes = [
                    g.axes for g in new_groups if g.axes is not None
                ]
                # Warn if any group changed (order decreased or axes changed).
                old_axes_set = {g.axes for g in perm_groups if g.axes is not None}
                new_axes_set = {g.axes for g in new_groups if g.axes is not None}
                if old_axes_set != new_axes_set:
                    lost = [
                        g.axes
                        for g in perm_groups
                        if g.axes is not None and g.axes not in new_axes_set
                    ]
                    if lost:
                        _warn_symmetry_loss(lost, f"{op_name} reduced dims")
            else:
                if isinstance(result, SymmetricTensor):
                    result = _np.asarray(result)
                if perm_groups:
                    _warn_symmetry_loss(
                        [g.axes for g in perm_groups if g.axes is not None],
                        f"{op_name} removed all symmetric dim groups",
                    )
        elif isinstance(result, SymmetricTensor):
            result = _np.asarray(result)
        if not isinstance(result, SymmetricTensor):
            result = _aswhest(result)
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
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
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
angle.__signature__ = _inspect.signature(_np.angle)
arccos = _counted_unary(_np.arccos, "arccos")
arccosh = _counted_unary(_np.arccosh, "arccosh")
arcsin = _counted_unary(_np.arcsin, "arcsin")
arcsinh = _counted_unary(_np.arcsinh, "arcsinh")
arctan = _counted_unary(_np.arctan, "arctan")
arctanh = _counted_unary(_np.arctanh, "arctanh")


def around(a, decimals=0, out=None):
    """Counted version of np.around. Cost = numel(output) FLOPs."""
    budget = require_budget()
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
    cost = pointwise_cost(a.shape, symmetry_info=sym_info)
    with budget.deduct("around", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.around(a, decimals=decimals, out=out)
    check_nan_inf(result, "around")
    if sym_info is not None:
        result = SymmetricTensor(result, symmetric_axes=sym_info.symmetric_axes)
    if (
        a_is_scalar
        and out is None
        and isinstance(result, _np.ndarray)
        and result.ndim == 0
    ):
        return result.item()
    return result


attach_docstring(around, _np.around, "counted_unary", "numel(output) FLOPs")
asin = _counted_unary(_np.asin, "asin")
asinh = _counted_unary(_np.asinh, "asinh")
atan = _counted_unary(_np.atan, "atan")
atanh = _counted_unary(_np.atanh, "atanh")
if hasattr(_np, "bitwise_count"):
    bitwise_count = _counted_unary(_np.bitwise_count, "bitwise_count")
else:

    def bitwise_count(*args, **kwargs):
        raise UnsupportedFunctionError("bitwise_count", min_version="2.1")


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
fix.__signature__ = _inspect.signature(_np.fix)
i0 = _counted_unary(_np.i0, "i0")
imag = _counted_unary(_np.imag, "imag")
imag.__signature__ = _inspect.signature(_np.imag)
invert = _counted_unary(_np.invert, "invert")
iscomplex = _counted_unary(_np.iscomplex, "iscomplex")
iscomplexobj = _counted_unary(_np.iscomplexobj, "iscomplexobj")
isnat = _counted_unary(_np.isnat, "isnat")
isneginf = _counted_unary(_np.isneginf, "isneginf")
isneginf.__signature__ = _inspect.signature(_np.isneginf)
isposinf = _counted_unary(_np.isposinf, "isposinf")
isposinf.__signature__ = _inspect.signature(_np.isposinf)
isreal = _counted_unary(_np.isreal, "isreal")
isrealobj = _counted_unary(_np.isrealobj, "isrealobj")
log1p = _counted_unary(_np.log1p, "log1p")
logical_not = _counted_unary(_np.logical_not, "logical_not")
nan_to_num = _counted_unary(_np.nan_to_num, "nan_to_num")
nan_to_num.__signature__ = _inspect.signature(_np.nan_to_num)
positive = _counted_unary(_np.positive, "positive")
rad2deg = _counted_unary(_np.rad2deg, "rad2deg")
radians = _counted_unary(_np.radians, "radians")
real = _counted_unary(_np.real, "real")
real.__signature__ = _inspect.signature(_np.real)
real_if_close = _counted_unary(_np.real_if_close, "real_if_close")
real_if_close.__signature__ = _inspect.signature(_np.real_if_close)
reciprocal = _counted_unary(_np.reciprocal, "reciprocal")
rint = _counted_unary(_np.rint, "rint")


def round(a, decimals=0, out=None):
    """Counted version of np.round. Cost = numel(output) FLOPs."""
    budget = require_budget()
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
    cost = pointwise_cost(a.shape, symmetry_info=sym_info)
    with budget.deduct("round", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.round(a, decimals=decimals, out=out)
    check_nan_inf(result, "round")
    if sym_info is not None:
        result = SymmetricTensor(result, symmetric_axes=sym_info.symmetric_axes)
    if (
        a_is_scalar
        and out is None
        and isinstance(result, _np.ndarray)
        and result.ndim == 0
    ):
        return result.item()
    return result


attach_docstring(round, _np.round, "counted_unary", "numel(output) FLOPs")
signbit = _counted_unary(_np.signbit, "signbit")
sinc = _counted_unary(_np.sinc, "sinc")
sinh = _counted_unary(_np.sinh, "sinh")


def sort_complex(a):
    """Counted version of np.sort_complex. Cost: n*ceil(log2(n))."""
    import math

    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.size
    log2n = math.ceil(math.log2(n)) if n > 1 else 1
    cost = n * log2n
    with budget.deduct(
        "sort_complex", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.sort_complex(a)
    return result


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
    a_is_scalar = not isinstance(a, _np.ndarray) and _np.ndim(a) == 0
    b_is_scalar = not isinstance(b, _np.ndarray) and _np.ndim(b) == 0
    # Keep Python scalars as-is so NEP 50 type promotion works correctly
    # (converting them to np.asarray before passing would coerce to float64
    # and break float32 vs Python-float comparisons).
    a_arr = a if isinstance(a, _np.ndarray) else _np.asarray(a)
    b_arr = b if isinstance(b, _np.ndarray) else _np.asarray(b)
    output_shape = _np.broadcast_shapes(a_arr.shape, b_arr.shape)
    cost = pointwise_cost(output_shape)
    with budget.deduct(
        "isclose", flop_cost=cost, subscripts=None, shapes=(a_arr.shape, b_arr.shape)
    ):
        result = _np.isclose(a, b, **kwargs)
    if (
        a_is_scalar
        and b_is_scalar
        and isinstance(result, _np.ndarray)
        and result.ndim == 0
    ):
        return result.item()
    return result


attach_docstring(isclose, _np.isclose, "counted_unary", "numel(output) FLOPs")
isclose.__signature__ = _inspect.signature(_np.isclose)


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


if hasattr(_np, "vecdot"):

    def vecdot(a, b, **kwargs):
        """Counted version of np.vecdot.

        Vector dot product along last axis. Each output element is the dot
        product of two vectors of length K (the last axis), costing K FLOPs.
        Total cost = batch_size * K = numel(a) when a and b have the same shape.
        """
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        if not isinstance(b, _np.ndarray):
            b = _np.asarray(b)
        # Cost = output_elements * contracted_axis_size
        # For vecdot, the last axis is contracted.
        contracted = a.shape[-1] if a.ndim > 0 else 1
        out_shape = (
            _np.broadcast_shapes(a.shape[:-1], b.shape[:-1]) if a.ndim > 0 else ()
        )
        cost = (
            _builtins.max(int(_np.prod(out_shape)) * contracted, 1)
            if out_shape
            else contracted
        )
        with budget.deduct(
            "vecdot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
        ):
            result = _np.vecdot(a, b, **kwargs)
        return result

else:

    def vecdot(*args, **kwargs):
        raise UnsupportedFunctionError("vecdot", min_version="2.1")


if hasattr(_np, "matvec"):

    def matvec(a, b, **kwargs):
        """Counted version of np.matvec.

        Matrix-vector product. A is (..., m, n), v is (..., n), result is (..., m).
        Cost = output_size * contracted_axis (A's last axis).
        """
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        if not isinstance(b, _np.ndarray):
            b = _np.asarray(b)
        contracted = a.shape[-1] if a.ndim > 0 else 1
        # output shape: (..., m) where m = a.shape[-2]
        out_m = a.shape[-2] if a.ndim >= 2 else 1
        batch = a.shape[:-2] if a.ndim > 2 else ()
        cost = _builtins.max(
            int(_np.prod(batch)) * out_m * contracted if batch else out_m * contracted,
            1,
        )
        with budget.deduct(
            "matvec", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
        ):
            result = _np.matvec(a, b, **kwargs)
        return result

else:

    def matvec(*args, **kwargs):
        raise UnsupportedFunctionError("matvec", min_version="2.2")


if hasattr(_np, "vecmat"):

    def vecmat(a, b, **kwargs):
        """Counted version of np.vecmat.

        Vector-matrix product. v is (..., n), A is (..., n, m), result is (..., m).
        Cost = output_size * contracted_axis (v's last axis).
        """
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        if not isinstance(b, _np.ndarray):
            b = _np.asarray(b)
        contracted = a.shape[-1] if a.ndim > 0 else 1
        # output shape: (..., m) where m = b.shape[-1]
        out_m = b.shape[-1] if b.ndim >= 2 else 1
        batch = b.shape[:-2] if b.ndim > 2 else ()
        cost = _builtins.max(
            int(_np.prod(batch)) * out_m * contracted if batch else out_m * contracted,
            1,
        )
        with budget.deduct(
            "vecmat", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
        ):
            result = _np.vecmat(a, b, **kwargs)
        return result

else:

    def vecmat(*args, **kwargs):
        raise UnsupportedFunctionError("vecmat", min_version="2.2")


# Multi-output binary ops
divmod = _counted_binary_multi(_np.divmod, "divmod")


# ---------------------------------------------------------------------------
# Special: clip
# ---------------------------------------------------------------------------


def clip(a, *args, out=None, **kwargs):
    """Counted version of np.clip. Cost = numel(input) or unique_elements if symmetric."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    sym_info = a.symmetry_info if isinstance(a, SymmetricTensor) else None
    cost = pointwise_cost(a.shape, symmetry_info=sym_info)
    with budget.deduct("clip", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        # Delegate all argument handling (validation, min/max/a_min/a_max) to numpy
        result = _np.clip(a, *args, out=out, **kwargs)
    if a.dtype.kind in ("f", "c"):
        check_nan_inf(result, "clip")
    if sym_info is not None:
        result = SymmetricTensor(result, symmetric_axes=sym_info.symmetric_axes)
    return result


attach_docstring(clip, _np.clip, "counted_custom", "numel(input) FLOPs")
clip.__signature__ = _inspect.signature(_np.clip)


# ---------------------------------------------------------------------------
# Reductions (original)
# ---------------------------------------------------------------------------

sum = _counted_reduction(_np.sum, "sum")
max = _counted_reduction(_np.max, "max")
min = _counted_reduction(_np.min, "min")
prod = _counted_reduction(_np.prod, "prod")
mean = _counted_reduction(_np.mean, "mean")
std = _counted_reduction(_np.std, "std")
var = _counted_reduction(_np.var, "var")
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
average = _counted_reduction(_np.average, "average")
_count_nonzero_counted = _counted_reduction(_np.count_nonzero, "count_nonzero")


def count_nonzero(a, axis=None, *, keepdims=False):
    """Counted version of ``numpy.count_nonzero``. Cost: numel(input) FLOPs.

    When ``axis is None`` (and not ``keepdims``) the result is always
    coerced to a Python ``int``. This is unconditional because whest's
    ``_counted_reduction`` factory wraps scalar results via ``_aswhest``
    on every numpy version, so without this coercion users would see a
    ``WhestArray`` rather than the plain ``int`` that ``numpy.count_nonzero``
    documents. The coercion also normalizes the numpy 2.3+ change where
    the raw numpy return type became a numpy scalar.
    """
    result = _count_nonzero_counted(a, axis=axis, keepdims=keepdims)
    if axis is None and not keepdims:
        return int(result)
    return result


attach_docstring(
    count_nonzero, _np.count_nonzero, "counted_reduction", "numel(input) FLOPs"
)
if hasattr(_np, "cumulative_prod"):
    cumulative_prod = _counted_reduction(_np.cumulative_prod, "cumulative_prod")
else:

    def cumulative_prod(*args, **kwargs):
        raise UnsupportedFunctionError("cumulative_prod", min_version="2.1")


if hasattr(_np, "cumulative_sum"):
    cumulative_sum = _counted_reduction(_np.cumulative_sum, "cumulative_sum")
else:

    def cumulative_sum(*args, **kwargs):
        raise UnsupportedFunctionError("cumulative_sum", min_version="2.1")


median = _counted_reduction(_np.median, "median")
nanargmax = _counted_reduction(_np.nanargmax, "nanargmax")
nanargmin = _counted_reduction(_np.nanargmin, "nanargmin")
nancumprod = _counted_reduction(_np.nancumprod, "nancumprod")
nancumsum = _counted_reduction(_np.nancumsum, "nancumsum")
nanmax = _counted_reduction(_np.nanmax, "nanmax")
nanmean = _counted_reduction(_np.nanmean, "nanmean")
nanmedian = _counted_reduction(_np.nanmedian, "nanmedian")
nanmin = _counted_reduction(_np.nanmin, "nanmin")
nanpercentile = _counted_reduction(_np.nanpercentile, "nanpercentile")
nanprod = _counted_reduction(_np.nanprod, "nanprod")
nanquantile = _counted_reduction(_np.nanquantile, "nanquantile")
nanstd = _counted_reduction(_np.nanstd, "nanstd")
nansum = _counted_reduction(_np.nansum, "nansum")
nanvar = _counted_reduction(_np.nanvar, "nanvar")
percentile = _counted_reduction(_np.percentile, "percentile")
quantile = _counted_reduction(_np.quantile, "quantile")

# ptp: numpy 2.0 removed it from ndarray but np.ptp still exists
if hasattr(_np, "ptp"):
    ptp = _counted_reduction(_np.ptp, "ptp")
else:

    def ptp(a, axis=None, **kwargs):
        """Peak-to-peak range. Cost = numel(input) FLOPs."""
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        cost = reduction_cost(a.shape, axis)
        with budget.deduct("ptp", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            result = _np.max(a, axis=axis, **kwargs) - _np.min(a, axis=axis, **kwargs)
        return result

    attach_docstring(ptp, _np.max, "counted_reduction", "numel(input) FLOPs")


# ---------------------------------------------------------------------------
# dot and matmul
# ---------------------------------------------------------------------------


def dot(a, b):
    """Counted version of np.dot."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
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
    with budget.deduct(
        "dot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.dot(a, b)
    check_nan_inf(result, "dot")
    return result


attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")


def matmul(a, b):
    """Counted version of np.matmul."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
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
    with budget.deduct(
        "matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
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
    with budget.deduct(
        "inner", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.inner(a, b)
    return result


attach_docstring(inner, _np.inner, "counted_custom", "product of matching dims")


def outer(a, b, out=None):
    """Counted version of np.outer."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    cost = a.size * b.size
    with budget.deduct(
        "outer", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.outer(a, b, out=out)
    return result


attach_docstring(outer, _np.outer, "counted_custom", "m * n FLOPs")


def tensordot(a, b, axes=2):
    """Counted version of np.tensordot."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    if isinstance(axes, int):
        contracted = 1
        for i in range(axes):
            contracted *= a.shape[a.ndim - axes + i]
    else:
        ax0 = axes[0]
        contracted = 1
        if isinstance(ax0, int):
            # axes=(scalar, scalar) form — single axis contracted
            contracted *= a.shape[ax0] if ax0 < a.ndim else 1
        else:
            for i in ax0:
                contracted *= a.shape[i]
    # output_size * contracted = (a.size / contracted) * (b.size / contracted) * contracted
    # = a.size * b.size / contracted
    cost = _builtins.max(a.size * b.size // contracted, 1) if contracted > 0 else 1
    with budget.deduct(
        "tensordot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.tensordot(a, b, axes=axes)
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
    with budget.deduct(
        "vdot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.vdot(a, b)
    return result


attach_docstring(vdot, _np.vdot, "counted_custom", "size of input FLOPs")


def kron(a, b):
    """Counted version of np.kron."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # kron output size = a.size * b.size
    cost = _builtins.max(a.size * b.size, 1)
    with budget.deduct(
        "kron", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.kron(a, b)
    return result


attach_docstring(kron, _np.kron, "counted_custom", "output size FLOPs")


def cross(a, b, **kwargs):
    """Counted version of np.cross."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # np.cross supports axisa/axisb/axisc kwargs that change output shape,
    # so we compute the result first, then deduct based on actual output size.
    result = _np.cross(a, b, **kwargs)
    cost = _builtins.max(_np.asarray(result).size * 3, 1)
    with budget.deduct(
        "cross",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, b.shape),
    ):
        pass  # numpy call already done; timer records near-zero duration
    return result


attach_docstring(cross, _np.cross, "counted_custom", "output_size * 3 FLOPs")
cross.__signature__ = _inspect.signature(_np.cross)


def diff(a, n=1, axis=-1, **kwargs):
    """Counted version of np.diff."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # Pre-compute output size: along the diff axis, size decreases by n
    ax = axis if axis >= 0 else axis + a.ndim
    out_axis_len = a.shape[ax] - n
    cost = _builtins.max(
        int(_np.prod(a.shape[:ax])) * out_axis_len * int(_np.prod(a.shape[ax + 1 :])), 1
    )
    with budget.deduct(
        "diff",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape,),
    ):
        result = _np.diff(a, n=n, axis=axis, **kwargs)
    return result


attach_docstring(diff, _np.diff, "counted_custom", "numel(output) FLOPs")
diff.__signature__ = _inspect.signature(_np.diff)


def gradient(f, *varargs, **kwargs):
    """Counted version of np.gradient."""
    budget = require_budget()
    if not isinstance(f, _np.ndarray):
        f = _np.asarray(f)
    with budget.deduct(
        "gradient", flop_cost=f.size, subscripts=None, shapes=(f.shape,)
    ):
        result = _np.gradient(f, *varargs, **kwargs)
    return result


attach_docstring(gradient, _np.gradient, "counted_custom", "numel(input) FLOPs")
gradient.__signature__ = _inspect.signature(_np.gradient)


def ediff1d(ary, **kwargs):
    """Counted version of np.ediff1d."""
    budget = require_budget()
    if not isinstance(ary, _np.ndarray):
        ary = _np.asarray(ary)
    # Output size = ary.size - 1 (plus any to_begin/to_end extras)
    to_begin = kwargs.get("to_begin", None)
    to_end = kwargs.get("to_end", None)
    extra = 0
    if to_begin is not None:
        extra += _np.asarray(to_begin).size
    if to_end is not None:
        extra += _np.asarray(to_end).size
    cost = _builtins.max(ary.size - 1 + extra, 1)
    with budget.deduct(
        "ediff1d",
        flop_cost=cost,
        subscripts=None,
        shapes=(ary.shape,),
    ):
        result = _np.ediff1d(ary, **kwargs)
    return result


attach_docstring(ediff1d, _np.ediff1d, "counted_custom", "numel(output) FLOPs")
ediff1d.__signature__ = _inspect.signature(_np.ediff1d)


def convolve(a, v, mode="full"):
    """Counted version of np.convolve."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    cost = _builtins.max(a.size * v.size, 1)
    with budget.deduct(
        "convolve",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v.shape),
    ):
        result = _np.convolve(a, v, mode=mode)
    return result


attach_docstring(convolve, _np.convolve, "counted_custom", "n * m FLOPs")


def correlate(a, v, mode="valid"):
    """Counted version of np.correlate."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(v, _np.ndarray):
        v = _np.asarray(v)
    cost = _builtins.max(a.size * v.size, 1)
    with budget.deduct(
        "correlate",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v.shape),
    ):
        result = _np.correlate(a, v, mode=mode)
    return result


attach_docstring(correlate, _np.correlate, "counted_custom", "n * m FLOPs")


def _cov_cost(x, y=None):
    """Cost for corrcoef/cov: 2 * f^2 * s.

    For a (f, s) input: f features, s samples.
    Covariance requires f^2 dot products of length s, plus mean subtraction.
    """
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if x.ndim == 1:
        f, s = 1, x.shape[0]
    else:
        f, s = x.shape[0], x.shape[1]
    if y is not None:
        y_arr = _np.asarray(y)
        f2 = 1 if y_arr.ndim == 1 else y_arr.shape[0]
        f += f2
    return _builtins.max(2 * f * f * s, 1)


def corrcoef(x, y=None, **kwargs):
    """Counted version of np.corrcoef. Cost: 2 * f^2 * s FLOPs."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = _cov_cost(x, y)
    with budget.deduct("corrcoef", flop_cost=cost, subscripts=None, shapes=(x.shape,)):
        result = _np.corrcoef(x, y=y, **kwargs)
    return result


attach_docstring(corrcoef, _np.corrcoef, "counted_custom", r"$2 f^2 s$ FLOPs")
corrcoef.__signature__ = _inspect.signature(_np.corrcoef)


def cov(m, y=None, **kwargs):
    """Counted version of np.cov. Cost: 2 * f^2 * s FLOPs."""
    budget = require_budget()
    if not isinstance(m, _np.ndarray):
        m = _np.asarray(m)
    cost = _cov_cost(m, y)
    with budget.deduct("cov", flop_cost=cost, subscripts=None, shapes=(m.shape,)):
        result = _np.cov(m, y=y, **kwargs)
    return result


attach_docstring(cov, _np.cov, "counted_custom", r"$2 f^2 s$ FLOPs")
cov.__signature__ = _inspect.signature(_np.cov)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Counted version of np.trapezoid."""
    budget = require_budget()
    if not isinstance(y, _np.ndarray):
        y = _np.asarray(y)
    with budget.deduct(
        "trapezoid", flop_cost=y.size, subscripts=None, shapes=(y.shape,)
    ):
        result = _np.trapezoid(y, x=x, dx=dx, axis=axis)
    return result


attach_docstring(trapezoid, _np.trapezoid, "counted_custom", "numel(input) FLOPs")


if hasattr(_np, "trapz"):

    def trapz(y, x=None, dx=1.0, axis=-1):
        """Counted version of np.trapz (deprecated alias for trapezoid)."""
        budget = require_budget()
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        with budget.deduct(
            "trapz", flop_cost=y.size, subscripts=None, shapes=(y.shape,)
        ):
            result = _np.trapz(y, x=x, dx=dx, axis=axis)
        return result

    attach_docstring(trapz, _np.trapz, "counted_custom", "numel(input) FLOPs")

else:

    def trapz(*args, **kwargs):
        raise UnsupportedFunctionError(
            "trapz", max_version="2.4", replacement="trapezoid"
        )


def interp(x, xp, fp, **kwargs):
    """Counted version of np.interp. Cost: n * ceil(log2(len(xp))) FLOPs."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    xp_arr = _np.asarray(xp)
    n = _builtins.max(x.size, 1)
    xp_len = _builtins.max(xp_arr.size, 1)
    cost = _builtins.max(n * _ceil_log2(xp_len), 1)
    with budget.deduct(
        "interp", flop_cost=cost, subscripts=None, shapes=(x.shape, xp_arr.shape)
    ):
        result = _np.interp(x, xp, fp, **kwargs)
    return result


attach_docstring(interp, _np.interp, "counted_custom", "n * ceil(log2(xp)) FLOPs")
interp.__signature__ = _inspect.signature(_np.interp)
