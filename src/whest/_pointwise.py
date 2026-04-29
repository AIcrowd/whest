"""Counted pointwise operations and reductions for whest."""

from __future__ import annotations

import builtins as _builtins
import inspect as _inspect

import numpy as _np

from whest._docstrings import attach_docstring
from whest._flops import (
    _ceil_log2,
    einsum_cost,
)
from whest._flops import (
    analytical_pointwise_cost as pointwise_cost,
)
from whest._flops import (
    analytical_reduction_cost as reduction_cost,
)
from whest._ndarray import _aswhest, _to_base_ndarray, _to_base_ndarray_tree
from whest._perm_group import SymmetryGroup
from whest._symmetric import (
    SymmetricTensor,
    _warn_symmetry_loss,
)
from whest._symmetric import (
    is_symmetric as _is_symmetric,
)
from whest._symmetry_utils import broadcast_group, intersect_groups, reduce_group
from whest._validation import check_nan_inf, require_budget
from whest.errors import SymmetryError, UnsupportedFunctionError

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _symmetry_of(value):
    return value.symmetry if isinstance(value, SymmetricTensor) else None


def _supports_out_argument(np_func) -> bool:
    if isinstance(np_func, _np.ufunc):
        return True
    try:
        return "out" in _inspect.signature(np_func).parameters
    except (TypeError, ValueError):
        return False


def _prepare_symmetric_out(out, target_symmetry):
    if not isinstance(out, SymmetricTensor):
        return target_symmetry
    carried_symmetry = out.symmetry
    if target_symmetry is None:
        raise ValueError("out symmetry does not match result symmetry")
    if carried_symmetry is not None and carried_symmetry != target_symmetry:
        raise ValueError("out symmetry does not match result symmetry")

    if not _is_symmetric(_np.asarray(out), symmetry=target_symmetry):
        axes = target_symmetry.axes
        if axes is None:
            axes = tuple(range(target_symmetry.degree))
        raise SymmetryError(axes=tuple(axes), max_deviation=float("inf"))
    return target_symmetry


def _validate_result_symmetry(result, symmetry):
    if symmetry is None:
        return
    if not _is_symmetric(_np.asarray(result), symmetry=symmetry):
        axes = symmetry.axes
        if axes is None:
            axes = tuple(range(symmetry.degree))
        raise SymmetryError(axes=tuple(axes), max_deviation=float("inf"))


def _call_with_optional_out(np_func, *args, out=None, supports_out=False, **kwargs):
    # Strip whest subclasses (WhestArray / SymmetricTensor) from arrays so
    # the raw NumPy call does not re-dispatch through ``__array_ufunc__`` /
    # ``__array_function__`` and recurse infinitely. Python scalars and
    # other non-array values pass through unchanged so NEP 50 weak-typing
    # rules continue to apply at the NumPy boundary.
    args = tuple(_to_base_ndarray(a) for a in args)
    # ``where=`` kwarg may be a WhestArray bool mask; strip it. Other
    # array-valued kwargs (e.g. ``axes`` lists for matmul / einsum
    # tensor-axis specs) typically aren't ndarrays, but tree-strip is
    # cheap and safe for nested arg containers.
    for k, v in list(kwargs.items()):
        if isinstance(v, _np.ndarray):
            kwargs[k] = _to_base_ndarray(v)
        elif isinstance(v, (tuple, list)):
            kwargs[k] = _to_base_ndarray_tree(v)
    out_stripped = _to_base_ndarray(out) if out is not None else None
    if out is None:
        return np_func(*args, **kwargs)
    if supports_out:
        return np_func(*args, out=out_stripped, **kwargs)
    result = np_func(*args, **kwargs)
    _np.copyto(out_stripped, _np.asarray(result), casting="unsafe")
    return out


def _wrap_result(result, *, out=None, symmetry=None):
    if out is not None:
        if not isinstance(out, SymmetricTensor):
            _validate_result_symmetry(result, symmetry)
            return out
        effective_symmetry = _prepare_symmetric_out(out, symmetry)
        _validate_result_symmetry(result, effective_symmetry)
        _np.copyto(_np.asarray(out), _np.asarray(result), casting="unsafe")
        return out
    if symmetry is not None:
        return SymmetricTensor(_np.asarray(result), symmetry=symmetry)
    return _aswhest(result)


def _wrap_multi_result(result, *, symmetry=None):
    if isinstance(result, tuple):
        return tuple(_wrap_result(part, symmetry=symmetry) for part in result)
    return _wrap_result(result, symmetry=symmetry)


def _pointwise_symmetry(operands, output_shape):
    aligned_groups = []
    dense_operand_present = False

    for operand, symmetry in operands:
        if operand.ndim == 0:
            continue
        if symmetry is None:
            dense_operand_present = True
            continue
        aligned = broadcast_group(
            symmetry,
            input_shape=operand.shape,
            output_shape=output_shape,
        )
        if aligned is not None:
            aligned_groups.append(aligned)

    if not aligned_groups:
        return None, []
    if dense_operand_present:
        return None, aligned_groups

    output_symmetry = aligned_groups[0]
    for aligned in aligned_groups[1:]:
        output_symmetry = intersect_groups(
            output_symmetry,
            aligned,
            ndim=len(output_shape),
        )
        if output_symmetry is None:
            break
    return output_symmetry, aligned_groups


def _counted_unary(np_func, op_name: str):
    supports_out = _supports_out_argument(np_func)

    def wrapper(x, out=None, **kwargs):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        symmetry = _prepare_symmetric_out(out, symmetry)
        cost = pointwise_cost(x.shape, symmetry=symmetry)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = _call_with_optional_out(
                np_func,
                x,
                out=None if isinstance(out, SymmetricTensor) else out,
                supports_out=supports_out,
                **kwargs,
            )
        check_nan_inf(result, op_name)
        return _wrap_result(result, out=out, symmetry=symmetry)

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


def _counted_unary_multi(np_func, op_name: str):
    """Factory for unary functions that return multiple arrays (e.g., modf, frexp)."""

    def wrapper(x, out=None, **kwargs):
        if out is not None:
            # Multi-output ``out=(out1, out2)`` requires per-buffer strip +
            # identity preservation; not implemented yet. Fail loudly so
            # callers know their ``out=`` was rejected, rather than silently
            # double-charging FLOPs through __array_function__ recursion.
            raise NotImplementedError(
                f"`out=` is not supported by counted multi-output wrappers "
                f"(op: {op_name}); call without `out=` and assign the "
                f"returned tuple instead."
            )
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        symmetry = _symmetry_of(x)
        cost = pointwise_cost(x.shape, symmetry=symmetry)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = np_func(_to_base_ndarray(x), **kwargs)
        return _wrap_multi_result(result, symmetry=symmetry)

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(input) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


def _counted_binary(np_func, op_name: str):
    supports_out = _supports_out_argument(np_func)

    def wrapper(x, y, out=None, **kwargs):
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
        x_sym = _symmetry_of(x)
        y_sym = _symmetry_of(y)
        x_is_scalar = x.ndim == 0
        y_is_scalar = y.ndim == 0
        if x_is_scalar ^ y_is_scalar:
            out_symmetry = y_sym if x_is_scalar else x_sym
            aligned_inputs = [out_symmetry] if out_symmetry is not None else []
        else:
            out_symmetry, aligned_inputs = _pointwise_symmetry(
                ((x, x_sym), (y, y_sym)),
                output_shape,
            )
        out_symmetry = _prepare_symmetric_out(out, out_symmetry)

        cost = pointwise_cost(output_shape, symmetry=out_symmetry)
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            # Call the underlying ufunc with the ORIGINAL inputs so that
            # Python-scalar dtype promotion (NEP 50) and FloatingPointError
            # propagation (np.errstate) work exactly as in plain numpy.
            result = _call_with_optional_out(
                np_func,
                x_orig,
                y_orig,
                out=None if isinstance(out, SymmetricTensor) else out,
                supports_out=supports_out,
                **kwargs,
            )
        check_nan_inf(result, op_name)
        if out_symmetry is not None:
            lost = []
            for group in aligned_inputs:
                if group != out_symmetry and group.axes is not None:
                    lost.append(group.axes)
            if lost:
                _warn_symmetry_loss(
                    list(dict.fromkeys(lost)),
                    f"{op_name} — groups not shared by both operands",
                )
        else:
            lost = [group.axes for group in aligned_inputs if group.axes is not None]
            if lost:
                _warn_symmetry_loss(
                    list(dict.fromkeys(lost)),
                    f"{op_name} — no symmetry groups shared by both operands",
                )
        return _wrap_result(result, out=out, symmetry=out_symmetry)

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


def _counted_binary_multi(np_func, op_name: str):
    """Factory for binary functions that return multiple arrays (e.g., divmod)."""

    def wrapper(x, y, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError(
                f"`out=` is not supported by counted multi-output wrappers "
                f"(op: {op_name}); call without `out=` and assign the "
                f"returned tuple instead."
            )
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        out_symmetry, _ = _pointwise_symmetry(
            ((x, _symmetry_of(x)), (y, _symmetry_of(y))), output_shape
        )
        cost = pointwise_cost(output_shape, symmetry=out_symmetry)
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            result = np_func(_to_base_ndarray(x), _to_base_ndarray(y), **kwargs)
        return _wrap_multi_result(result, symmetry=out_symmetry)

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    try:
        wrapper.__signature__ = _inspect.signature(np_func)
    except (ValueError, TypeError):
        pass
    return wrapper


def _counted_reduction(
    np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False
):
    supports_out = _supports_out_argument(np_func)

    # Per-factory signature introspection for positional `out`.
    # NumPy reductions place `out` at different positional slots;
    # method overrides forwarding through ``*args`` need to find it
    # correctly for each underlying function. ``_axis_is_second_positional``
    # tracks whether `axis` is at slot 1 AND positional-acceptable (true for
    # sum/prod/argmax) or otherwise (false for cumulative_sum where axis is
    # KEYWORD_ONLY, and for percentile/quantile whose slot 1 is `q`).
    try:
        _sig_params = _inspect.signature(np_func).parameters
        _params = list(_sig_params)
    except (ValueError, TypeError):
        _sig_params = {}
        _params = []
    _axis_is_second_positional = (
        len(_params) >= 2
        and _params[1] == "axis"
        and _sig_params["axis"].kind
        in (
            _inspect.Parameter.POSITIONAL_ONLY,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    _args_offset = 2 if _axis_is_second_positional else 1
    _out_args_idx = (
        _params.index("out") - _args_offset
        if "out" in _params and _params.index("out") >= _args_offset
        else None
    )

    def wrapper(a, axis=None, *args, **kwargs):
        budget = require_budget()
        if not isinstance(a, _np.ndarray):
            a = _np.asarray(a)
        symmetry = a.symmetry if isinstance(a, SymmetricTensor) else None
        keepdims = kwargs.get("keepdims", False)

        # Resolve `out` from either kwargs OR a positional slot in args
        # (per-function — see _out_args_idx computed at factory build time).
        args_list = list(args)
        out = kwargs.pop("out", None)
        out_came_from_args = False
        if (
            out is None
            and _out_args_idx is not None
            and 0 <= _out_args_idx < len(args_list)
            and isinstance(args_list[_out_args_idx], _np.ndarray)
        ):
            out = args_list[_out_args_idx]
            out_came_from_args = True

        new_symmetry = (
            reduce_group(symmetry, ndim=len(a.shape), axis=axis, keepdims=keepdims)
            if symmetry is not None
            else None
        )
        _prepare_symmetric_out(out, new_symmetry)
        cost = reduction_cost(a.shape, axis, symmetry=symmetry) * cost_multiplier
        if extra_output:
            # Pre-compute extra cost from output shape without running numpy yet
            if axis is None:
                extra_cost = 1  # scalar output
            else:
                ax = axis if axis >= 0 else axis + a.ndim
                if keepdims:
                    out_shape = a.shape[:ax] + (1,) + a.shape[ax + 1 :]
                else:
                    out_shape = a.shape[:ax] + a.shape[ax + 1 :]
                extra_cost = pointwise_cost(out_shape)
            cost += extra_cost
        out_for_np = None if isinstance(out, SymmetricTensor) else out
        if out_came_from_args:
            # Stripped out goes back into the same positional slot.
            args_list[_out_args_idx] = out_for_np
            np_out_kwarg = None
            np_supports_out_for_call = False
        else:
            np_out_kwarg = out_for_np
            np_supports_out_for_call = supports_out

        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            if _axis_is_second_positional:
                result = _call_with_optional_out(
                    np_func,
                    a,
                    axis,
                    *args_list,
                    out=np_out_kwarg,
                    supports_out=np_supports_out_for_call,
                    **kwargs,
                )
            else:
                # axis is keyword-only or at slot 3+; pass via kwargs.
                result = _call_with_optional_out(
                    np_func,
                    a,
                    *args_list,
                    axis=axis,
                    out=np_out_kwarg,
                    supports_out=np_supports_out_for_call,
                    **kwargs,
                )

        # Propagate symmetry through reduction.
        if out is not None:
            return _wrap_result(result, out=out, symmetry=new_symmetry)

        if symmetry is not None:
            if new_symmetry is not None:
                reduced_axes = (
                    set(range(a.ndim))
                    if axis is None
                    else (
                        {axis % a.ndim}
                        if isinstance(axis, int)
                        else {ax % a.ndim for ax in axis}
                    )
                )
                symmetry_axes = (
                    set(symmetry.axes)
                    if symmetry.axes is not None
                    else set(range(symmetry.degree))
                )
                if reduced_axes & symmetry_axes and new_symmetry != symmetry:
                    if symmetry.axes is not None:
                        _warn_symmetry_loss([symmetry.axes], f"{op_name} reduced dims")
            else:
                if symmetry is not None and symmetry.axes is not None:
                    _warn_symmetry_loss(
                        [symmetry.axes],
                        f"{op_name} removed all symmetric dim groups",
                    )
        return _wrap_result(result, symmetry=new_symmetry)

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
    symmetry = _symmetry_of(a)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    with budget.deduct("around", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _call_with_optional_out(
            _np.around,
            a,
            decimals=decimals,
            out=None if isinstance(out, SymmetricTensor) else out,
            supports_out=True,
        )
    check_nan_inf(result, "around")
    if (
        a_is_scalar
        and out is None
        and _np.ndim(result) == 0
        and hasattr(result, "item")
    ):
        return result.item()
    return _wrap_result(result, out=out, symmetry=symmetry)


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
    symmetry = _symmetry_of(a)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    with budget.deduct("round", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _call_with_optional_out(
            _np.round,
            a,
            decimals=decimals,
            out=None if isinstance(out, SymmetricTensor) else out,
            supports_out=True,
        )
    check_nan_inf(result, "round")
    if (
        a_is_scalar
        and out is None
        and _np.ndim(result) == 0
        and hasattr(result, "item")
    ):
        return result.item()
    return _wrap_result(result, out=out, symmetry=symmetry)


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
        result = _np.sort_complex(_to_base_ndarray(a))
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
    out_symmetry, _ = _pointwise_symmetry(
        ((a_arr, _symmetry_of(a_arr)), (b_arr, _symmetry_of(b_arr))),
        output_shape,
    )
    cost = pointwise_cost(output_shape, symmetry=out_symmetry)
    with budget.deduct(
        "isclose", flop_cost=cost, subscripts=None, shapes=(a_arr.shape, b_arr.shape)
    ):
        result = _np.isclose(_to_base_ndarray(a), _to_base_ndarray(b), **kwargs)
    if a_is_scalar and b_is_scalar and _np.ndim(result) == 0:
        return result
    return _wrap_result(result, symmetry=out_symmetry)


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
            result = _np.vecdot(_to_base_ndarray(a), _to_base_ndarray(b), **kwargs)
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
            result = _np.matvec(_to_base_ndarray(a), _to_base_ndarray(b), **kwargs)
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
            result = _np.vecmat(_to_base_ndarray(a), _to_base_ndarray(b), **kwargs)
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
    operand_arrays = [(a, _symmetry_of(a))]
    for value in args:
        if value is None:
            continue
        arr = value if isinstance(value, _np.ndarray) else _np.asarray(value)
        operand_arrays.append((arr, _symmetry_of(arr)))
    for key in ("a_min", "a_max", "min", "max"):
        value = kwargs.get(key)
        if value is None:
            continue
        arr = value if isinstance(value, _np.ndarray) else _np.asarray(value)
        operand_arrays.append((arr, _symmetry_of(arr)))
    symmetry, _ = _pointwise_symmetry(operand_arrays, a.shape)
    _prepare_symmetric_out(out, symmetry)
    cost = pointwise_cost(a.shape, symmetry=symmetry)
    with budget.deduct("clip", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        # Delegate all argument handling (validation, min/max/a_min/a_max) to numpy
        result = _call_with_optional_out(
            _np.clip,
            a,
            *args,
            out=None if isinstance(out, SymmetricTensor) else out,
            supports_out=True,
            **kwargs,
        )
    if a.dtype.kind in ("f", "c"):
        check_nan_inf(result, "clip")
    return _wrap_result(result, out=out, symmetry=symmetry)


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
            stripped = _to_base_ndarray(a)
            result = _np.max(stripped, axis=axis, **kwargs) - _np.min(
                stripped, axis=axis, **kwargs
            )
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
    # Extract exact symmetry groups for cost calculation
    operand_symmetries = [
        a.symmetry if isinstance(a, SymmetricTensor) else None,
        b.symmetry if isinstance(b, SymmetricTensor) else None,
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
    # Track whether either operand was already a whest subclass; if so,
    # preserve the subclass on the result the way ``__array_wrap__`` did
    # pre-Stage-3 when the raw call dispatched through it. With Stage 3
    # we strip subclasses before the raw NumPy call (to avoid recursion),
    # so the wrap must be re-applied explicitly here.
    inputs_were_whest = isinstance(a, _np.ndarray) and (
        type(a) is not _np.ndarray or type(b) is not _np.ndarray
    )
    with budget.deduct(
        "dot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        # Strip whest subclasses so the raw NumPy call does not re-dispatch
        # through __array_ufunc__ (matmul is a ufunc) / __array_function__.
        result = _np.dot(_to_base_ndarray(a), _to_base_ndarray(b))
    check_nan_inf(result, "dot")
    return _aswhest(result) if inputs_were_whest else result


attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")


def matmul(a, b):
    """Counted version of np.matmul."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # Extract exact symmetry groups for cost calculation
    operand_symmetries = [
        a.symmetry if isinstance(a, SymmetricTensor) else None,
        b.symmetry if isinstance(b, SymmetricTensor) else None,
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
    inputs_were_whest = isinstance(a, _np.ndarray) and (
        type(a) is not _np.ndarray or type(b) is not _np.ndarray
    )
    with budget.deduct(
        "matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = _np.matmul(_to_base_ndarray(a), _to_base_ndarray(b))
    check_nan_inf(result, "matmul")
    return _aswhest(result) if inputs_were_whest else result


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
        result = _np.inner(_to_base_ndarray(a), _to_base_ndarray(b))
    return result


attach_docstring(inner, _np.inner, "counted_custom", "product of matching dims")


def outer(a, b, out=None):
    """Counted version of np.outer."""
    budget = require_budget()
    a_orig = a
    b_orig = b
    target_symmetry = SymmetryGroup.symmetric(axes=(0, 1)) if a_orig is b_orig else None
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    if target_symmetry is not None:
        target_symmetry = _prepare_symmetric_out(out, target_symmetry)
    cost = a.size * b.size
    with budget.deduct(
        "outer", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    ):
        result = _np.outer(
            _to_base_ndarray(a),
            _to_base_ndarray(b),
            out=None if isinstance(out, SymmetricTensor) else out,
        )
    if target_symmetry is None:
        if out is not None:
            return out
        return result
    return _wrap_result(result, out=out, symmetry=target_symmetry)


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
        result = _np.tensordot(_to_base_ndarray(a), _to_base_ndarray(b), axes=axes)
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
        result = _np.vdot(_to_base_ndarray(a), _to_base_ndarray(b))
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
        result = _np.kron(_to_base_ndarray(a), _to_base_ndarray(b))
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
    result = _np.cross(_to_base_ndarray(a), _to_base_ndarray(b), **kwargs)
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
        result = _np.diff(_to_base_ndarray(a), n=n, axis=axis, **kwargs)
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
        result = _np.gradient(
            _to_base_ndarray(f),
            *[_to_base_ndarray(v) for v in varargs],
            **kwargs,
        )
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
        # ``to_begin`` / ``to_end`` kwargs may be WhestArrays — strip via tree.
        stripped_kwargs = {
            k: _to_base_ndarray(v) if isinstance(v, _np.ndarray) else v
            for k, v in kwargs.items()
        }
        result = _np.ediff1d(_to_base_ndarray(ary), **stripped_kwargs)
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
        result = _np.convolve(_to_base_ndarray(a), _to_base_ndarray(v), mode=mode)
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
        result = _np.correlate(_to_base_ndarray(a), _to_base_ndarray(v), mode=mode)
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
        result = _np.corrcoef(
            _to_base_ndarray(x),
            y=_to_base_ndarray(y) if y is not None else None,
            **kwargs,
        )
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
        result = _np.cov(
            _to_base_ndarray(m),
            y=_to_base_ndarray(y) if y is not None else None,
            **kwargs,
        )
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
        result = _np.trapezoid(
            _to_base_ndarray(y),
            x=_to_base_ndarray(x) if x is not None else None,
            dx=dx,
            axis=axis,
        )
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
            result = _np.trapz(
                _to_base_ndarray(y),
                x=_to_base_ndarray(x) if x is not None else None,
                dx=dx,
                axis=axis,
            )
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
        result = _np.interp(
            _to_base_ndarray(x),
            _to_base_ndarray(xp),
            _to_base_ndarray(fp),
            **kwargs,
        )
    return result


attach_docstring(interp, _np.interp, "counted_custom", "n * ceil(log2(xp)) FLOPs")
interp.__signature__ = _inspect.signature(_np.interp)
