"""Zero-FLOP wrappers around NumPy tensor creation and manipulation.

Every function in this module delegates directly to the corresponding
NumPy function and costs **0 FLOPs**, so they work both inside and
outside a :class:`~whest._budget.BudgetContext`.
"""

from __future__ import annotations

import inspect as _inspect

import numpy as _np

from whest._docstrings import attach_docstring
from whest._perm_group import SymmetryGroup
from whest._symmetric import SymmetricTensor
from whest._symmetry_utils import (
    broadcast_group,
    remap_group_axes,
    validate_symmetry_group,
    wrap_with_symmetry,
)
from whest._validation import require_budget
from whest.errors import SymmetryError, UnsupportedFunctionError


def _symmetric_2d(result):
    """Wrap a 2D square result as SymmetricTensor with axes (0,1)."""
    if result.ndim == 2 and result.shape[0] == result.shape[1]:
        return wrap_with_symmetry(result, SymmetryGroup.symmetric(axes=(0, 1)))
    return result


def _compatible_symmetry_for_shape(symmetry, shape):
    """Return ``symmetry`` only when ``shape`` still supports it exactly."""
    if symmetry is None:
        return None
    try:
        validate_symmetry_group(symmetry, ndim=len(shape), shape=shape)
    except (SymmetryError, ValueError):
        return None
    return symmetry


def _normalize_axis_order(axes, ndim):
    return tuple(axis % ndim for axis in axes)


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------


def array(object, dtype=None, **kwargs):
    """Create an array. Cost: numel(output)."""
    budget = require_budget()
    # Pre-compute cost from input to keep numpy call inside the timer
    _probe = _np.asarray(object)
    cost = max(_probe.size, 1)
    with budget.deduct(
        "array", flop_cost=cost, subscripts=None, shapes=(_probe.shape,)
    ):
        result = _np.array(object, dtype=dtype, **kwargs)
    return result


attach_docstring(array, _np.array, "counted_custom", "numel(input) FLOPs")


def zeros(shape, dtype=float, **kwargs):
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    return _symmetric_2d(_np.zeros(shape, dtype=dtype, **kwargs))


attach_docstring(zeros, _np.zeros, "free", "0 FLOPs")


def ones(shape, dtype=float, **kwargs):
    """Return array of ones. Wraps ``numpy.ones``. Cost: 0 FLOPs."""
    return _symmetric_2d(_np.ones(shape, dtype=dtype, **kwargs))


attach_docstring(ones, _np.ones, "free", "0 FLOPs")


def full(shape, fill_value, dtype=None, **kwargs):
    """Return array filled with *fill_value*. Cost: numel(output)."""
    budget = require_budget()
    result = _np.full(shape, fill_value, dtype=dtype, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("full", flop_cost=cost, subscripts=None, shapes=()):
        result = _symmetric_2d(result)
    return result


attach_docstring(full, _np.full, "free", "0 FLOPs")


def eye(N, M=None, k=0, dtype=float, **kwargs):
    """Return identity matrix. Wraps ``numpy.eye``. Cost: 0 FLOPs."""
    result = _np.eye(N, M=M, k=k, dtype=dtype, **kwargs)
    if k == 0 and (M is None or M == N):
        return wrap_with_symmetry(result, SymmetryGroup.symmetric(axes=(0, 1)))
    return result


attach_docstring(eye, _np.eye, "free", "0 FLOPs")


def diag(v, k=0):
    """Extract diagonal or construct diagonal array.

    Cost: numel(output) when constructing (1D→2D), min(m,n) when extracting (2D→1D).
    """
    budget = require_budget()
    v = _np.asarray(v)
    if v.ndim == 1:
        # Constructing diagonal matrix: output is (n+|k|) x (n+|k|)
        n = v.shape[0] + abs(k)
        cost = n * n
    else:
        # Extracting diagonal: reads min(m,n) elements
        m, n = v.shape[0], v.shape[1] if v.ndim > 1 else v.shape[0]
        cost = min(m, n)
    with budget.deduct("diag", flop_cost=cost, subscripts=None, shapes=(v.shape,)):
        result = _np.diag(v, k=k)
    if v.ndim == 1 and k == 0:
        return wrap_with_symmetry(result, SymmetryGroup.symmetric(axes=(0, 1)))
    return result


attach_docstring(diag, _np.diag, "free", "0 FLOPs")


def arange(*args, **kwargs):
    """Return evenly spaced values. Cost: numel(output)."""
    budget = require_budget()
    # cost depends on result; duration is post-hoc
    # arange output size depends on start/stop/step parsing which is non-trivial
    result = _np.arange(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("arange", flop_cost=cost, subscripts=None, shapes=()):
        pass
    return result


attach_docstring(arange, _np.arange, "counted_custom", "numel(output) FLOPs")


def linspace(start, stop, num=50, **kwargs):
    """Return evenly spaced numbers. Cost: numel(output)."""
    budget = require_budget()
    cost = max(int(num), 1)
    with budget.deduct("linspace", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.linspace(start, stop, num=num, **kwargs)
    return result


attach_docstring(linspace, _np.linspace, "counted_custom", "numel(output) FLOPs")


def zeros_like(a, dtype=None, **kwargs):
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    result = _np.zeros_like(a, dtype=dtype, **kwargs)
    if isinstance(a, SymmetricTensor):
        return wrap_with_symmetry(
            result,
            _compatible_symmetry_for_shape(a.symmetry, result.shape),
        )
    return result


attach_docstring(zeros_like, _np.zeros_like, "free", "0 FLOPs")


def ones_like(a, dtype=None, **kwargs):
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: 0 FLOPs."""
    result = _np.ones_like(a, dtype=dtype, **kwargs)
    if isinstance(a, SymmetricTensor):
        return wrap_with_symmetry(
            result,
            _compatible_symmetry_for_shape(a.symmetry, result.shape),
        )
    return result


attach_docstring(ones_like, _np.ones_like, "free", "0 FLOPs")


def full_like(a, fill_value, dtype=None, **kwargs):
    """Return full array with same shape. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    with budget.deduct("full_like", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.full_like(a, fill_value, dtype=dtype, **kwargs)
    if isinstance(a, SymmetricTensor):
        return wrap_with_symmetry(
            result,
            _compatible_symmetry_for_shape(a.symmetry, result.shape),
        )
    return result


attach_docstring(full_like, _np.full_like, "free", "0 FLOPs")


def empty(shape, dtype=float, **kwargs):
    """Return uninitialized array. Wraps ``numpy.empty``. Cost: 0 FLOPs."""
    return _np.empty(shape, dtype=dtype, **kwargs)


attach_docstring(empty, _np.empty, "free", "0 FLOPs")


def empty_like(a, dtype=None, **kwargs):
    """Return uninitialized array with same shape. Wraps ``numpy.empty_like``. Cost: 0 FLOPs."""
    return _np.empty_like(a, dtype=dtype, **kwargs)


attach_docstring(empty_like, _np.empty_like, "free", "0 FLOPs")


def identity(n, dtype=float):
    """Return identity matrix. Wraps ``numpy.identity``. Cost: 0 FLOPs."""
    result = _np.identity(n, dtype=dtype)
    return wrap_with_symmetry(result, SymmetryGroup.symmetric(axes=(0, 1)))


attach_docstring(identity, _np.identity, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Tensor manipulation
# ---------------------------------------------------------------------------


def reshape(a, /, *args, **kwargs):
    """Reshape an array. Wraps ``numpy.reshape``. Cost: 0 FLOPs."""
    return _np.reshape(_np.asarray(a), *args, **kwargs)


attach_docstring(reshape, _np.reshape, "free", "0 FLOPs")


def transpose(a, axes=None):
    """Permute array dimensions. Wraps ``numpy.transpose``. Cost: 0 FLOPs."""
    result = _np.transpose(_np.asarray(a), axes=axes)
    if not isinstance(a, SymmetricTensor):
        return result
    if axes is None:
        order = tuple(reversed(range(a.ndim)))
    else:
        order = _normalize_axis_order(tuple(axes), a.ndim)
    mapping = {old: new for new, old in enumerate(order)}
    return wrap_with_symmetry(result, remap_group_axes(a.symmetry, mapping))


attach_docstring(transpose, _np.transpose, "free", "0 FLOPs")


def swapaxes(a, axis1, axis2):
    """Swap two axes. Wraps ``numpy.swapaxes``. Cost: 0 FLOPs."""
    result = _np.swapaxes(_np.asarray(a), axis1, axis2)
    if not isinstance(a, SymmetricTensor):
        return result
    order = list(range(a.ndim))
    axis1 %= a.ndim
    axis2 %= a.ndim
    order[axis1], order[axis2] = order[axis2], order[axis1]
    mapping = {old: new for new, old in enumerate(order)}
    return wrap_with_symmetry(result, remap_group_axes(a.symmetry, mapping))


attach_docstring(swapaxes, _np.swapaxes, "free", "0 FLOPs")


def moveaxis(a, source, destination):
    """Move axes to new positions. Wraps ``numpy.moveaxis``. Cost: 0 FLOPs."""
    result = _np.moveaxis(_np.asarray(a), source, destination)
    if not isinstance(a, SymmetricTensor):
        return result
    if _np.ndim(source) == 0:
        source_axes = (int(source),)
    else:
        source_axes = tuple(source)
    if _np.ndim(destination) == 0:
        destination_axes = (int(destination),)
    else:
        destination_axes = tuple(destination)
    source_axes = _normalize_axis_order(source_axes, a.ndim)
    destination_axes = _normalize_axis_order(destination_axes, a.ndim)
    order = [axis for axis in range(a.ndim) if axis not in source_axes]
    for dest, src in sorted(zip(destination_axes, source_axes)):
        order.insert(dest, src)
    mapping = {old: new for new, old in enumerate(order)}
    return wrap_with_symmetry(result, remap_group_axes(a.symmetry, mapping))


attach_docstring(moveaxis, _np.moveaxis, "free", "0 FLOPs")


def concatenate(arrays, axis=0, **kwargs):
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    cost = max(sum(_np.asarray(a).size for a in arrays), 1)
    with budget.deduct("concatenate", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.concatenate(arrays, axis=axis, **kwargs)
    return result


attach_docstring(concatenate, _np.concatenate, "counted_custom", "numel(output) FLOPs")


def stack(arrays, axis=0, **kwargs):
    """Stack arrays along a new axis. Cost: numel(output)."""
    budget = require_budget()
    cost = max(sum(_np.asarray(a).size for a in arrays), 1)
    with budget.deduct("stack", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.stack(arrays, axis=axis, **kwargs)
    return result


attach_docstring(stack, _np.stack, "free", "0 FLOPs")


def vstack(tup):
    """Stack arrays vertically. Cost: numel(output)."""
    budget = require_budget()
    cost = max(sum(_np.asarray(a).size for a in tup), 1)
    with budget.deduct("vstack", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.vstack(tup)
    return result


attach_docstring(vstack, _np.vstack, "free", "0 FLOPs")


def hstack(tup):
    """Stack arrays horizontally. Wraps ``numpy.hstack``. Cost: 0 FLOPs."""
    return _np.hstack(tup)


attach_docstring(hstack, _np.hstack, "free", "0 FLOPs")


def split(ary, indices_or_sections, axis=0):
    """Split array. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    with budget.deduct(
        "split", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        result = _np.split(ary, indices_or_sections, axis=axis)
    return result


attach_docstring(split, _np.split, "free", "0 FLOPs")


def hsplit(ary, indices_or_sections):
    """Split array horizontally. Wraps ``numpy.hsplit``. Cost: 0 FLOPs."""
    return _np.hsplit(ary, indices_or_sections)


attach_docstring(hsplit, _np.hsplit, "free", "0 FLOPs")


def vsplit(ary, indices_or_sections):
    """Split array vertically. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    with budget.deduct(
        "vsplit", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        result = _np.vsplit(ary, indices_or_sections)
    return result


attach_docstring(vsplit, _np.vsplit, "free", "0 FLOPs")


def squeeze(a, axis=None):
    """Remove length-1 axes. Wraps ``numpy.squeeze``. Cost: 0 FLOPs."""
    return _np.squeeze(a, axis=axis)


attach_docstring(squeeze, _np.squeeze, "free", "0 FLOPs")


def expand_dims(a, axis):
    """Insert a new axis. Wraps ``numpy.expand_dims``. Cost: 0 FLOPs."""
    return _np.expand_dims(a, axis=axis)


attach_docstring(expand_dims, _np.expand_dims, "free", "0 FLOPs")


def ravel(a, **kwargs):
    """Flatten array. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    with budget.deduct("ravel", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)):
        result = _np.ravel(a_arr, **kwargs)
    return result


attach_docstring(ravel, _np.ravel, "free", "0 FLOPs")


def copy(a, **kwargs):
    """Return copy of array. Wraps ``numpy.copy``. Cost: 0 FLOPs."""
    result = _np.copy(_np.asarray(a), **kwargs)
    if isinstance(a, SymmetricTensor):
        return wrap_with_symmetry(result, a.symmetry)
    return result


attach_docstring(copy, _np.copy, "free", "0 FLOPs")


def where(condition, x=None, y=None):
    """Return elements chosen from *x* or *y*. Cost: numel(input)."""
    budget = require_budget()
    cond_arr = _np.asarray(condition)
    cost = cond_arr.size
    with budget.deduct(
        "where", flop_cost=cost, subscripts=None, shapes=(cond_arr.shape,)
    ):
        if x is None and y is None:
            result = _np.where(condition)
        else:
            result = _np.where(condition, x, y)
    return result


attach_docstring(where, _np.where, "free", "0 FLOPs")


def tile(A, reps):
    """Construct array by repeating. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(A)
    reps_tup = (reps,) if _np.ndim(reps) == 0 else tuple(reps)
    # Output size = input size * product of reps
    cost = max(a_arr.size * int(_np.prod(reps_tup)), 1)
    with budget.deduct("tile", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.tile(A, reps)
    return result


attach_docstring(tile, _np.tile, "free", "0 FLOPs")


def repeat(a, repeats, axis=None):
    """Repeat elements. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    # Output size: each element repeated; total = sum of repeats or size * repeats
    reps = _np.asarray(repeats)
    if reps.ndim == 0:
        cost = max(a_arr.size * int(reps), 1)
    else:
        cost = max(int(reps.sum()), 1)
    with budget.deduct("repeat", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.repeat(a, repeats, axis=axis)
    return result


attach_docstring(repeat, _np.repeat, "free", "0 FLOPs")


def flip(m, axis=None):
    """Reverse order of elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    return _np.flip(m, axis=axis)


attach_docstring(flip, _np.flip, "free", "0 FLOPs")


def roll(a, shift, axis=None):
    """Roll array elements. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = max(a_arr.size, 1)
    with budget.deduct("roll", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.roll(a, shift, axis=axis)
    return result


attach_docstring(roll, _np.roll, "free", "0 FLOPs")


def pad(array, pad_width, **kwargs):
    """Pad an array. Cost: numel(output)."""
    budget = require_budget()
    # cost depends on result; duration is post-hoc
    # pad_width parsing is complex (scalar, per-axis, per-side) — not worth replicating
    result = _np.pad(array, pad_width, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("pad", flop_cost=cost, subscripts=None, shapes=()):
        pass
    return result


attach_docstring(pad, _np.pad, "free", "0 FLOPs")


def triu(m, k=0):
    """Upper triangle. Wraps ``numpy.triu``. Cost: 0 FLOPs."""
    return _np.triu(m, k=k)


attach_docstring(triu, _np.triu, "free", "0 FLOPs")


def tril(m, k=0):
    """Lower triangle. Wraps ``numpy.tril``. Cost: 0 FLOPs."""
    return _np.tril(m, k=k)


attach_docstring(tril, _np.tril, "free", "0 FLOPs")


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return diagonal. Cost: numel(output)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    # Diagonal length along axis1/axis2
    m, n = a_arr.shape[axis1], a_arr.shape[axis2]
    if offset >= 0:
        diag_len = max(min(m, n - offset), 0)
    else:
        diag_len = max(min(m + offset, n), 0)
    cost = max(diag_len, 1)
    with budget.deduct(
        "diagonal", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    return result


attach_docstring(diagonal, _np.diagonal, "free", "0 FLOPs")


def broadcast_to(array, shape):
    """Broadcast array to shape. Cost: numel(output)."""
    output_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    input_array = _np.asarray(array)
    budget = require_budget()
    cost = max(int(_np.prod(output_shape)), 1)
    with budget.deduct("broadcast_to", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.broadcast_to(input_array, output_shape)
    symmetry = broadcast_group(
        array.symmetry if isinstance(array, SymmetricTensor) else None,
        input_shape=input_array.shape,
        output_shape=output_shape,
    )
    return wrap_with_symmetry(result, symmetry)


attach_docstring(broadcast_to, _np.broadcast_to, "free", "0 FLOPs")


def meshgrid(*xi, **kwargs):
    """Return coordinate matrices. Cost: numel(output)."""
    budget = require_budget()
    # Each output grid has shape = product of all input lengths; there are len(xi) grids
    sizes = [_np.asarray(x).size for x in xi]
    grid_size = int(_np.prod(sizes)) if sizes else 0
    cost = max(grid_size * len(sizes), 1)
    with budget.deduct("meshgrid", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.meshgrid(*xi, **kwargs)
    return result


attach_docstring(meshgrid, _np.meshgrid, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Type / info helpers
# ---------------------------------------------------------------------------


def astype(x, dtype, /, *, copy=True, device=None):
    """Cast array to *dtype*. Wraps ``np.astype(x, dtype)``. Cost: 0 FLOPs."""
    return _np.astype(_np.asarray(x), dtype, copy=copy, device=device)


def asarray(a, dtype=None, **kwargs):
    """Convert to array. Cost: numel(output)."""
    budget = require_budget()
    # Pre-compute cost; asarray on an already-array is a no-op
    _probe = _np.asarray(a)
    cost = max(_probe.size, 1)
    with budget.deduct(
        "asarray", flop_cost=cost, subscripts=None, shapes=(_probe.shape,)
    ):
        result = _np.asarray(a, dtype=dtype, **kwargs)
    return result


attach_docstring(asarray, _np.asarray, "free", "0 FLOPs")


def isnan(x, **kwargs):
    """Test element-wise for NaN. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct("isnan", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)):
        result = _np.isnan(x, **kwargs)
    return result


attach_docstring(isnan, _np.isnan, "free", "0 FLOPs")


def isfinite(x, **kwargs):
    """Test element-wise for finiteness. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct(
        "isfinite", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
    ):
        result = _np.isfinite(x, **kwargs)
    return result


attach_docstring(isfinite, _np.isfinite, "free", "0 FLOPs")


def isinf(x, **kwargs):
    """Test element-wise for Inf. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    with budget.deduct("isinf", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)):
        result = _np.isinf(x, **kwargs)
    return result


attach_docstring(isinf, _np.isinf, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# New free ops
# ---------------------------------------------------------------------------


def append(arr, values, axis=None, **kwargs):
    """Append values. Cost: numel(appended values)."""
    budget = require_budget()
    values_arr = _np.asarray(values)
    cost = values_arr.size  # num appended
    with budget.deduct("append", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.append(arr, values, axis=axis, **kwargs)
    return result


attach_docstring(append, _np.append, "free", "0 FLOPs")


def argwhere(a, *args, **kwargs):
    """Find indices of non-zero elements. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "argwhere", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _np.argwhere(a, *args, **kwargs)
    return result


attach_docstring(argwhere, _np.argwhere, "free", "0 FLOPs")


def array_split(ary, *args, **kwargs):
    """Split array into sub-arrays. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    with budget.deduct(
        "array_split", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        result = _np.array_split(ary, *args, **kwargs)
    return result


attach_docstring(array_split, _np.array_split, "free", "0 FLOPs")


def asarray_chkfinite(a, *args, **kwargs):
    """Convert to array checking for NaN/Inf. Cost: numel(output)."""
    budget = require_budget()
    result = _np.asarray_chkfinite(a, *args, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "asarray_chkfinite", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(asarray_chkfinite, _np.asarray_chkfinite, "free", "0 FLOPs")


def atleast_1d(*args, **kwargs):
    """Convert to 1-D or higher. Wraps ``numpy.atleast_1d``. Cost: 0 FLOPs."""
    return _np.atleast_1d(*args, **kwargs)


attach_docstring(atleast_1d, _np.atleast_1d, "free", "0 FLOPs")


def atleast_2d(*args, **kwargs):
    """Convert to 2-D or higher. Wraps ``numpy.atleast_2d``. Cost: 0 FLOPs."""
    return _np.atleast_2d(*args, **kwargs)


attach_docstring(atleast_2d, _np.atleast_2d, "free", "0 FLOPs")


def atleast_3d(*args, **kwargs):
    """Convert to 3-D or higher. Wraps ``numpy.atleast_3d``. Cost: 0 FLOPs."""
    return _np.atleast_3d(*args, **kwargs)


attach_docstring(atleast_3d, _np.atleast_3d, "free", "0 FLOPs")


def base_repr(*args, **kwargs):
    """Return string representation of number. Cost: numel(output)."""
    budget = require_budget()
    result = _np.base_repr(*args, **kwargs)
    cost = len(result)
    with budget.deduct("base_repr", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(base_repr, _np.base_repr, "free", "0 FLOPs")


def binary_repr(*args, **kwargs):
    """Return binary representation of integer. Cost: numel(output)."""
    budget = require_budget()
    result = _np.binary_repr(*args, **kwargs)
    cost = len(result)
    with budget.deduct("binary_repr", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(binary_repr, _np.binary_repr, "free", "0 FLOPs")


def block(*args, **kwargs):
    """Assemble array from nested lists. Cost: numel(output)."""
    budget = require_budget()
    result = _np.block(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("block", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(block, _np.block, "free", "0 FLOPs")


def bmat(*args, **kwargs):
    """Build matrix from string/nested sequence. Cost: numel(output)."""
    budget = require_budget()
    result = _np.bmat(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("bmat", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(bmat, _np.bmat, "free", "0 FLOPs")


def broadcast_arrays(*args, **kwargs):
    """Broadcast any number of arrays. Cost: numel(output)."""
    arrays = tuple(_np.asarray(arg) for arg in args)
    budget = require_budget()
    result = _np.broadcast_arrays(*arrays, **kwargs)
    cost = sum(a.size for a in result)
    with budget.deduct("broadcast_arrays", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    if not result:
        return result
    output_shape = result[0].shape
    wrapped = []
    for original, array, broadcasted in zip(args, arrays, result):
        symmetry = broadcast_group(
            original.symmetry if isinstance(original, SymmetricTensor) else None,
            input_shape=array.shape,
            output_shape=output_shape,
        )
        wrapped.append(wrap_with_symmetry(broadcasted, symmetry))
    return tuple(wrapped)


attach_docstring(broadcast_arrays, _np.broadcast_arrays, "free", "0 FLOPs")


def broadcast_shapes(*args, **kwargs):
    """Broadcast shapes to a common shape. Wraps ``numpy.broadcast_shapes``. Cost: 0 FLOPs."""
    return _np.broadcast_shapes(*args, **kwargs)


attach_docstring(broadcast_shapes, _np.broadcast_shapes, "free", "0 FLOPs")


def can_cast(*args, **kwargs):
    """Returns True if cast between data types can occur. Wraps ``numpy.can_cast``. Cost: 0 FLOPs."""
    return _np.can_cast(*args, **kwargs)


attach_docstring(can_cast, _np.can_cast, "free", "0 FLOPs")


def choose(*args, **kwargs):
    """Construct array from index array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.choose(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("choose", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(choose, _np.choose, "free", "0 FLOPs")


def column_stack(*args, **kwargs):
    """Stack 1-D arrays as columns. Wraps ``numpy.column_stack``. Cost: 0 FLOPs."""
    return _np.column_stack(*args, **kwargs)


attach_docstring(column_stack, _np.column_stack, "free", "0 FLOPs")


def common_type(*args, **kwargs):
    """Return scalar type common to input arrays. Wraps ``numpy.common_type``. Cost: 0 FLOPs."""
    return _np.common_type(*args, **kwargs)


attach_docstring(common_type, _np.common_type, "free", "0 FLOPs")


def compress(condition, a, *args, **kwargs):
    """Return selected slices along an axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.compress(condition, a, *args, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "compress", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(compress, _np.compress, "free", "0 FLOPs")


def concat(*args, **kwargs):
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.concat(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("concat", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(concat, _np.concat, "free", "0 FLOPs")


def copyto(dst, src, casting="same_kind", where=True):
    """Copies values from one array to another. Cost: num elements copied."""
    budget = require_budget()
    src_arr = _np.asarray(src)
    if where is not True:
        where_arr = _np.asarray(where)
        cost = int(_np.count_nonzero(where_arr))
    else:
        cost = src_arr.size
    with budget.deduct("copyto", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.copyto(dst, src, casting=casting, where=where)
    return result


attach_docstring(copyto, _np.copyto, "free", "0 FLOPs")


def delete(arr, obj, axis=None, **kwargs):
    """Return new array with sub-arrays deleted. Cost: num elements removed."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    result = _np.delete(arr, obj, axis=axis, **kwargs)
    cost = max(arr_np.size - result.size, 0)  # num deleted
    with budget.deduct("delete", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(delete, _np.delete, "free", "0 FLOPs")


def diag_indices(*args, **kwargs):
    """Return indices to access main diagonal. Wraps ``numpy.diag_indices``. Cost: 0 FLOPs."""
    return _np.diag_indices(*args, **kwargs)


attach_docstring(diag_indices, _np.diag_indices, "free", "0 FLOPs")


def diag_indices_from(*args, **kwargs):
    """Return indices to access main diagonal of array. Wraps ``numpy.diag_indices_from``. Cost: 0 FLOPs."""
    return _np.diag_indices_from(*args, **kwargs)


attach_docstring(diag_indices_from, _np.diag_indices_from, "free", "0 FLOPs")


def diagflat(v, k=0):
    """Create diagonal array from flattened input. Cost: numel(output)."""
    budget = require_budget()
    v_arr = _np.asarray(v)
    result = _np.diagflat(v, k=k)
    cost = result.size  # output is (n+|k|)×(n+|k|) matrix
    with budget.deduct(
        "diagflat", flop_cost=cost, subscripts=None, shapes=(v_arr.shape,)
    ):
        pass  # numpy call already executed above
    if k == 0:
        return wrap_with_symmetry(result, SymmetryGroup.symmetric(axes=(0, 1)))
    return result


attach_docstring(diagflat, _np.diagflat, "free", "0 FLOPs")


def dsplit(ary, *args, **kwargs):
    """Split array along third axis. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    with budget.deduct(
        "dsplit", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    ):
        result = _np.dsplit(ary, *args, **kwargs)
    return result


attach_docstring(dsplit, _np.dsplit, "free", "0 FLOPs")


def dstack(*args, **kwargs):
    """Stack arrays along third axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.dstack(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("dstack", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(dstack, _np.dstack, "free", "0 FLOPs")


def extract(condition, arr, *args, **kwargs):
    """Return elements satisfying condition. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    with budget.deduct(
        "extract", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
    ):
        result = _np.extract(condition, arr, *args, **kwargs)
    return result


attach_docstring(extract, _np.extract, "free", "0 FLOPs")


def fill_diagonal(a, val, wrap=False, **kwargs):
    """Fill main diagonal of array in-place. Cost: min(m,n)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = min(a_arr.shape[0], a_arr.shape[1]) if a_arr.ndim >= 2 else a_arr.size
    with budget.deduct(
        "fill_diagonal", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _np.fill_diagonal(a, val, wrap=wrap, **kwargs)
    return result


attach_docstring(fill_diagonal, _np.fill_diagonal, "free", "0 FLOPs")


def flatnonzero(a, *args, **kwargs):
    """Return indices of non-zero elements in flattened array. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "flatnonzero", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _np.flatnonzero(a, *args, **kwargs)
    return result


attach_docstring(flatnonzero, _np.flatnonzero, "free", "0 FLOPs")


def fliplr(*args, **kwargs):
    """Reverse elements along axis 1. Wraps ``numpy.fliplr``. Cost: 0 FLOPs."""
    return _np.fliplr(*args, **kwargs)


attach_docstring(fliplr, _np.fliplr, "free", "0 FLOPs")


def flipud(*args, **kwargs):
    """Reverse elements along axis 0. Wraps ``numpy.flipud``. Cost: 0 FLOPs."""
    return _np.flipud(*args, **kwargs)


attach_docstring(flipud, _np.flipud, "free", "0 FLOPs")


def from_dlpack(*args, **kwargs):
    """Create array from DLPack capsule. Cost: numel(output)."""
    budget = require_budget()
    result = _np.from_dlpack(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("from_dlpack", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(from_dlpack, _np.from_dlpack, "free", "0 FLOPs")


def frombuffer(*args, **kwargs):
    """Interpret buffer as 1-D array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.frombuffer(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("frombuffer", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(frombuffer, _np.frombuffer, "free", "0 FLOPs")


def fromfile(*args, **kwargs):
    """Construct array from data in text or binary file. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfile(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromfile", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromfile, _np.fromfile, "free", "0 FLOPs")


def fromfunction(*args, **kwargs):
    """Construct array by executing function over each coordinate. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfunction(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromfunction", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromfunction, _np.fromfunction, "free", "0 FLOPs")


def fromiter(*args, **kwargs):
    """Create array from iterable object. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromiter(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromiter", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromiter, _np.fromiter, "free", "0 FLOPs")


def fromregex(*args, **kwargs):
    """Construct array from text file using regex. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromregex(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromregex", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromregex, _np.fromregex, "free", "0 FLOPs")


def fromstring(*args, **kwargs):
    """Construct array from string. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromstring(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("fromstring", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(fromstring, _np.fromstring, "free", "0 FLOPs")


def indices(*args, **kwargs):
    """Return array representing indices of a grid. Cost: numel(output)."""
    budget = require_budget()
    result = _np.indices(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("indices", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(indices, _np.indices, "free", "0 FLOPs")


def insert(arr, obj, values, axis=None, **kwargs):
    """Insert values along axis before given indices. Cost: numel(inserted values)."""
    budget = require_budget()
    values_arr = _np.asarray(values)
    cost = values_arr.size  # num inserted
    with budget.deduct("insert", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.insert(arr, obj, values, axis=axis, **kwargs)
    return result


attach_docstring(insert, _np.insert, "free", "0 FLOPs")


def isdtype(*args, **kwargs):
    """Returns boolean indicating whether a provided dtype is of a specified kind. Wraps ``numpy.isdtype``. Cost: 0 FLOPs."""
    return _np.isdtype(*args, **kwargs)


attach_docstring(isdtype, _np.isdtype, "free", "0 FLOPs")


def isfortran(*args, **kwargs):
    """Returns True if array is Fortran contiguous. Wraps ``numpy.isfortran``. Cost: 0 FLOPs."""
    return _np.isfortran(*args, **kwargs)


attach_docstring(isfortran, _np.isfortran, "free", "0 FLOPs")


def isin(*args, **kwargs):
    """Test element-wise membership in a set. Wraps ``numpy.isin``. Cost: 0 FLOPs."""
    return _np.isin(*args, **kwargs)


attach_docstring(isin, _np.isin, "free", "0 FLOPs")


def isscalar(*args, **kwargs):
    """Returns True if element is scalar type. Wraps ``numpy.isscalar``. Cost: 0 FLOPs."""
    return _np.isscalar(*args, **kwargs)


attach_docstring(isscalar, _np.isscalar, "free", "0 FLOPs")


def issubdtype(*args, **kwargs):
    """Returns True if first argument is a typecode lower/equal in type hierarchy. Wraps ``numpy.issubdtype``. Cost: 0 FLOPs."""
    return _np.issubdtype(*args, **kwargs)


attach_docstring(issubdtype, _np.issubdtype, "free", "0 FLOPs")


def iterable(*args, **kwargs):
    """Check whether or not object is iterable. Wraps ``numpy.iterable``. Cost: 0 FLOPs."""
    return _np.iterable(*args, **kwargs)


attach_docstring(iterable, _np.iterable, "free", "0 FLOPs")


def ix_(*args, **kwargs):
    """Construct open mesh from multiple sequences. Cost: numel(output)."""
    budget = require_budget()
    result = _np.ix_(*args, **kwargs)
    cost = sum(a.size for a in result)
    with budget.deduct("ix_", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(ix_, _np.ix_, "free", "0 FLOPs")


def mask_indices(*args, **kwargs):
    """Return indices to access main or off-diagonal of array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.mask_indices(*args, **kwargs)
    cost = sum(a.size for a in result) if isinstance(result, tuple) else 1
    with budget.deduct("mask_indices", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(mask_indices, _np.mask_indices, "free", "0 FLOPs")


def matrix_transpose(*args, **kwargs):
    """Transpose a matrix or stack of matrices. Wraps ``numpy.matrix_transpose``. Cost: 0 FLOPs."""
    return _np.matrix_transpose(*args, **kwargs)


attach_docstring(matrix_transpose, _np.matrix_transpose, "free", "0 FLOPs")


def may_share_memory(*args, **kwargs):
    """Determine if two arrays might share memory. Wraps ``numpy.may_share_memory``. Cost: 0 FLOPs."""
    return _np.may_share_memory(*args, **kwargs)


attach_docstring(may_share_memory, _np.may_share_memory, "free", "0 FLOPs")


def min_scalar_type(*args, **kwargs):
    """Return smallest scalar type. Wraps ``numpy.min_scalar_type``. Cost: 0 FLOPs."""
    return _np.min_scalar_type(*args, **kwargs)


attach_docstring(min_scalar_type, _np.min_scalar_type, "free", "0 FLOPs")


def mintypecode(*args, **kwargs):
    """Return minimum data type character. Wraps ``numpy.mintypecode``. Cost: 0 FLOPs."""
    return _np.mintypecode(*args, **kwargs)


attach_docstring(mintypecode, _np.mintypecode, "free", "0 FLOPs")


def ndim(*args, **kwargs):
    """Return number of dimensions. Wraps ``numpy.ndim``. Cost: 0 FLOPs."""
    return _np.ndim(*args, **kwargs)


attach_docstring(ndim, _np.ndim, "free", "0 FLOPs")


def nonzero(a, *args, **kwargs):
    """Return indices of non-zero elements. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "nonzero", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _np.nonzero(a, *args, **kwargs)
    return result


attach_docstring(nonzero, _np.nonzero, "free", "0 FLOPs")


def packbits(a, *args, **kwargs):
    """Pack binary-valued array into bits. Cost: numel(output)."""
    budget = require_budget()
    result = _np.packbits(a, *args, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "packbits", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(packbits, _np.packbits, "free", "0 FLOPs")


def permute_dims(*args, **kwargs):
    """Permute dimensions of array. Wraps ``numpy.permute_dims``. Cost: 0 FLOPs."""
    return _np.permute_dims(*args, **kwargs)


attach_docstring(permute_dims, _np.permute_dims, "free", "0 FLOPs")


def place(arr, mask, vals, *args, **kwargs):
    """Change elements of array based on conditional. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    with budget.deduct(
        "place", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
    ):
        result = _np.place(arr, mask, vals, *args, **kwargs)
    return result


attach_docstring(place, _np.place, "free", "0 FLOPs")


def promote_types(*args, **kwargs):
    """Return smallest size and least significant type. Wraps ``numpy.promote_types``. Cost: 0 FLOPs."""
    return _np.promote_types(*args, **kwargs)


attach_docstring(promote_types, _np.promote_types, "free", "0 FLOPs")


def put(a, ind, v, *args, **kwargs):
    """Replace elements at given flat indices. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct("put", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)):
        result = _np.put(a, ind, v, *args, **kwargs)
    return result


attach_docstring(put, _np.put, "free", "0 FLOPs")


def put_along_axis(arr, indices, values, axis, *args, **kwargs):
    """Put values into destination array along axis. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    with budget.deduct(
        "put_along_axis", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
    ):
        result = _np.put_along_axis(arr, indices, values, axis, *args, **kwargs)
    return result


attach_docstring(put_along_axis, _np.put_along_axis, "free", "0 FLOPs")


def putmask(a, mask, values, *args, **kwargs):
    """Change elements of array based on condition. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    with budget.deduct(
        "putmask", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    ):
        result = _np.putmask(a, mask, values, *args, **kwargs)
    return result


attach_docstring(putmask, _np.putmask, "free", "0 FLOPs")


def ravel_multi_index(*args, **kwargs):
    """Convert multi-index to flat index. Wraps ``numpy.ravel_multi_index``. Cost: 0 FLOPs."""
    return _np.ravel_multi_index(*args, **kwargs)


attach_docstring(ravel_multi_index, _np.ravel_multi_index, "free", "0 FLOPs")


def require(*args, **kwargs):
    """Return array satisfying requirements. Wraps ``numpy.require``. Cost: 0 FLOPs."""
    return _np.require(*args, **kwargs)


attach_docstring(require, _np.require, "free", "0 FLOPs")


def resize(*args, **kwargs):
    """Return new array with given shape. Cost: numel(output)."""
    budget = require_budget()
    result = _np.resize(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("resize", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(resize, _np.resize, "free", "0 FLOPs")


def result_type(*args, **kwargs):
    """Returns type that results from applying type promotion. Wraps ``numpy.result_type``. Cost: 0 FLOPs."""
    return _np.result_type(*args, **kwargs)


attach_docstring(result_type, _np.result_type, "free", "0 FLOPs")


def rollaxis(*args, **kwargs):
    """Roll specified axis backwards. Cost: numel(output)."""
    budget = require_budget()
    result = _np.rollaxis(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("rollaxis", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(rollaxis, _np.rollaxis, "free", "0 FLOPs")


def rot90(*args, **kwargs):
    """Rotate array 90 degrees. Wraps ``numpy.rot90``. Cost: 0 FLOPs."""
    return _np.rot90(*args, **kwargs)


attach_docstring(rot90, _np.rot90, "free", "0 FLOPs")


def row_stack(*args, **kwargs):
    """Stack arrays vertically (alias for vstack). Wraps ``numpy.row_stack``. Cost: 0 FLOPs."""
    return _np.row_stack(*args, **kwargs)


attach_docstring(row_stack, _np.row_stack, "free", "0 FLOPs")


def select(condlist, choicelist, default=0):
    """Return array drawn from elements depending on conditions. Cost: numel(input)."""
    budget = require_budget()
    # Cost based on the size of the choice arrays
    cost = max((_np.asarray(c).size for c in choicelist), default=1)
    with budget.deduct("select", flop_cost=cost, subscripts=None, shapes=()):
        result = _np.select(condlist, choicelist, default=default)
    return result


attach_docstring(select, _np.select, "free", "0 FLOPs")


def shape(*args, **kwargs):
    """Return shape of array. Wraps ``numpy.shape``. Cost: 0 FLOPs."""
    return _np.shape(*args, **kwargs)


attach_docstring(shape, _np.shape, "free", "0 FLOPs")


def shares_memory(*args, **kwargs):
    """Determine if two arrays share memory. Wraps ``numpy.shares_memory``. Cost: 0 FLOPs."""
    return _np.shares_memory(*args, **kwargs)


attach_docstring(shares_memory, _np.shares_memory, "free", "0 FLOPs")


def size(*args, **kwargs):
    """Return number of elements along a given axis. Wraps ``numpy.size``. Cost: 0 FLOPs."""
    return _np.size(*args, **kwargs)


attach_docstring(size, _np.size, "free", "0 FLOPs")


def take(*args, **kwargs):
    """Take elements from array along axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.take(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("take", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(take, _np.take, "free", "0 FLOPs")


def take_along_axis(*args, **kwargs):
    """Take values from input array along axis using indices. Cost: numel(output)."""
    budget = require_budget()
    result = _np.take_along_axis(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    with budget.deduct("take_along_axis", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(take_along_axis, _np.take_along_axis, "free", "0 FLOPs")


def tri(*args, **kwargs):
    """Array with ones at and below the given diagonal. Wraps ``numpy.tri``. Cost: 0 FLOPs."""
    return _np.tri(*args, **kwargs)


attach_docstring(tri, _np.tri, "free", "0 FLOPs")


def tril_indices(*args, **kwargs):
    """Return indices for lower-triangle of array. Wraps ``numpy.tril_indices``. Cost: 0 FLOPs."""
    return _np.tril_indices(*args, **kwargs)


attach_docstring(tril_indices, _np.tril_indices, "free", "0 FLOPs")


def tril_indices_from(*args, **kwargs):
    """Return indices for lower-triangle of given array. Wraps ``numpy.tril_indices_from``. Cost: 0 FLOPs."""
    return _np.tril_indices_from(*args, **kwargs)


attach_docstring(tril_indices_from, _np.tril_indices_from, "free", "0 FLOPs")


def trim_zeros(filt, trim="fb", **kwargs):
    """Trim leading and/or trailing zeros from 1-D array. Cost: num elements trimmed."""
    budget = require_budget()
    filt_arr = _np.asarray(filt)
    result = _np.trim_zeros(filt, trim=trim, **kwargs)
    result_arr = _np.asarray(result)
    cost = max(filt_arr.size - result_arr.size, 0)  # num trimmed
    with budget.deduct("trim_zeros", flop_cost=cost, subscripts=None, shapes=()):
        pass  # numpy call already executed above
    return result


attach_docstring(trim_zeros, _np.trim_zeros, "free", "0 FLOPs")


def triu_indices(*args, **kwargs):
    """Return indices for upper-triangle of array. Wraps ``numpy.triu_indices``. Cost: 0 FLOPs."""
    return _np.triu_indices(*args, **kwargs)


attach_docstring(triu_indices, _np.triu_indices, "free", "0 FLOPs")


def triu_indices_from(*args, **kwargs):
    """Return indices for upper-triangle of given array. Wraps ``numpy.triu_indices_from``. Cost: 0 FLOPs."""
    return _np.triu_indices_from(*args, **kwargs)


attach_docstring(triu_indices_from, _np.triu_indices_from, "free", "0 FLOPs")


def typename(*args, **kwargs):
    """Return description for given data type code. Wraps ``numpy.typename``. Cost: 0 FLOPs."""
    return _np.typename(*args, **kwargs)


attach_docstring(typename, _np.typename, "free", "0 FLOPs")


def unpackbits(a, *args, **kwargs):
    """Unpack elements of uint8 array into binary-valued bit array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.unpackbits(a, *args, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    with budget.deduct(
        "unpackbits", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    ):
        pass  # numpy call already executed above
    return result


attach_docstring(unpackbits, _np.unpackbits, "free", "0 FLOPs")


def unravel_index(*args, **kwargs):
    """Convert flat indices to multi-dimensional index. Wraps ``numpy.unravel_index``. Cost: 0 FLOPs."""
    return _np.unravel_index(*args, **kwargs)


attach_docstring(unravel_index, _np.unravel_index, "free", "0 FLOPs")


if hasattr(_np, "unstack"):

    def unstack(x, *args, **kwargs):
        """Split array into sequence of arrays along an axis. Cost: numel(input)."""
        budget = require_budget()
        x_arr = _np.asarray(x)
        cost = x_arr.size
        with budget.deduct(
            "unstack", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
        ):
            result = _np.unstack(x, *args, **kwargs)
        return result

    attach_docstring(unstack, _np.unstack, "free", "0 FLOPs")

else:

    def unstack(*args, **kwargs):
        raise UnsupportedFunctionError("unstack", min_version="2.1")


# ---------------------------------------------------------------------------
# Wrap all free op return values as WhestArray
# ---------------------------------------------------------------------------

from whest._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_FREE_OPS_SKIP = {
    "shape",
    "size",
    "ndim",
    "isscalar",
    "isfortran",
    "isfinite",
    "isinf",
    "isnan",
    "isdtype",
    "issubdtype",
    "iscomplex",
    "iscomplexobj",
    "isnat",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "iterable",
    "may_share_memory",
    "shares_memory",
    "can_cast",
    "common_type",
    "min_scalar_type",
    "promote_types",
    "result_type",
    "typename",
    "base_repr",
    "binary_repr",
    "broadcast_shapes",
    "fill_diagonal",
}

import sys as _sys  # noqa: E402

# ---------------------------------------------------------------------------
# Signature conformance: set __signature__ to match numpy exactly
# ---------------------------------------------------------------------------
_this_module = _sys.modules[__name__]


def _set_sig(func_name, np_func):
    """Set __signature__ of a module-level function to match numpy."""
    fn = globals().get(func_name)
    if fn is not None and callable(np_func):
        try:
            fn.__signature__ = _inspect.signature(np_func)
        except (ValueError, TypeError):
            pass


# Functions with *args/**kwargs that need numpy's signature
_set_sig("arange", _np.arange)
_set_sig("array", _np.array)
_set_sig("zeros", _np.zeros)
_set_sig("ones", _np.ones)
_set_sig("full", _np.full)
_set_sig("eye", _np.eye)
_set_sig("linspace", _np.linspace)
_set_sig("zeros_like", _np.zeros_like)
_set_sig("ones_like", _np.ones_like)
_set_sig("full_like", _np.full_like)
_set_sig("empty", _np.empty)
_set_sig("empty_like", _np.empty_like)
_set_sig("identity", _np.identity)
_set_sig("reshape", _np.reshape)
_set_sig("concatenate", _np.concatenate)
_set_sig("stack", _np.stack)
_set_sig("vstack", _np.vstack)
_set_sig("hstack", _np.hstack)
_set_sig("ravel", _np.ravel)
_set_sig("copy", _np.copy)
_set_sig("pad", _np.pad)
_set_sig("broadcast_to", _np.broadcast_to)
_set_sig("meshgrid", _np.meshgrid)
_set_sig("asarray", _np.asarray)
_set_sig("astype", _np.astype)
_set_sig("append", _np.append)
_set_sig("argwhere", _np.argwhere)
_set_sig("array_split", _np.array_split)
_set_sig("asarray_chkfinite", _np.asarray_chkfinite)
_set_sig("atleast_1d", _np.atleast_1d)
_set_sig("atleast_2d", _np.atleast_2d)
_set_sig("atleast_3d", _np.atleast_3d)
_set_sig("base_repr", _np.base_repr)
_set_sig("binary_repr", _np.binary_repr)
_set_sig("block", _np.block)
_set_sig("bmat", _np.bmat)
_set_sig("broadcast_arrays", _np.broadcast_arrays)
_set_sig("broadcast_shapes", _np.broadcast_shapes)
_set_sig("can_cast", _np.can_cast)
_set_sig("choose", _np.choose)
_set_sig("column_stack", _np.column_stack)
_set_sig("common_type", _np.common_type)
_set_sig("compress", _np.compress)
_set_sig("concat", _np.concat)
_set_sig("delete", _np.delete)
_set_sig("diag_indices", _np.diag_indices)
_set_sig("diag_indices_from", _np.diag_indices_from)
_set_sig("dsplit", _np.dsplit)
_set_sig("dstack", _np.dstack)
_set_sig("extract", _np.extract)
_set_sig("flatnonzero", _np.flatnonzero)
_set_sig("fliplr", _np.fliplr)
_set_sig("flipud", _np.flipud)
_set_sig("from_dlpack", _np.from_dlpack)
_set_sig("frombuffer", _np.frombuffer)
_set_sig("fromfile", _np.fromfile)
_set_sig("fromfunction", _np.fromfunction)
_set_sig("fromiter", _np.fromiter)
_set_sig("fromregex", _np.fromregex)
_set_sig("fromstring", _np.fromstring)
_set_sig("indices", _np.indices)
_set_sig("insert", _np.insert)
_set_sig("isdtype", _np.isdtype)
_set_sig("isfortran", _np.isfortran)
_set_sig("isin", _np.isin)
_set_sig("isnan", _np.isnan)
_set_sig("isfinite", _np.isfinite)
_set_sig("isinf", _np.isinf)
_set_sig("isscalar", _np.isscalar)
_set_sig("issubdtype", _np.issubdtype)
_set_sig("iterable", _np.iterable)
_set_sig("ix_", _np.ix_)
_set_sig("mask_indices", _np.mask_indices)
_set_sig("matrix_transpose", _np.matrix_transpose)
_set_sig("may_share_memory", _np.may_share_memory)
_set_sig("min_scalar_type", _np.min_scalar_type)
_set_sig("mintypecode", _np.mintypecode)
_set_sig("ndim", _np.ndim)
_set_sig("nonzero", _np.nonzero)
_set_sig("permute_dims", _np.permute_dims)
_set_sig("put", _np.put)
_set_sig("require", _np.require)
_set_sig("resize", _np.resize)
_set_sig("rollaxis", _np.rollaxis)
_set_sig("rot90", _np.rot90)
_set_sig("row_stack", _np.row_stack)
_set_sig("shape", _np.shape)
_set_sig("size", _np.size)
_set_sig("take", _np.take)
_set_sig("take_along_axis", _np.take_along_axis)
_set_sig("tri", _np.tri)
_set_sig("tril_indices", _np.tril_indices)
_set_sig("tril_indices_from", _np.tril_indices_from)
_set_sig("trim_zeros", _np.trim_zeros)
_set_sig("triu_indices", _np.triu_indices)
_set_sig("triu_indices_from", _np.triu_indices_from)
_set_sig("typename", _np.typename)
_set_sig("unravel_index", _np.unravel_index)
if hasattr(_np, "unstack"):
    _set_sig("unstack", _np.unstack)

del _set_sig, _this_module

_wrap_module_returns(_sys.modules[__name__], skip_names=_FREE_OPS_SKIP)
