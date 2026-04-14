"""Zero-FLOP wrappers around NumPy tensor creation and manipulation.

Every function in this module delegates directly to the corresponding
NumPy function and costs **0 FLOPs**, so they work both inside and
outside a :class:`~whest._budget.BudgetContext`.
"""

from __future__ import annotations

import numpy as _np

from whest._docstrings import attach_docstring
from whest._symmetric import SymmetricTensor
from whest._validation import require_budget
from whest.errors import UnsupportedFunctionError


def _symmetric_2d(result):
    """Wrap a 2D square result as SymmetricTensor with axes (0,1)."""
    if result.ndim == 2 and result.shape[0] == result.shape[1]:
        return SymmetricTensor(result, symmetric_axes=[(0, 1)])
    return result


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------


def array(object, dtype=None, **kwargs):
    """Create an array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.array(object, dtype=dtype, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    budget.deduct("array", flop_cost=cost, subscripts=None, shapes=(result.shape,))
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
    budget.deduct("full", flop_cost=cost, subscripts=None, shapes=())
    return _symmetric_2d(result)


attach_docstring(full, _np.full, "free", "0 FLOPs")


def eye(N, M=None, k=0, dtype=float, **kwargs):
    """Return identity matrix. Wraps ``numpy.eye``. Cost: 0 FLOPs."""
    result = _np.eye(N, M=M, k=k, dtype=dtype, **kwargs)
    if k == 0 and (M is None or M == N):
        return SymmetricTensor(result, symmetric_axes=[(0, 1)])
    return result


attach_docstring(eye, _np.eye, "free", "0 FLOPs")


def diag(v, k=0):
    """Extract diagonal or construct diagonal array.

    Cost: numel(output) when constructing (1D→2D), min(m,n) when extracting (2D→1D).
    """
    budget = require_budget()
    v = _np.asarray(v)
    result = _np.diag(v, k=k)
    if v.ndim == 1:
        # Constructing diagonal matrix: real work is allocating + zeroing the output
        cost = result.size
    else:
        # Extracting diagonal: reads min(m,n) elements
        m, n = v.shape[0], v.shape[1] if v.ndim > 1 else v.shape[0]
        cost = min(m, n)
    budget.deduct("diag", flop_cost=cost, subscripts=None, shapes=(v.shape,))
    if v.ndim == 1 and k == 0:
        return SymmetricTensor(result, symmetric_axes=[(0, 1)])
    return result


attach_docstring(diag, _np.diag, "free", "0 FLOPs")


def arange(*args, **kwargs):
    """Return evenly spaced values. Cost: numel(output)."""
    budget = require_budget()
    result = _np.arange(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("arange", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(arange, _np.arange, "counted_custom", "numel(output) FLOPs")


def linspace(start, stop, num=50, **kwargs):
    """Return evenly spaced numbers. Cost: numel(output)."""
    budget = require_budget()
    result = _np.linspace(start, stop, num=num, **kwargs)
    # linspace may return (array, step) tuple if retstep=True
    if isinstance(result, tuple):
        cost = result[0].size if hasattr(result[0], "size") else 1
    else:
        cost = result.size if hasattr(result, "size") else 1
    budget.deduct("linspace", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(linspace, _np.linspace, "counted_custom", "numel(output) FLOPs")


def zeros_like(a, dtype=None, **kwargs):
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    result = _np.zeros_like(a, dtype=dtype, **kwargs)
    if isinstance(a, SymmetricTensor) and a._symmetric_axes:
        return SymmetricTensor(result, symmetric_axes=list(a._symmetric_axes))
    return result


attach_docstring(zeros_like, _np.zeros_like, "free", "0 FLOPs")


def ones_like(a, dtype=None, **kwargs):
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: 0 FLOPs."""
    result = _np.ones_like(a, dtype=dtype, **kwargs)
    if isinstance(a, SymmetricTensor) and a._symmetric_axes:
        return SymmetricTensor(result, symmetric_axes=list(a._symmetric_axes))
    return result


attach_docstring(ones_like, _np.ones_like, "free", "0 FLOPs")


def full_like(a, fill_value, dtype=None, **kwargs):
    """Return full array with same shape. Cost: numel(output)."""
    budget = require_budget()
    result = _np.full_like(a, fill_value, dtype=dtype, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("full_like", flop_cost=cost, subscripts=None, shapes=())
    if isinstance(a, SymmetricTensor) and a._symmetric_axes:
        return SymmetricTensor(result, symmetric_axes=list(a._symmetric_axes))
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
    return SymmetricTensor(result, symmetric_axes=[(0, 1)])


attach_docstring(identity, _np.identity, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Tensor manipulation
# ---------------------------------------------------------------------------


def reshape(a, /, *args, **kwargs):
    """Reshape an array. Wraps ``numpy.reshape``. Cost: 0 FLOPs."""
    return _np.reshape(a, *args, **kwargs)


attach_docstring(reshape, _np.reshape, "free", "0 FLOPs")


def transpose(a, axes=None):
    """Permute array dimensions. Wraps ``numpy.transpose``. Cost: 0 FLOPs."""
    return _np.transpose(a, axes=axes)


attach_docstring(transpose, _np.transpose, "free", "0 FLOPs")


def swapaxes(a, axis1, axis2):
    """Swap two axes. Wraps ``numpy.swapaxes``. Cost: 0 FLOPs."""
    return _np.swapaxes(a, axis1, axis2)


attach_docstring(swapaxes, _np.swapaxes, "free", "0 FLOPs")


def moveaxis(a, source, destination):
    """Move axes to new positions. Wraps ``numpy.moveaxis``. Cost: 0 FLOPs."""
    return _np.moveaxis(a, source, destination)


attach_docstring(moveaxis, _np.moveaxis, "free", "0 FLOPs")


def concatenate(arrays, axis=0, **kwargs):
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.concatenate(arrays, axis=axis, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("concatenate", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(concatenate, _np.concatenate, "counted_custom", "numel(output) FLOPs")


def stack(arrays, axis=0, **kwargs):
    """Stack arrays along a new axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.stack(arrays, axis=axis, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("stack", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(stack, _np.stack, "free", "0 FLOPs")


def vstack(tup):
    """Stack arrays vertically. Cost: numel(output)."""
    budget = require_budget()
    result = _np.vstack(tup)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("vstack", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("split", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,))
    return _np.split(ary, indices_or_sections, axis=axis)


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
    budget.deduct("vsplit", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,))
    return _np.vsplit(ary, indices_or_sections)


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
    result = _np.ravel(a, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    budget.deduct("ravel", flop_cost=cost, subscripts=None, shapes=(result.shape,))
    return result


attach_docstring(ravel, _np.ravel, "free", "0 FLOPs")


def copy(a, **kwargs):
    """Return copy of array. Wraps ``numpy.copy``. Cost: 0 FLOPs."""
    return _np.copy(a, **kwargs)


attach_docstring(copy, _np.copy, "free", "0 FLOPs")


def where(condition, x=None, y=None):
    """Return elements chosen from *x* or *y*. Cost: numel(input)."""
    budget = require_budget()
    cond_arr = _np.asarray(condition)
    cost = cond_arr.size
    budget.deduct("where", flop_cost=cost, subscripts=None, shapes=(cond_arr.shape,))
    if x is None and y is None:
        return _np.where(condition)
    return _np.where(condition, x, y)


attach_docstring(where, _np.where, "free", "0 FLOPs")


def tile(A, reps):
    """Construct array by repeating. Cost: numel(output)."""
    budget = require_budget()
    result = _np.tile(A, reps)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("tile", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(tile, _np.tile, "free", "0 FLOPs")


def repeat(a, repeats, axis=None):
    """Repeat elements. Cost: numel(output)."""
    budget = require_budget()
    result = _np.repeat(a, repeats, axis=axis)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("repeat", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(repeat, _np.repeat, "free", "0 FLOPs")


def flip(m, axis=None):
    """Reverse order of elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    return _np.flip(m, axis=axis)


attach_docstring(flip, _np.flip, "free", "0 FLOPs")


def roll(a, shift, axis=None):
    """Roll array elements. Cost: numel(output)."""
    budget = require_budget()
    result = _np.roll(a, shift, axis=axis)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("roll", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(roll, _np.roll, "free", "0 FLOPs")


def pad(array, pad_width, **kwargs):
    """Pad an array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.pad(array, pad_width, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("pad", flop_cost=cost, subscripts=None, shapes=())
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
    result = _np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    budget.deduct("diagonal", flop_cost=cost, subscripts=None, shapes=(result.shape,))
    return result


attach_docstring(diagonal, _np.diagonal, "free", "0 FLOPs")


def broadcast_to(array, shape):
    """Broadcast array to shape. Cost: numel(output)."""
    budget = require_budget()
    result = _np.broadcast_to(array, shape)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("broadcast_to", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(broadcast_to, _np.broadcast_to, "free", "0 FLOPs")


def meshgrid(*xi, **kwargs):
    """Return coordinate matrices. Cost: numel(output)."""
    budget = require_budget()
    result = _np.meshgrid(*xi, **kwargs)
    cost = sum(a.size for a in result)
    budget.deduct("meshgrid", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(meshgrid, _np.meshgrid, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Type / info helpers
# ---------------------------------------------------------------------------


def astype(x, dtype):
    """Cast array to *dtype*. Wraps ``x.astype(dtype)``. Cost: 0 FLOPs."""
    return x.astype(dtype)


def asarray(a, dtype=None, **kwargs):
    """Convert to array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.asarray(a, dtype=dtype, **kwargs)
    cost = (
        result.size
        if hasattr(result, "size")
        else len(result)
        if hasattr(result, "__len__")
        else 1
    )
    budget.deduct("asarray", flop_cost=cost, subscripts=None, shapes=(result.shape,))
    return result


attach_docstring(asarray, _np.asarray, "free", "0 FLOPs")


def isnan(x, **kwargs):
    """Test element-wise for NaN. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    budget.deduct("isnan", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,))
    return _np.isnan(x, **kwargs)


attach_docstring(isnan, _np.isnan, "free", "0 FLOPs")


def isfinite(x, **kwargs):
    """Test element-wise for finiteness. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    budget.deduct("isfinite", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,))
    return _np.isfinite(x, **kwargs)


attach_docstring(isfinite, _np.isfinite, "free", "0 FLOPs")


def isinf(x, **kwargs):
    """Test element-wise for Inf. Cost: numel(input)."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = x_arr.size
    budget.deduct("isinf", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,))
    return _np.isinf(x, **kwargs)


attach_docstring(isinf, _np.isinf, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# New free ops
# ---------------------------------------------------------------------------


def append(arr, values, axis=None, **kwargs):
    """Append values. Cost: numel(appended values)."""
    budget = require_budget()
    values_arr = _np.asarray(values)
    cost = values_arr.size  # num appended
    budget.deduct("append", flop_cost=cost, subscripts=None, shapes=())
    result = _np.append(arr, values, axis=axis, **kwargs)
    return result


attach_docstring(append, _np.append, "free", "0 FLOPs")


def argwhere(a, *args, **kwargs):
    """Find indices of non-zero elements. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    budget.deduct("argwhere", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,))
    return _np.argwhere(a, *args, **kwargs)


attach_docstring(argwhere, _np.argwhere, "free", "0 FLOPs")


def array_split(ary, *args, **kwargs):
    """Split array into sub-arrays. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    budget.deduct(
        "array_split", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,)
    )
    return _np.array_split(ary, *args, **kwargs)


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
    budget.deduct(
        "asarray_chkfinite", flop_cost=cost, subscripts=None, shapes=(result.shape,)
    )
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
    budget.deduct("base_repr", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(base_repr, _np.base_repr, "free", "0 FLOPs")


def binary_repr(*args, **kwargs):
    """Return binary representation of integer. Cost: numel(output)."""
    budget = require_budget()
    result = _np.binary_repr(*args, **kwargs)
    cost = len(result)
    budget.deduct("binary_repr", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(binary_repr, _np.binary_repr, "free", "0 FLOPs")


def block(*args, **kwargs):
    """Assemble array from nested lists. Cost: numel(output)."""
    budget = require_budget()
    result = _np.block(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("block", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(block, _np.block, "free", "0 FLOPs")


def bmat(*args, **kwargs):
    """Build matrix from string/nested sequence. Cost: numel(output)."""
    budget = require_budget()
    result = _np.bmat(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("bmat", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(bmat, _np.bmat, "free", "0 FLOPs")


def broadcast_arrays(*args, **kwargs):
    """Broadcast any number of arrays. Cost: numel(output)."""
    budget = require_budget()
    result = _np.broadcast_arrays(*args, **kwargs)
    cost = sum(a.size for a in result)
    budget.deduct("broadcast_arrays", flop_cost=cost, subscripts=None, shapes=())
    return result


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
    budget.deduct("choose", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("compress", flop_cost=cost, subscripts=None, shapes=(result.shape,))
    return result


attach_docstring(compress, _np.compress, "free", "0 FLOPs")


def concat(*args, **kwargs):
    """Join arrays along an axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.concat(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("concat", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("copyto", flop_cost=cost, subscripts=None, shapes=())
    return _np.copyto(dst, src, casting=casting, where=where)


attach_docstring(copyto, _np.copyto, "free", "0 FLOPs")


def delete(arr, obj, axis=None, **kwargs):
    """Return new array with sub-arrays deleted. Cost: num elements removed."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    result = _np.delete(arr, obj, axis=axis, **kwargs)
    cost = max(arr_np.size - result.size, 0)  # num deleted
    budget.deduct("delete", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("diagflat", flop_cost=cost, subscripts=None, shapes=(v_arr.shape,))
    if k == 0:
        return SymmetricTensor(result, symmetric_axes=[(0, 1)])
    return result


attach_docstring(diagflat, _np.diagflat, "free", "0 FLOPs")


def dsplit(ary, *args, **kwargs):
    """Split array along third axis. Cost: numel(input)."""
    budget = require_budget()
    ary_arr = _np.asarray(ary)
    cost = ary_arr.size
    budget.deduct("dsplit", flop_cost=cost, subscripts=None, shapes=(ary_arr.shape,))
    return _np.dsplit(ary, *args, **kwargs)


attach_docstring(dsplit, _np.dsplit, "free", "0 FLOPs")


def dstack(*args, **kwargs):
    """Stack arrays along third axis. Cost: numel(output)."""
    budget = require_budget()
    result = _np.dstack(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("dstack", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(dstack, _np.dstack, "free", "0 FLOPs")


def extract(condition, arr, *args, **kwargs):
    """Return elements satisfying condition. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    budget.deduct("extract", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,))
    return _np.extract(condition, arr, *args, **kwargs)


attach_docstring(extract, _np.extract, "free", "0 FLOPs")


def fill_diagonal(a, val, wrap=False, **kwargs):
    """Fill main diagonal of array in-place. Cost: min(m,n)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = min(a_arr.shape[0], a_arr.shape[1]) if a_arr.ndim >= 2 else a_arr.size
    budget.deduct(
        "fill_diagonal", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,)
    )
    return _np.fill_diagonal(a, val, wrap=wrap, **kwargs)


attach_docstring(fill_diagonal, _np.fill_diagonal, "free", "0 FLOPs")


def flatnonzero(a, *args, **kwargs):
    """Return indices of non-zero elements in flattened array. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    budget.deduct("flatnonzero", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,))
    return _np.flatnonzero(a, *args, **kwargs)


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
    budget.deduct("from_dlpack", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(from_dlpack, _np.from_dlpack, "free", "0 FLOPs")


def frombuffer(*args, **kwargs):
    """Interpret buffer as 1-D array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.frombuffer(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("frombuffer", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(frombuffer, _np.frombuffer, "free", "0 FLOPs")


def fromfile(*args, **kwargs):
    """Construct array from data in text or binary file. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfile(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("fromfile", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(fromfile, _np.fromfile, "free", "0 FLOPs")


def fromfunction(*args, **kwargs):
    """Construct array by executing function over each coordinate. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromfunction(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("fromfunction", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(fromfunction, _np.fromfunction, "free", "0 FLOPs")


def fromiter(*args, **kwargs):
    """Create array from iterable object. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromiter(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("fromiter", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(fromiter, _np.fromiter, "free", "0 FLOPs")


def fromregex(*args, **kwargs):
    """Construct array from text file using regex. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromregex(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("fromregex", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(fromregex, _np.fromregex, "free", "0 FLOPs")


def fromstring(*args, **kwargs):
    """Construct array from string. Cost: numel(output)."""
    budget = require_budget()
    result = _np.fromstring(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("fromstring", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(fromstring, _np.fromstring, "free", "0 FLOPs")


def indices(*args, **kwargs):
    """Return array representing indices of a grid. Cost: numel(output)."""
    budget = require_budget()
    result = _np.indices(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("indices", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(indices, _np.indices, "free", "0 FLOPs")


def insert(arr, obj, values, axis=None, **kwargs):
    """Insert values along axis before given indices. Cost: numel(inserted values)."""
    budget = require_budget()
    values_arr = _np.asarray(values)
    cost = values_arr.size  # num inserted
    budget.deduct("insert", flop_cost=cost, subscripts=None, shapes=())
    return _np.insert(arr, obj, values, axis=axis, **kwargs)


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
    budget.deduct("ix_", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(ix_, _np.ix_, "free", "0 FLOPs")


def mask_indices(*args, **kwargs):
    """Return indices to access main or off-diagonal of array. Cost: numel(output)."""
    budget = require_budget()
    result = _np.mask_indices(*args, **kwargs)
    cost = sum(a.size for a in result) if isinstance(result, tuple) else 1
    budget.deduct("mask_indices", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("nonzero", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,))
    return _np.nonzero(a, *args, **kwargs)


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
    budget.deduct("packbits", flop_cost=cost, subscripts=None, shapes=(result.shape,))
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
    budget.deduct("place", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,))
    return _np.place(arr, mask, vals, *args, **kwargs)


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
    budget.deduct("put", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,))
    return _np.put(a, ind, v, *args, **kwargs)


attach_docstring(put, _np.put, "free", "0 FLOPs")


def put_along_axis(arr, indices, values, axis, *args, **kwargs):
    """Put values into destination array along axis. Cost: numel(input)."""
    budget = require_budget()
    arr_np = _np.asarray(arr)
    cost = arr_np.size
    budget.deduct(
        "put_along_axis", flop_cost=cost, subscripts=None, shapes=(arr_np.shape,)
    )
    return _np.put_along_axis(arr, indices, values, axis, *args, **kwargs)


attach_docstring(put_along_axis, _np.put_along_axis, "free", "0 FLOPs")


def putmask(a, mask, values, *args, **kwargs):
    """Change elements of array based on condition. Cost: numel(input)."""
    budget = require_budget()
    a_arr = _np.asarray(a)
    cost = a_arr.size
    budget.deduct("putmask", flop_cost=cost, subscripts=None, shapes=(a_arr.shape,))
    return _np.putmask(a, mask, values, *args, **kwargs)


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
    budget.deduct("resize", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("rollaxis", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("select", flop_cost=cost, subscripts=None, shapes=())
    return _np.select(condlist, choicelist, default=default)


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
    budget.deduct("take", flop_cost=cost, subscripts=None, shapes=())
    return result


attach_docstring(take, _np.take, "free", "0 FLOPs")


def take_along_axis(*args, **kwargs):
    """Take values from input array along axis using indices. Cost: numel(output)."""
    budget = require_budget()
    result = _np.take_along_axis(*args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("take_along_axis", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("trim_zeros", flop_cost=cost, subscripts=None, shapes=())
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
    budget.deduct("unpackbits", flop_cost=cost, subscripts=None, shapes=(result.shape,))
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
        budget.deduct("unstack", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,))
        return _np.unstack(x, *args, **kwargs)

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

_wrap_module_returns(_sys.modules[__name__], skip_names=_FREE_OPS_SKIP)
