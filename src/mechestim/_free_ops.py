"""Zero-FLOP wrappers around NumPy tensor creation and manipulation.

Every function in this module delegates directly to the corresponding
NumPy function and costs **0 FLOPs**, so they work both inside and
outside a :class:`~mechestim._budget.BudgetContext`.
"""
from __future__ import annotations

import numpy as _np

from mechestim._docstrings import attach_docstring

# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def array(object, dtype=None, **kwargs):
    """Create an array. Wraps ``numpy.array``. Cost: 0 FLOPs."""
    return _np.array(object, dtype=dtype, **kwargs)

attach_docstring(array, _np.array, "free", "0 FLOPs")

def zeros(shape, dtype=float, **kwargs):
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    return _np.zeros(shape, dtype=dtype, **kwargs)

attach_docstring(zeros, _np.zeros, "free", "0 FLOPs")

def ones(shape, dtype=float, **kwargs):
    """Return array of ones. Wraps ``numpy.ones``. Cost: 0 FLOPs."""
    return _np.ones(shape, dtype=dtype, **kwargs)

attach_docstring(ones, _np.ones, "free", "0 FLOPs")

def full(shape, fill_value, dtype=None, **kwargs):
    """Return array filled with *fill_value*. Wraps ``numpy.full``. Cost: 0 FLOPs."""
    return _np.full(shape, fill_value, dtype=dtype, **kwargs)

attach_docstring(full, _np.full, "free", "0 FLOPs")

def eye(N, M=None, k=0, dtype=float, **kwargs):
    """Return identity matrix. Wraps ``numpy.eye``. Cost: 0 FLOPs."""
    return _np.eye(N, M=M, k=k, dtype=dtype, **kwargs)

attach_docstring(eye, _np.eye, "free", "0 FLOPs")

def diag(v, k=0):
    """Extract diagonal or construct diagonal array. Wraps ``numpy.diag``. Cost: 0 FLOPs."""
    return _np.diag(v, k=k)

attach_docstring(diag, _np.diag, "free", "0 FLOPs")

def arange(*args, **kwargs):
    """Return evenly spaced values. Wraps ``numpy.arange``. Cost: 0 FLOPs."""
    return _np.arange(*args, **kwargs)

attach_docstring(arange, _np.arange, "free", "0 FLOPs")

def linspace(start, stop, num=50, **kwargs):
    """Return evenly spaced numbers. Wraps ``numpy.linspace``. Cost: 0 FLOPs."""
    return _np.linspace(start, stop, num=num, **kwargs)

attach_docstring(linspace, _np.linspace, "free", "0 FLOPs")

def zeros_like(a, dtype=None, **kwargs):
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    return _np.zeros_like(a, dtype=dtype, **kwargs)

attach_docstring(zeros_like, _np.zeros_like, "free", "0 FLOPs")

def ones_like(a, dtype=None, **kwargs):
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: 0 FLOPs."""
    return _np.ones_like(a, dtype=dtype, **kwargs)

attach_docstring(ones_like, _np.ones_like, "free", "0 FLOPs")

def full_like(a, fill_value, dtype=None, **kwargs):
    """Return full array with same shape. Wraps ``numpy.full_like``. Cost: 0 FLOPs."""
    return _np.full_like(a, fill_value, dtype=dtype, **kwargs)

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
    return _np.identity(n, dtype=dtype)

attach_docstring(identity, _np.identity, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Tensor manipulation
# ---------------------------------------------------------------------------

def reshape(a, newshape, **kwargs):
    """Reshape an array. Wraps ``numpy.reshape``. Cost: 0 FLOPs."""
    return _np.reshape(a, newshape, **kwargs)

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
    """Join arrays along an axis. Wraps ``numpy.concatenate``. Cost: 0 FLOPs."""
    return _np.concatenate(arrays, axis=axis, **kwargs)

attach_docstring(concatenate, _np.concatenate, "free", "0 FLOPs")

def stack(arrays, axis=0, **kwargs):
    """Stack arrays along a new axis. Wraps ``numpy.stack``. Cost: 0 FLOPs."""
    return _np.stack(arrays, axis=axis, **kwargs)

attach_docstring(stack, _np.stack, "free", "0 FLOPs")

def vstack(tup):
    """Stack arrays vertically. Wraps ``numpy.vstack``. Cost: 0 FLOPs."""
    return _np.vstack(tup)

attach_docstring(vstack, _np.vstack, "free", "0 FLOPs")

def hstack(tup):
    """Stack arrays horizontally. Wraps ``numpy.hstack``. Cost: 0 FLOPs."""
    return _np.hstack(tup)

attach_docstring(hstack, _np.hstack, "free", "0 FLOPs")

def split(ary, indices_or_sections, axis=0):
    """Split array. Wraps ``numpy.split``. Cost: 0 FLOPs."""
    return _np.split(ary, indices_or_sections, axis=axis)

attach_docstring(split, _np.split, "free", "0 FLOPs")

def hsplit(ary, indices_or_sections):
    """Split array horizontally. Wraps ``numpy.hsplit``. Cost: 0 FLOPs."""
    return _np.hsplit(ary, indices_or_sections)

attach_docstring(hsplit, _np.hsplit, "free", "0 FLOPs")

def vsplit(ary, indices_or_sections):
    """Split array vertically. Wraps ``numpy.vsplit``. Cost: 0 FLOPs."""
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
    """Flatten array. Wraps ``numpy.ravel``. Cost: 0 FLOPs."""
    return _np.ravel(a, **kwargs)

attach_docstring(ravel, _np.ravel, "free", "0 FLOPs")

def copy(a, **kwargs):
    """Return copy of array. Wraps ``numpy.copy``. Cost: 0 FLOPs."""
    return _np.copy(a, **kwargs)

attach_docstring(copy, _np.copy, "free", "0 FLOPs")

def where(condition, x=None, y=None):
    """Return elements chosen from *x* or *y*. Wraps ``numpy.where``. Cost: 0 FLOPs."""
    if x is None and y is None:
        return _np.where(condition)
    return _np.where(condition, x, y)

attach_docstring(where, _np.where, "free", "0 FLOPs")

def tile(A, reps):
    """Construct array by repeating. Wraps ``numpy.tile``. Cost: 0 FLOPs."""
    return _np.tile(A, reps)

attach_docstring(tile, _np.tile, "free", "0 FLOPs")

def repeat(a, repeats, axis=None):
    """Repeat elements. Wraps ``numpy.repeat``. Cost: 0 FLOPs."""
    return _np.repeat(a, repeats, axis=axis)

attach_docstring(repeat, _np.repeat, "free", "0 FLOPs")

def flip(m, axis=None):
    """Reverse order of elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    return _np.flip(m, axis=axis)

attach_docstring(flip, _np.flip, "free", "0 FLOPs")

def roll(a, shift, axis=None):
    """Roll array elements. Wraps ``numpy.roll``. Cost: 0 FLOPs."""
    return _np.roll(a, shift, axis=axis)

attach_docstring(roll, _np.roll, "free", "0 FLOPs")

def sort(a, axis=-1, **kwargs):
    """Return sorted copy. Wraps ``numpy.sort``. Cost: 0 FLOPs."""
    return _np.sort(a, axis=axis, **kwargs)

attach_docstring(sort, _np.sort, "free", "0 FLOPs")

def argsort(a, axis=-1, **kwargs):
    """Return indices that would sort. Wraps ``numpy.argsort``. Cost: 0 FLOPs."""
    return _np.argsort(a, axis=axis, **kwargs)

attach_docstring(argsort, _np.argsort, "free", "0 FLOPs")

def searchsorted(a, v, **kwargs):
    """Find insertion indices. Wraps ``numpy.searchsorted``. Cost: 0 FLOPs."""
    return _np.searchsorted(a, v, **kwargs)

attach_docstring(searchsorted, _np.searchsorted, "free", "0 FLOPs")

def unique(ar, **kwargs):
    """Find unique elements. Wraps ``numpy.unique``. Cost: 0 FLOPs."""
    return _np.unique(ar, **kwargs)

attach_docstring(unique, _np.unique, "free", "0 FLOPs")

def pad(array, pad_width, **kwargs):
    """Pad an array. Wraps ``numpy.pad``. Cost: 0 FLOPs."""
    return _np.pad(array, pad_width, **kwargs)

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
    """Return diagonal. Wraps ``numpy.diagonal``. Cost: 0 FLOPs."""
    return _np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)

attach_docstring(diagonal, _np.diagonal, "free", "0 FLOPs")

def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    """Return sum along diagonal. Wraps ``numpy.trace``. Cost: 0 FLOPs."""
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

attach_docstring(trace, _np.trace, "free", "0 FLOPs")

def broadcast_to(array, shape):
    """Broadcast array to shape. Wraps ``numpy.broadcast_to``. Cost: 0 FLOPs."""
    return _np.broadcast_to(array, shape)

attach_docstring(broadcast_to, _np.broadcast_to, "free", "0 FLOPs")

def meshgrid(*xi, **kwargs):
    """Return coordinate matrices. Wraps ``numpy.meshgrid``. Cost: 0 FLOPs."""
    return _np.meshgrid(*xi, **kwargs)

attach_docstring(meshgrid, _np.meshgrid, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Type / info helpers
# ---------------------------------------------------------------------------

def astype(x, dtype):
    """Cast array to *dtype*. Wraps ``x.astype(dtype)``. Cost: 0 FLOPs."""
    return x.astype(dtype)

def asarray(a, dtype=None, **kwargs):
    """Convert to array. Wraps ``numpy.asarray``. Cost: 0 FLOPs."""
    return _np.asarray(a, dtype=dtype, **kwargs)

attach_docstring(asarray, _np.asarray, "free", "0 FLOPs")

def isnan(x, **kwargs):
    """Test element-wise for NaN. Wraps ``numpy.isnan``. Cost: 0 FLOPs."""
    return _np.isnan(x, **kwargs)

attach_docstring(isnan, _np.isnan, "free", "0 FLOPs")

def isinf(x, **kwargs):
    """Test element-wise for Inf. Wraps ``numpy.isinf``. Cost: 0 FLOPs."""
    return _np.isinf(x, **kwargs)

attach_docstring(isinf, _np.isinf, "free", "0 FLOPs")

def isfinite(x, **kwargs):
    """Test element-wise for finiteness. Wraps ``numpy.isfinite``. Cost: 0 FLOPs."""
    return _np.isfinite(x, **kwargs)

attach_docstring(isfinite, _np.isfinite, "free", "0 FLOPs")

def allclose(a, b, **kwargs):
    """Check if all elements are close. Wraps ``numpy.allclose``. Cost: 0 FLOPs."""
    return _np.allclose(a, b, **kwargs)

attach_docstring(allclose, _np.allclose, "free", "0 FLOPs")
