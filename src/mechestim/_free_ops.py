"""Zero-FLOP wrappers around NumPy tensor creation and manipulation.

Every function in this module delegates directly to the corresponding
NumPy function and costs **0 FLOPs**, so they work both inside and
outside a :class:`~mechestim._budget.BudgetContext`.
"""
from __future__ import annotations

import numpy as _np

# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def array(object, dtype=None, **kwargs):
    """Create an array. Wraps ``numpy.array``. Cost: 0 FLOPs."""
    return _np.array(object, dtype=dtype, **kwargs)

def zeros(shape, dtype=float, **kwargs):
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    return _np.zeros(shape, dtype=dtype, **kwargs)

def ones(shape, dtype=float, **kwargs):
    """Return array of ones. Wraps ``numpy.ones``. Cost: 0 FLOPs."""
    return _np.ones(shape, dtype=dtype, **kwargs)

def full(shape, fill_value, dtype=None, **kwargs):
    """Return array filled with *fill_value*. Wraps ``numpy.full``. Cost: 0 FLOPs."""
    return _np.full(shape, fill_value, dtype=dtype, **kwargs)

def eye(N, M=None, k=0, dtype=float, **kwargs):
    """Return identity matrix. Wraps ``numpy.eye``. Cost: 0 FLOPs."""
    return _np.eye(N, M=M, k=k, dtype=dtype, **kwargs)

def diag(v, k=0):
    """Extract diagonal or construct diagonal array. Wraps ``numpy.diag``. Cost: 0 FLOPs."""
    return _np.diag(v, k=k)

def arange(*args, **kwargs):
    """Return evenly spaced values. Wraps ``numpy.arange``. Cost: 0 FLOPs."""
    return _np.arange(*args, **kwargs)

def linspace(start, stop, num=50, **kwargs):
    """Return evenly spaced numbers. Wraps ``numpy.linspace``. Cost: 0 FLOPs."""
    return _np.linspace(start, stop, num=num, **kwargs)

def zeros_like(a, dtype=None, **kwargs):
    """Return array of zeros with same shape. Wraps ``numpy.zeros_like``. Cost: 0 FLOPs."""
    return _np.zeros_like(a, dtype=dtype, **kwargs)

def ones_like(a, dtype=None, **kwargs):
    """Return array of ones with same shape. Wraps ``numpy.ones_like``. Cost: 0 FLOPs."""
    return _np.ones_like(a, dtype=dtype, **kwargs)

def full_like(a, fill_value, dtype=None, **kwargs):
    """Return full array with same shape. Wraps ``numpy.full_like``. Cost: 0 FLOPs."""
    return _np.full_like(a, fill_value, dtype=dtype, **kwargs)

def empty(shape, dtype=float, **kwargs):
    """Return uninitialized array. Wraps ``numpy.empty``. Cost: 0 FLOPs."""
    return _np.empty(shape, dtype=dtype, **kwargs)

def empty_like(a, dtype=None, **kwargs):
    """Return uninitialized array with same shape. Wraps ``numpy.empty_like``. Cost: 0 FLOPs."""
    return _np.empty_like(a, dtype=dtype, **kwargs)

def identity(n, dtype=float):
    """Return identity matrix. Wraps ``numpy.identity``. Cost: 0 FLOPs."""
    return _np.identity(n, dtype=dtype)

# ---------------------------------------------------------------------------
# Tensor manipulation
# ---------------------------------------------------------------------------

def reshape(a, newshape, **kwargs):
    """Reshape an array. Wraps ``numpy.reshape``. Cost: 0 FLOPs."""
    return _np.reshape(a, newshape, **kwargs)

def transpose(a, axes=None):
    """Permute array dimensions. Wraps ``numpy.transpose``. Cost: 0 FLOPs."""
    return _np.transpose(a, axes=axes)

def swapaxes(a, axis1, axis2):
    """Swap two axes. Wraps ``numpy.swapaxes``. Cost: 0 FLOPs."""
    return _np.swapaxes(a, axis1, axis2)

def moveaxis(a, source, destination):
    """Move axes to new positions. Wraps ``numpy.moveaxis``. Cost: 0 FLOPs."""
    return _np.moveaxis(a, source, destination)

def concatenate(arrays, axis=0, **kwargs):
    """Join arrays along an axis. Wraps ``numpy.concatenate``. Cost: 0 FLOPs."""
    return _np.concatenate(arrays, axis=axis, **kwargs)

def stack(arrays, axis=0, **kwargs):
    """Stack arrays along a new axis. Wraps ``numpy.stack``. Cost: 0 FLOPs."""
    return _np.stack(arrays, axis=axis, **kwargs)

def vstack(tup):
    """Stack arrays vertically. Wraps ``numpy.vstack``. Cost: 0 FLOPs."""
    return _np.vstack(tup)

def hstack(tup):
    """Stack arrays horizontally. Wraps ``numpy.hstack``. Cost: 0 FLOPs."""
    return _np.hstack(tup)

def split(ary, indices_or_sections, axis=0):
    """Split array. Wraps ``numpy.split``. Cost: 0 FLOPs."""
    return _np.split(ary, indices_or_sections, axis=axis)

def hsplit(ary, indices_or_sections):
    """Split array horizontally. Wraps ``numpy.hsplit``. Cost: 0 FLOPs."""
    return _np.hsplit(ary, indices_or_sections)

def vsplit(ary, indices_or_sections):
    """Split array vertically. Wraps ``numpy.vsplit``. Cost: 0 FLOPs."""
    return _np.vsplit(ary, indices_or_sections)

def squeeze(a, axis=None):
    """Remove length-1 axes. Wraps ``numpy.squeeze``. Cost: 0 FLOPs."""
    return _np.squeeze(a, axis=axis)

def expand_dims(a, axis):
    """Insert a new axis. Wraps ``numpy.expand_dims``. Cost: 0 FLOPs."""
    return _np.expand_dims(a, axis=axis)

def ravel(a, **kwargs):
    """Flatten array. Wraps ``numpy.ravel``. Cost: 0 FLOPs."""
    return _np.ravel(a, **kwargs)

def copy(a, **kwargs):
    """Return copy of array. Wraps ``numpy.copy``. Cost: 0 FLOPs."""
    return _np.copy(a, **kwargs)

def where(condition, x=None, y=None):
    """Return elements chosen from *x* or *y*. Wraps ``numpy.where``. Cost: 0 FLOPs."""
    if x is None and y is None:
        return _np.where(condition)
    return _np.where(condition, x, y)

def tile(A, reps):
    """Construct array by repeating. Wraps ``numpy.tile``. Cost: 0 FLOPs."""
    return _np.tile(A, reps)

def repeat(a, repeats, axis=None):
    """Repeat elements. Wraps ``numpy.repeat``. Cost: 0 FLOPs."""
    return _np.repeat(a, repeats, axis=axis)

def flip(m, axis=None):
    """Reverse order of elements. Wraps ``numpy.flip``. Cost: 0 FLOPs."""
    return _np.flip(m, axis=axis)

def roll(a, shift, axis=None):
    """Roll array elements. Wraps ``numpy.roll``. Cost: 0 FLOPs."""
    return _np.roll(a, shift, axis=axis)

def sort(a, axis=-1, **kwargs):
    """Return sorted copy. Wraps ``numpy.sort``. Cost: 0 FLOPs."""
    return _np.sort(a, axis=axis, **kwargs)

def argsort(a, axis=-1, **kwargs):
    """Return indices that would sort. Wraps ``numpy.argsort``. Cost: 0 FLOPs."""
    return _np.argsort(a, axis=axis, **kwargs)

def searchsorted(a, v, **kwargs):
    """Find insertion indices. Wraps ``numpy.searchsorted``. Cost: 0 FLOPs."""
    return _np.searchsorted(a, v, **kwargs)

def unique(ar, **kwargs):
    """Find unique elements. Wraps ``numpy.unique``. Cost: 0 FLOPs."""
    return _np.unique(ar, **kwargs)

def pad(array, pad_width, **kwargs):
    """Pad an array. Wraps ``numpy.pad``. Cost: 0 FLOPs."""
    return _np.pad(array, pad_width, **kwargs)

def triu(m, k=0):
    """Upper triangle. Wraps ``numpy.triu``. Cost: 0 FLOPs."""
    return _np.triu(m, k=k)

def tril(m, k=0):
    """Lower triangle. Wraps ``numpy.tril``. Cost: 0 FLOPs."""
    return _np.tril(m, k=k)

def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return diagonal. Wraps ``numpy.diagonal``. Cost: 0 FLOPs."""
    return _np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)

def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    """Return sum along diagonal. Wraps ``numpy.trace``. Cost: 0 FLOPs."""
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

def broadcast_to(array, shape):
    """Broadcast array to shape. Wraps ``numpy.broadcast_to``. Cost: 0 FLOPs."""
    return _np.broadcast_to(array, shape)

def meshgrid(*xi, **kwargs):
    """Return coordinate matrices. Wraps ``numpy.meshgrid``. Cost: 0 FLOPs."""
    return _np.meshgrid(*xi, **kwargs)

# ---------------------------------------------------------------------------
# Type / info helpers
# ---------------------------------------------------------------------------

def astype(x, dtype):
    """Cast array to *dtype*. Wraps ``x.astype(dtype)``. Cost: 0 FLOPs."""
    return x.astype(dtype)

def asarray(a, dtype=None, **kwargs):
    """Convert to array. Wraps ``numpy.asarray``. Cost: 0 FLOPs."""
    return _np.asarray(a, dtype=dtype, **kwargs)

def isnan(x, **kwargs):
    """Test element-wise for NaN. Wraps ``numpy.isnan``. Cost: 0 FLOPs."""
    return _np.isnan(x, **kwargs)

def isinf(x, **kwargs):
    """Test element-wise for Inf. Wraps ``numpy.isinf``. Cost: 0 FLOPs."""
    return _np.isinf(x, **kwargs)

def isfinite(x, **kwargs):
    """Test element-wise for finiteness. Wraps ``numpy.isfinite``. Cost: 0 FLOPs."""
    return _np.isfinite(x, **kwargs)

def allclose(a, b, **kwargs):
    """Check if all elements are close. Wraps ``numpy.allclose``. Cost: 0 FLOPs."""
    return _np.allclose(a, b, **kwargs)
