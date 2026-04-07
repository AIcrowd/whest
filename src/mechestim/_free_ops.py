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


def isfinite(x, **kwargs):
    """Test element-wise for finiteness. Wraps ``numpy.isfinite``. Cost: 0 FLOPs."""
    return _np.isfinite(x, **kwargs)


attach_docstring(isfinite, _np.isfinite, "free", "0 FLOPs")


def isinf(x, **kwargs):
    """Test element-wise for Inf. Wraps ``numpy.isinf``. Cost: 0 FLOPs."""
    return _np.isinf(x, **kwargs)


attach_docstring(isinf, _np.isinf, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# New free ops
# ---------------------------------------------------------------------------


def append(*args, **kwargs):
    """Append values. Wraps ``numpy.append``. Cost: 0 FLOPs."""
    return _np.append(*args, **kwargs)


attach_docstring(append, _np.append, "free", "0 FLOPs")


def argwhere(*args, **kwargs):
    """Find indices of non-zero elements. Wraps ``numpy.argwhere``. Cost: 0 FLOPs."""
    return _np.argwhere(*args, **kwargs)


attach_docstring(argwhere, _np.argwhere, "free", "0 FLOPs")


def array_split(*args, **kwargs):
    """Split array into sub-arrays. Wraps ``numpy.array_split``. Cost: 0 FLOPs."""
    return _np.array_split(*args, **kwargs)


attach_docstring(array_split, _np.array_split, "free", "0 FLOPs")


def asarray_chkfinite(*args, **kwargs):
    """Convert to array checking for NaN/Inf. Wraps ``numpy.asarray_chkfinite``. Cost: 0 FLOPs."""
    return _np.asarray_chkfinite(*args, **kwargs)


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
    """Return string representation of number. Wraps ``numpy.base_repr``. Cost: 0 FLOPs."""
    return _np.base_repr(*args, **kwargs)


attach_docstring(base_repr, _np.base_repr, "free", "0 FLOPs")


def binary_repr(*args, **kwargs):
    """Return binary representation of integer. Wraps ``numpy.binary_repr``. Cost: 0 FLOPs."""
    return _np.binary_repr(*args, **kwargs)


attach_docstring(binary_repr, _np.binary_repr, "free", "0 FLOPs")


def block(*args, **kwargs):
    """Assemble array from nested lists. Wraps ``numpy.block``. Cost: 0 FLOPs."""
    return _np.block(*args, **kwargs)


attach_docstring(block, _np.block, "free", "0 FLOPs")


def bmat(*args, **kwargs):
    """Build matrix from string/nested sequence. Wraps ``numpy.bmat``. Cost: 0 FLOPs."""
    return _np.bmat(*args, **kwargs)


attach_docstring(bmat, _np.bmat, "free", "0 FLOPs")


def broadcast_arrays(*args, **kwargs):
    """Broadcast any number of arrays. Wraps ``numpy.broadcast_arrays``. Cost: 0 FLOPs."""
    return _np.broadcast_arrays(*args, **kwargs)


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
    """Construct array from index array. Wraps ``numpy.choose``. Cost: 0 FLOPs."""
    return _np.choose(*args, **kwargs)


attach_docstring(choose, _np.choose, "free", "0 FLOPs")


def column_stack(*args, **kwargs):
    """Stack 1-D arrays as columns. Wraps ``numpy.column_stack``. Cost: 0 FLOPs."""
    return _np.column_stack(*args, **kwargs)


attach_docstring(column_stack, _np.column_stack, "free", "0 FLOPs")


def common_type(*args, **kwargs):
    """Return scalar type common to input arrays. Wraps ``numpy.common_type``. Cost: 0 FLOPs."""
    return _np.common_type(*args, **kwargs)


attach_docstring(common_type, _np.common_type, "free", "0 FLOPs")


def compress(*args, **kwargs):
    """Return selected slices along an axis. Wraps ``numpy.compress``. Cost: 0 FLOPs."""
    return _np.compress(*args, **kwargs)


attach_docstring(compress, _np.compress, "free", "0 FLOPs")


def concat(*args, **kwargs):
    """Join arrays along an axis. Wraps ``numpy.concat``. Cost: 0 FLOPs."""
    return _np.concat(*args, **kwargs)


attach_docstring(concat, _np.concat, "free", "0 FLOPs")


def copyto(*args, **kwargs):
    """Copies values from one array to another. Wraps ``numpy.copyto``. Cost: 0 FLOPs."""
    return _np.copyto(*args, **kwargs)


attach_docstring(copyto, _np.copyto, "free", "0 FLOPs")


def delete(*args, **kwargs):
    """Return new array with sub-arrays deleted. Wraps ``numpy.delete``. Cost: 0 FLOPs."""
    return _np.delete(*args, **kwargs)


attach_docstring(delete, _np.delete, "free", "0 FLOPs")


def diag_indices(*args, **kwargs):
    """Return indices to access main diagonal. Wraps ``numpy.diag_indices``. Cost: 0 FLOPs."""
    return _np.diag_indices(*args, **kwargs)


attach_docstring(diag_indices, _np.diag_indices, "free", "0 FLOPs")


def diag_indices_from(*args, **kwargs):
    """Return indices to access main diagonal of array. Wraps ``numpy.diag_indices_from``. Cost: 0 FLOPs."""
    return _np.diag_indices_from(*args, **kwargs)


attach_docstring(diag_indices_from, _np.diag_indices_from, "free", "0 FLOPs")


def diagflat(*args, **kwargs):
    """Create diagonal array from flattened input. Wraps ``numpy.diagflat``. Cost: 0 FLOPs."""
    return _np.diagflat(*args, **kwargs)


attach_docstring(diagflat, _np.diagflat, "free", "0 FLOPs")


def dsplit(*args, **kwargs):
    """Split array along third axis. Wraps ``numpy.dsplit``. Cost: 0 FLOPs."""
    return _np.dsplit(*args, **kwargs)


attach_docstring(dsplit, _np.dsplit, "free", "0 FLOPs")


def dstack(*args, **kwargs):
    """Stack arrays along third axis. Wraps ``numpy.dstack``. Cost: 0 FLOPs."""
    return _np.dstack(*args, **kwargs)


attach_docstring(dstack, _np.dstack, "free", "0 FLOPs")


def extract(*args, **kwargs):
    """Return elements satisfying condition. Wraps ``numpy.extract``. Cost: 0 FLOPs."""
    return _np.extract(*args, **kwargs)


attach_docstring(extract, _np.extract, "free", "0 FLOPs")


def fill_diagonal(*args, **kwargs):
    """Fill main diagonal of array in-place. Wraps ``numpy.fill_diagonal``. Cost: 0 FLOPs."""
    return _np.fill_diagonal(*args, **kwargs)


attach_docstring(fill_diagonal, _np.fill_diagonal, "free", "0 FLOPs")


def flatnonzero(*args, **kwargs):
    """Return indices of non-zero elements in flattened array. Wraps ``numpy.flatnonzero``. Cost: 0 FLOPs."""
    return _np.flatnonzero(*args, **kwargs)


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
    """Create array from DLPack capsule. Wraps ``numpy.from_dlpack``. Cost: 0 FLOPs."""
    return _np.from_dlpack(*args, **kwargs)


attach_docstring(from_dlpack, _np.from_dlpack, "free", "0 FLOPs")


def frombuffer(*args, **kwargs):
    """Interpret buffer as 1-D array. Wraps ``numpy.frombuffer``. Cost: 0 FLOPs."""
    return _np.frombuffer(*args, **kwargs)


attach_docstring(frombuffer, _np.frombuffer, "free", "0 FLOPs")


def fromfile(*args, **kwargs):
    """Construct array from data in text or binary file. Wraps ``numpy.fromfile``. Cost: 0 FLOPs."""
    return _np.fromfile(*args, **kwargs)


attach_docstring(fromfile, _np.fromfile, "free", "0 FLOPs")


def fromfunction(*args, **kwargs):
    """Construct array by executing function over each coordinate. Wraps ``numpy.fromfunction``. Cost: 0 FLOPs."""
    return _np.fromfunction(*args, **kwargs)


attach_docstring(fromfunction, _np.fromfunction, "free", "0 FLOPs")


def fromiter(*args, **kwargs):
    """Create array from iterable object. Wraps ``numpy.fromiter``. Cost: 0 FLOPs."""
    return _np.fromiter(*args, **kwargs)


attach_docstring(fromiter, _np.fromiter, "free", "0 FLOPs")


def fromregex(*args, **kwargs):
    """Construct array from text file using regex. Wraps ``numpy.fromregex``. Cost: 0 FLOPs."""
    return _np.fromregex(*args, **kwargs)


attach_docstring(fromregex, _np.fromregex, "free", "0 FLOPs")


def fromstring(*args, **kwargs):
    """Construct array from string. Wraps ``numpy.fromstring``. Cost: 0 FLOPs."""
    return _np.fromstring(*args, **kwargs)


attach_docstring(fromstring, _np.fromstring, "free", "0 FLOPs")


def indices(*args, **kwargs):
    """Return array representing indices of a grid. Wraps ``numpy.indices``. Cost: 0 FLOPs."""
    return _np.indices(*args, **kwargs)


attach_docstring(indices, _np.indices, "free", "0 FLOPs")


def insert(*args, **kwargs):
    """Insert values along axis before given indices. Wraps ``numpy.insert``. Cost: 0 FLOPs."""
    return _np.insert(*args, **kwargs)


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
    """Construct open mesh from multiple sequences. Wraps ``numpy.ix_``. Cost: 0 FLOPs."""
    return _np.ix_(*args, **kwargs)


attach_docstring(ix_, _np.ix_, "free", "0 FLOPs")


def mask_indices(*args, **kwargs):
    """Return indices to access main or off-diagonal of array. Wraps ``numpy.mask_indices``. Cost: 0 FLOPs."""
    return _np.mask_indices(*args, **kwargs)


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


def nonzero(*args, **kwargs):
    """Return indices of non-zero elements. Wraps ``numpy.nonzero``. Cost: 0 FLOPs."""
    return _np.nonzero(*args, **kwargs)


attach_docstring(nonzero, _np.nonzero, "free", "0 FLOPs")


def packbits(*args, **kwargs):
    """Pack binary-valued array into bits. Wraps ``numpy.packbits``. Cost: 0 FLOPs."""
    return _np.packbits(*args, **kwargs)


attach_docstring(packbits, _np.packbits, "free", "0 FLOPs")


def permute_dims(*args, **kwargs):
    """Permute dimensions of array. Wraps ``numpy.permute_dims``. Cost: 0 FLOPs."""
    return _np.permute_dims(*args, **kwargs)


attach_docstring(permute_dims, _np.permute_dims, "free", "0 FLOPs")


def place(*args, **kwargs):
    """Change elements of array based on conditional. Wraps ``numpy.place``. Cost: 0 FLOPs."""
    return _np.place(*args, **kwargs)


attach_docstring(place, _np.place, "free", "0 FLOPs")


def promote_types(*args, **kwargs):
    """Return smallest size and least significant type. Wraps ``numpy.promote_types``. Cost: 0 FLOPs."""
    return _np.promote_types(*args, **kwargs)


attach_docstring(promote_types, _np.promote_types, "free", "0 FLOPs")


def put(*args, **kwargs):
    """Replace elements at given flat indices. Wraps ``numpy.put``. Cost: 0 FLOPs."""
    return _np.put(*args, **kwargs)


attach_docstring(put, _np.put, "free", "0 FLOPs")


def put_along_axis(*args, **kwargs):
    """Put values into destination array along axis. Wraps ``numpy.put_along_axis``. Cost: 0 FLOPs."""
    return _np.put_along_axis(*args, **kwargs)


attach_docstring(put_along_axis, _np.put_along_axis, "free", "0 FLOPs")


def putmask(*args, **kwargs):
    """Change elements of array based on condition. Wraps ``numpy.putmask``. Cost: 0 FLOPs."""
    return _np.putmask(*args, **kwargs)


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
    """Return new array with given shape. Wraps ``numpy.resize``. Cost: 0 FLOPs."""
    return _np.resize(*args, **kwargs)


attach_docstring(resize, _np.resize, "free", "0 FLOPs")


def result_type(*args, **kwargs):
    """Returns type that results from applying type promotion. Wraps ``numpy.result_type``. Cost: 0 FLOPs."""
    return _np.result_type(*args, **kwargs)


attach_docstring(result_type, _np.result_type, "free", "0 FLOPs")


def rollaxis(*args, **kwargs):
    """Roll specified axis backwards. Wraps ``numpy.rollaxis``. Cost: 0 FLOPs."""
    return _np.rollaxis(*args, **kwargs)


attach_docstring(rollaxis, _np.rollaxis, "free", "0 FLOPs")


def rot90(*args, **kwargs):
    """Rotate array 90 degrees. Wraps ``numpy.rot90``. Cost: 0 FLOPs."""
    return _np.rot90(*args, **kwargs)


attach_docstring(rot90, _np.rot90, "free", "0 FLOPs")


def row_stack(*args, **kwargs):
    """Stack arrays vertically (alias for vstack). Wraps ``numpy.row_stack``. Cost: 0 FLOPs."""
    return _np.row_stack(*args, **kwargs)


attach_docstring(row_stack, _np.row_stack, "free", "0 FLOPs")


def select(*args, **kwargs):
    """Return array drawn from elements depending on conditions. Wraps ``numpy.select``. Cost: 0 FLOPs."""
    return _np.select(*args, **kwargs)


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
    """Take elements from array along axis. Wraps ``numpy.take``. Cost: 0 FLOPs."""
    return _np.take(*args, **kwargs)


attach_docstring(take, _np.take, "free", "0 FLOPs")


def take_along_axis(*args, **kwargs):
    """Take values from input array along axis using indices. Wraps ``numpy.take_along_axis``. Cost: 0 FLOPs."""
    return _np.take_along_axis(*args, **kwargs)


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


def trim_zeros(*args, **kwargs):
    """Trim leading and/or trailing zeros from 1-D array. Wraps ``numpy.trim_zeros``. Cost: 0 FLOPs."""
    return _np.trim_zeros(*args, **kwargs)


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


def unpackbits(*args, **kwargs):
    """Unpack elements of uint8 array into binary-valued bit array. Wraps ``numpy.unpackbits``. Cost: 0 FLOPs."""
    return _np.unpackbits(*args, **kwargs)


attach_docstring(unpackbits, _np.unpackbits, "free", "0 FLOPs")


def unravel_index(*args, **kwargs):
    """Convert flat indices to multi-dimensional index. Wraps ``numpy.unravel_index``. Cost: 0 FLOPs."""
    return _np.unravel_index(*args, **kwargs)


attach_docstring(unravel_index, _np.unravel_index, "free", "0 FLOPs")


def unstack(*args, **kwargs):
    """Split array into sequence of arrays along an axis. Wraps ``numpy.unstack``. Cost: 0 FLOPs."""
    return _np.unstack(*args, **kwargs)


attach_docstring(unstack, _np.unstack, "free", "0 FLOPs")

# ---------------------------------------------------------------------------
# Wrap all free op return values as MechestimArray
# ---------------------------------------------------------------------------

from mechestim._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

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
