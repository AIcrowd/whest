"""Subclass of numpy.ndarray whose operators track FLOP usage.

This module defines WhestArray, a thin numpy.ndarray subclass
that overrides arithmetic, matmul, unary, comparison, bitwise, and
shift operators to route through whest's FLOP-counted me.*
functions.

Because WhestArray inherits from numpy.ndarray, isinstance(x,
numpy.ndarray) returns True. All me.* functions return WhestArray.
"""

from __future__ import annotations

import numpy as _np


def _me():
    """Lazy import of whest namespace to avoid circular imports.

    The dunder methods need me.add, me.multiply etc. but those are
    defined in whest/__init__.py which itself imports this module.
    Defer the import until first use.
    """
    import whest as _whest

    return _whest


class WhestArray(_np.ndarray):
    """A numpy ndarray subclass with FLOP-tracked operators.

    Behaves exactly like numpy.ndarray except that arithmetic and
    related operators route through whest's counted me.* functions
    so the active BudgetContext sees them.
    """

    def __new__(
        cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __array_finalize__(self, obj):
        # Called when numpy creates a view or slice of this subclass.
        # No subclass state to propagate.
        pass

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        """Honor numpy's `return_scalar` request from ufunc reductions.

        When a ufunc reduction collapses to a single value (e.g.
        ``np.bitwise_or.reduce(np.array([True], dtype=object))``), numpy
        passes ``return_scalar=True`` to ``__array_wrap__`` so the caller
        receives a Python scalar rather than a 0-d ndarray. The default
        ndarray behaviour respects this flag; we forward it explicitly so
        the same behaviour holds when the input is a WhestArray.

        For non-scalar results we let numpy preserve the subclass (the
        default behaviour) so views, slices, and ufunc outputs stay
        WhestArrays — keeping operator overloads and FLOP tracking
        intact for chained expressions.

        When the ufunc allocated a fresh output array (out_arr.owndata is
        True), numpy's view-casting to WhestArray loses the OWNDATA flag.
        We copy the result in that case so the WhestArray correctly reports
        OWNDATA=True, matching the flag semantics of the underlying data.
        """
        if return_scalar:
            return out_arr[()]
        result = super().__array_wrap__(out_arr, context, return_scalar)
        # numpy's view-cast drops OWNDATA; restore it when the ufunc
        # allocated a fresh buffer (out_arr owns its data).
        if (
            isinstance(result, WhestArray)
            and out_arr.flags.owndata
            and not result.flags.owndata
        ):
            result = result.copy(order='A')
        return result

    # ----- Binary arithmetic -----

    def __add__(self, other):
        return _me().add(self, other)

    def __radd__(self, other):
        return _me().add(other, self)

    def __iadd__(self, other):
        return _me().add(self, other)

    def __sub__(self, other):
        return _me().subtract(self, other)

    def __rsub__(self, other):
        return _me().subtract(other, self)

    def __isub__(self, other):
        return _me().subtract(self, other)

    def __mul__(self, other):
        return _me().multiply(self, other)

    def __rmul__(self, other):
        return _me().multiply(other, self)

    def __imul__(self, other):
        return _me().multiply(self, other)

    def __truediv__(self, other):
        return _me().true_divide(self, other)

    def __rtruediv__(self, other):
        return _me().true_divide(other, self)

    def __itruediv__(self, other):
        return _me().true_divide(self, other)

    def __floordiv__(self, other):
        return _me().floor_divide(self, other)

    def __rfloordiv__(self, other):
        return _me().floor_divide(other, self)

    def __ifloordiv__(self, other):
        return _me().floor_divide(self, other)

    def __mod__(self, other):
        return _me().mod(self, other)

    def __rmod__(self, other):
        return _me().mod(other, self)

    def __imod__(self, other):
        return _me().mod(self, other)

    def __pow__(self, other):
        return _me().power(self, other)

    def __rpow__(self, other):
        return _me().power(other, self)

    def __ipow__(self, other):
        return _me().power(self, other)

    def __matmul__(self, other):
        return _me().matmul(self, other)

    def __rmatmul__(self, other):
        return _me().matmul(other, self)

    def __imatmul__(self, other):
        return _me().matmul(self, other)

    # ----- Unary arithmetic -----

    def __neg__(self):
        return _me().negative(self)

    def __pos__(self):
        return _me().positive(self)

    def __abs__(self):
        return _me().abs(self)

    def __invert__(self):
        return _me().invert(self)

    # ----- Comparison -----

    def __eq__(self, other):
        return _me().equal(self, other)

    def __ne__(self, other):
        return _me().not_equal(self, other)

    def __lt__(self, other):
        return _me().less(self, other)

    def __le__(self, other):
        return _me().less_equal(self, other)

    def __gt__(self, other):
        return _me().greater(self, other)

    def __ge__(self, other):
        return _me().greater_equal(self, other)

    def __hash__(self):
        # numpy ndarray is unhashable; preserve that.
        raise TypeError(f"unhashable type: '{type(self).__name__}'")

    # ----- Bitwise -----

    def __and__(self, other):
        return _me().bitwise_and(self, other)

    def __rand__(self, other):
        return _me().bitwise_and(other, self)

    def __iand__(self, other):
        return _me().bitwise_and(self, other)

    def __or__(self, other):
        return _me().bitwise_or(self, other)

    def __ror__(self, other):
        return _me().bitwise_or(other, self)

    def __ior__(self, other):
        return _me().bitwise_or(self, other)

    def __xor__(self, other):
        return _me().bitwise_xor(self, other)

    def __rxor__(self, other):
        return _me().bitwise_xor(other, self)

    def __ixor__(self, other):
        return _me().bitwise_xor(self, other)

    def __lshift__(self, other):
        return _me().left_shift(self, other)

    def __rlshift__(self, other):
        return _me().left_shift(other, self)

    def __ilshift__(self, other):
        return _me().left_shift(self, other)

    def __rshift__(self, other):
        return _me().right_shift(self, other)

    def __rrshift__(self, other):
        return _me().right_shift(other, self)

    def __irshift__(self, other):
        return _me().right_shift(self, other)


def wrap_module_returns(module, skip_names=None, check_module=True):
    """Patch all public callables in a module to wrap return values.

    Walks the module's namespace, finds public functions defined in
    that module, and replaces them with wrappers that convert any
    numpy.ndarray return value into a WhestArray (zero-copy view).

    Tuple/list of arrays are also handled element-wise.

    Args:
        module: The module object to patch.
        skip_names: Optional set of function names to leave unwrapped
                    (e.g. functions that return scalars or shape tuples).
        check_module: If True (default), only wrap functions whose
                      __module__ matches the module being patched.
                      Set to False for modules that re-export from
                      sub-modules (e.g. whest.linalg).
    """
    import functools

    skip = set(skip_names or ())

    for name in list(vars(module)):
        if name.startswith("_") or name in skip:
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        if check_module:
            if not hasattr(obj, "__module__") or obj.__module__ != module.__name__:
                continue

        original = obj

        @functools.wraps(original)
        def wrapped(*args, _orig=original, **kwargs):
            result = _orig(*args, **kwargs)
            if isinstance(result, _np.ndarray):
                return _aswhest(result)
            if isinstance(result, tuple):
                wrapped_elems = [
                    _aswhest(r) if isinstance(r, _np.ndarray) else r for r in result
                ]
                # Preserve named tuple type (e.g. UniqueAllResult).
                if type(result) is not tuple and hasattr(type(result), "_fields"):
                    return type(result)(*wrapped_elems)
                return tuple(wrapped_elems)
            if isinstance(result, list):
                return [
                    _aswhest(r) if isinstance(r, _np.ndarray) else r for r in result
                ]
            return result

        wrapped.__wrapped__ = original

        setattr(module, name, wrapped)


def _aswhest(x):
    """Convert any array-like to WhestArray, preserving OWNDATA flag.

    - WhestArray: returned as-is
    - numpy.ndarray subclass (e.g. SymmetricTensor): returned as-is to
      preserve subclass metadata
    - plain numpy.ndarray that owns its data: copied so OWNDATA is
      preserved on the resulting WhestArray
    - plain numpy.ndarray view (OWNDATA=False): view-cast to WhestArray
      (zero-copy); OWNDATA remains False, which is correct for views
    - other: np.asarray first, then same logic as plain ndarray
    """
    if isinstance(x, WhestArray):
        return x
    if type(x) is not _np.ndarray and isinstance(x, _np.ndarray):
        # Other ndarray subclass (e.g. SymmetricTensor) — preserve as-is.
        return x
    if isinstance(x, _np.ndarray):
        result = x.view(WhestArray)
        if x.flags.owndata:
            result = result.copy(order='A')
        return result
    arr = _np.asarray(x)
    result = arr.view(WhestArray)
    if arr.flags.owndata:
        result = result.copy(order='A')
    return result
