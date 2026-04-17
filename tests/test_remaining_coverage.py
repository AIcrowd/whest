"""Tests targeting low-coverage modules to push total coverage toward 95%.

Covers uncovered branches in:
  - _counting_ops (array_equiv, histogram2d, histogramdd, apply_*, piecewise)
  - _validation (validate_ndarray, check_nan_inf, coerce_arrays)
  - _opt_einsum/_testing (build_shapes, build_views, rand_equation, build_arrays_from_tuples)
  - _ndarray (reverse ops, in-place ops, __array_wrap__)
  - _config (unknown setting)
  - _docstrings (attach_docstring with empty docstring)
  - random (dims sampler no-args, size-only sampler, choice, bytes, shuffle, __getattr__)
  - linalg/_solvers (inv with symmetric, tensorsolve, tensorinv, lstsq, pinv)
  - linalg/_decompositions (svdvals with k, eigvalsh, qr)
  - linalg/_properties (norm variants, cond with p, matrix_rank, vector_norm, matrix_norm)
  - linalg/_aliases (cross, outer, tensordot, vecdot, diagonal, matrix_transpose)
  - fft/_transforms (hfft, ihfft, irfft2, irfftn, fft2 with s, batch helpers)
  - _sorting_ops (lexsort, partition, argpartition, digitize, set ops)
  - _opt_einsum/_contract (memory_limit branches, PathInfo formatting)
  - _opt_einsum/_blas (BLAS classification edge cases)
"""

from __future__ import annotations

import warnings

import numpy
import numpy as np
import pytest

from whest._budget import BudgetContext

# ============================================================================
# _counting_ops
# ============================================================================


class TestArrayEquivBroadcastFail:
    """Cover the ValueError branch when broadcast_shapes fails (lines 79-80)."""

    def test_incompatible_shapes_returns_false(self):
        # Shapes (2,3) and (4,5) cannot broadcast
        a = numpy.ones((2, 3))
        b = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import array_equiv

            result = array_equiv(a, b)
        assert result is np.False_ or result == False  # noqa: E712
        # Cost should be max(a.size, b.size, 1) = max(6, 20, 1) = 20
        assert budget.flops_used == 20


class TestHistogram2dTupleBinsNonInt:
    """Cover lines 135-145: tuple bins with non-int arrays and scalar fallback."""

    def test_tuple_bins_with_arrays(self):
        x = numpy.random.rand(20)
        y = numpy.random.rand(20)
        bins_x = numpy.linspace(0, 1, 5)
        bins_y = numpy.linspace(0, 1, 8)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogram2d

            histogram2d(x, y, bins=[bins_x, bins_y])
        assert budget.flops_used > 0

    def test_tuple_bins_with_scalar_fallback(self):
        """Cover lines 142-143: scalar bin entries in tuple."""
        x = numpy.random.rand(10)
        y = numpy.random.rand(10)
        # Pass scalars (0-d arrays) as bins -- hits the else branch
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogram2d

            histogram2d(x, y, bins=[numpy.array(5), numpy.array(5)])
        assert budget.flops_used == 10  # fallback: max(n, 1) = 10

    def test_non_tuple_bins_fallback(self):
        """Cover line 144-145: bins is not int, not list/tuple of length 2."""
        x = numpy.random.rand(10)
        y = numpy.random.rand(10)
        bins_arr = numpy.linspace(0, 1, 6)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogram2d

            histogram2d(x, y, bins=bins_arr)
        assert budget.flops_used == 10


class TestHistogramddNonIntBins:
    """Cover lines 178-183, 186: non-int bins in list and else fallback."""

    def test_list_bins_with_arrays(self):
        sample = numpy.random.rand(15, 2)
        bins = [numpy.linspace(0, 1, 5), numpy.linspace(0, 1, 8)]
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogramdd

            histogramdd(sample, bins=bins)
        assert budget.flops_used > 0

    def test_list_bins_with_empty_array(self):
        """Cover line 182-183: empty bin array -> total_log += 1."""
        sample = numpy.random.rand(10, 1)
        bins = [numpy.array(5)]  # 0-d array, ndim=0 -> total_log += 1
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogramdd

            histogramdd(sample, bins=bins)
        assert budget.flops_used > 0

    def test_1d_sample(self):
        """Cover lines 166-167: 1-d sample."""
        sample = numpy.random.rand(20)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogramdd

            histogramdd(sample, bins=10)
        assert budget.flops_used > 0

    def test_non_list_bins_fallback(self):
        """Cover line 186: bins is not int and not list/tuple."""
        sample = numpy.random.rand(10, 2)
        bins_arr = numpy.array(5)  # scalar, not list/tuple, not int
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import histogramdd

            histogramdd(sample, bins=bins_arr)
        assert budget.flops_used == 10


class TestApplyAlongAxis:
    """Cover lines 273-281: apply_along_axis cost = result.size."""

    def test_basic(self):
        a = numpy.arange(12).reshape(3, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import apply_along_axis

            result = apply_along_axis(numpy.sum, 1, a)
        assert result.shape == (3,)
        assert budget.flops_used == 3

    def test_with_list_input(self):
        """Cover line 274-275: non-ndarray input conversion."""
        a = [[1, 2, 3], [4, 5, 6]]
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import apply_along_axis

            result = apply_along_axis(numpy.sum, 0, a)
        assert budget.flops_used == result.size


class TestApplyOverAxes:
    """Cover lines 294-300: apply_over_axes cost = result.size."""

    def test_basic(self):
        a = numpy.arange(24).reshape(2, 3, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import apply_over_axes

            result = apply_over_axes(numpy.sum, a, [0, 2])
        assert budget.flops_used == result.size

    def test_with_list_input(self):
        """Cover line 295-296: non-ndarray input conversion."""
        a = [[1, 2], [3, 4]]
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import apply_over_axes

            result = apply_over_axes(numpy.sum, a, [0])
        assert budget.flops_used == result.size


class TestPiecewise:
    """Cover lines 313-319: piecewise cost = x.size."""

    def test_basic(self):
        x = numpy.linspace(-2, 2, 10)
        condlist = [x < 0, x >= 0]
        funclist = [lambda x: -x, lambda x: x]
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import piecewise

            result = piecewise(x, condlist, funclist)
        assert budget.flops_used == 10
        numpy.testing.assert_array_equal(result, numpy.abs(x))

    def test_with_list_input(self):
        """Cover line 314-315: non-ndarray input conversion."""
        x = [1.0, -2.0, 3.0]
        condlist = [numpy.array([True, False, True]), numpy.array([False, True, False])]
        funclist = [lambda x: x * 2, lambda x: x * 3]
        with BudgetContext(flop_budget=10**6) as budget:
            from whest._counting_ops import piecewise

            piecewise(x, condlist, funclist)
        assert budget.flops_used == 3


# ============================================================================
# _validation
# ============================================================================


class TestValidateNdarray:
    """Cover lines 25-27: type validation raises TypeError."""

    def test_raises_for_list(self):
        from whest._validation import validate_ndarray

        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validate_ndarray([1, 2, 3])

    def test_raises_for_int(self):
        from whest._validation import validate_ndarray

        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validate_ndarray(42)

    def test_accepts_ndarray(self):
        from whest._validation import validate_ndarray

        validate_ndarray(numpy.array([1, 2, 3]))  # Should not raise


class TestCheckNanInf:
    """Cover lines 32-34, 39: NaN/Inf warnings and dtype skipping."""

    def test_float_with_nan(self):
        from whest._validation import check_nan_inf
        from whest.errors import WhestWarning

        arr = numpy.array([1.0, float("nan"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 1
        assert issubclass(w[0].category, WhestWarning)
        assert "1 NaN" in str(w[0].message)

    def test_float_with_inf(self):
        from whest._validation import check_nan_inf

        arr = numpy.array([1.0, float("inf"), -float("inf")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 1
        assert "2 Inf" in str(w[0].message)

    def test_complex_with_nan(self):
        from whest._validation import check_nan_inf

        arr = numpy.array([1 + 0j, float("nan") + 0j])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 1

    def test_integer_dtype_skipped(self):
        from whest._validation import check_nan_inf

        arr = numpy.array([1, 2, 3], dtype=numpy.int64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 0

    def test_object_dtype_skipped(self):
        from whest._validation import check_nan_inf

        arr = numpy.array(["a", "b", "c"], dtype=object)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(arr, "test_op")
        assert len(w) == 0

    def test_non_array_skipped(self):
        from whest._validation import check_nan_inf

        # Non-ndarray input should be silently skipped
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_inf(42, "test_op")
        assert len(w) == 0


class TestCoerceArrays:
    """Cover line 39: coerce_arrays tuple conversion."""

    def test_coerce_multiple(self):
        from whest._validation import coerce_arrays

        a, b = coerce_arrays([1, 2], numpy.array([3, 4]))
        assert isinstance(a, numpy.ndarray)
        assert isinstance(b, numpy.ndarray)

    def test_coerce_preserves_ndarray(self):
        from whest._validation import coerce_array

        arr = numpy.array([1, 2, 3])
        result = coerce_array(arr)
        assert result is arr  # same object, not a copy


# ============================================================================
# _opt_einsum/_testing
# ============================================================================


class TestBuildShapesEllipsis:
    """Cover lines 75-80: ellipsis validation."""

    def test_ellipsis_raises_without_flag(self):
        from whest._opt_einsum._testing import build_shapes

        with pytest.raises(ValueError, match="Ellipsis found"):
            build_shapes("...ab,...bc->...ac")

    def test_ellipsis_with_replace(self):
        from whest._opt_einsum._testing import build_shapes

        shapes = build_shapes("...ab,...bc->...ac", replace_ellipsis=True)
        assert len(shapes) == 2
        # Each shape should have 3 (ellipsis replacement) + 2 original dims = 5 dims
        assert len(shapes[0]) == 5
        assert len(shapes[1]) == 5


class TestBuildViewsScalar:
    """Cover lines 115-127: scalar array handling and default array_function."""

    def test_scalar_term(self):
        from whest._opt_einsum._testing import build_views

        # ",->" means one empty-shape term (scalar) and one empty output
        views = build_views(",a->a", {"a": 3})
        assert len(views) == 2
        # First view should be a scalar (float)
        assert isinstance(views[0], float)
        # Second view should be shape (3,)
        assert hasattr(views[1], "shape")
        assert views[1].shape == (3,)

    def test_default_array_function(self):
        from whest._opt_einsum._testing import build_views

        views = build_views("ab,bc->ac", {"a": 2, "b": 3, "c": 4})
        assert len(views) == 2
        assert views[0].shape == (2, 3)
        assert views[1].shape == (3, 4)


class TestRandEquation:
    """Cover lines 223-224, 245-249, 261: return_size_dict=True and global_dim=True."""

    def test_return_size_dict(self):
        from whest._opt_einsum._testing import rand_equation

        eq, shapes, size_dict = rand_equation(
            n=3, regularity=2, seed=42, return_size_dict=True
        )
        assert isinstance(size_dict, dict)
        assert len(size_dict) > 0
        assert isinstance(eq, str)
        assert "->" in eq

    def test_global_dim(self):
        from whest._opt_einsum._testing import rand_equation

        eq, shapes = rand_equation(n=3, regularity=2, seed=42, global_dim=True)
        # With global_dim, all operands should share a common index
        inputs = eq.split("->")[0].split(",")
        # Check all inputs have at least one common character
        common = set(inputs[0])
        for inp in inputs[1:]:
            common &= set(inp)
        assert len(common) >= 1

    def test_return_size_dict_and_global_dim(self):
        from whest._opt_einsum._testing import rand_equation

        eq, shapes, size_dict = rand_equation(
            n=4, regularity=3, n_out=2, seed=123, global_dim=True, return_size_dict=True
        )
        assert isinstance(size_dict, dict)
        assert len(shapes) == 4


class TestBuildArraysFromTuples:
    """Cover lines 275-277."""

    def test_basic(self):
        from whest._opt_einsum._testing import build_arrays_from_tuples

        path = [(2, 3), (4, 5)]
        arrays = build_arrays_from_tuples(path)
        assert len(arrays) == 2
        assert arrays[0].shape == (2, 3)
        assert arrays[1].shape == (4, 5)


# ============================================================================
# _ndarray (reverse and in-place operators, __array_wrap__)
# ============================================================================


class TestWhestArrayReverseOps:
    """Cover lines for __radd__, __rsub__, __rmul__, etc."""

    def test_radd(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 5.0 + arr  # triggers __radd__
            numpy.testing.assert_array_equal(result, [6.0, 7.0, 8.0])

    def test_rsub(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 10.0 - arr  # triggers __rsub__
            numpy.testing.assert_array_equal(result, [9.0, 8.0, 7.0])

    def test_rmul(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 2.0 * arr  # triggers __rmul__
            numpy.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_rtruediv(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 6.0 / arr  # triggers __rtruediv__
            numpy.testing.assert_array_equal(result, [6.0, 3.0, 2.0])

    def test_rfloordiv(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 7.0 // arr  # triggers __rfloordiv__
            numpy.testing.assert_array_equal(result, [7.0, 3.0, 2.0])

    def test_rmod(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 7.0 % arr  # triggers __rmod__
            numpy.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_rpow(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            result = 2.0**arr  # triggers __rpow__
            numpy.testing.assert_array_equal(result, [2.0, 4.0, 8.0])

    def test_rmatmul(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            a = we.array([[1.0, 2.0], [3.0, 4.0]])
            b = numpy.array([[5.0, 6.0], [7.0, 8.0]])
            result = b @ a  # triggers __rmatmul__ on a
            expected = numpy.array([[5.0, 6.0], [7.0, 8.0]]) @ numpy.array(
                [[1.0, 2.0], [3.0, 4.0]]
            )
            numpy.testing.assert_array_almost_equal(result, expected)


class TestWhestArrayInPlaceOps:
    """Cover lines for __iadd__, __isub__, __imul__, etc."""

    def test_iadd(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            arr += 1.0
            numpy.testing.assert_array_equal(arr, [2.0, 3.0, 4.0])

    def test_isub(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([10.0, 20.0, 30.0])
            arr -= 5.0
            numpy.testing.assert_array_equal(arr, [5.0, 15.0, 25.0])

    def test_imul(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1.0, 2.0, 3.0])
            arr *= 3.0
            numpy.testing.assert_array_equal(arr, [3.0, 6.0, 9.0])

    def test_itruediv(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([10.0, 20.0, 30.0])
            arr /= 10.0
            numpy.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_ifloordiv(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([10.0, 21.0, 35.0])
            arr //= 10.0
            numpy.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_imod(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([10.0, 21.0, 35.0])
            arr %= 10.0
            numpy.testing.assert_array_equal(arr, [0.0, 1.0, 5.0])

    def test_ipow(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([2.0, 3.0, 4.0])
            arr **= 2.0
            numpy.testing.assert_array_equal(arr, [4.0, 9.0, 16.0])

    def test_imatmul(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([[1.0, 0.0], [0.0, 1.0]])
            arr @= numpy.array([[2.0, 3.0], [4.0, 5.0]])
            numpy.testing.assert_array_equal(arr, [[2.0, 3.0], [4.0, 5.0]])


class TestWhestArrayBitwiseOps:
    """Cover reverse/in-place bitwise and shift operators."""

    def test_rand(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([0b1010, 0b1100], dtype=numpy.int32)
            result = numpy.int32(0b1111) & arr
            numpy.testing.assert_array_equal(result, [0b1010, 0b1100])

    def test_iand(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([0b1010, 0b1100], dtype=numpy.int32)
            arr &= numpy.int32(0b1010)
            numpy.testing.assert_array_equal(arr, [0b1010, 0b1000])

    def test_ror(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([0b1010, 0b0001], dtype=numpy.int32)
            result = numpy.int32(0b0100) | arr
            numpy.testing.assert_array_equal(result, [0b1110, 0b0101])

    def test_ior(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([0b1010, 0b0001], dtype=numpy.int32)
            arr |= numpy.int32(0b0100)
            numpy.testing.assert_array_equal(arr, [0b1110, 0b0101])

    def test_rxor(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([0b1010, 0b1100], dtype=numpy.int32)
            result = numpy.int32(0b1111) ^ arr
            numpy.testing.assert_array_equal(result, [0b0101, 0b0011])

    def test_ixor(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([0b1010, 0b1100], dtype=numpy.int32)
            arr ^= numpy.int32(0b1111)
            numpy.testing.assert_array_equal(arr, [0b0101, 0b0011])

    def test_rlshift(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1, 2], dtype=numpy.int32)
            result = numpy.int32(1).__lshift__(arr)
            # numpy.int32(1) << arr doesn't trigger __rlshift__,
            # but we can still test __lshift__ and __ilshift__
            arr2 = we.array([1, 2], dtype=numpy.int32)
            result2 = arr2 << numpy.int32(2)
            numpy.testing.assert_array_equal(result2, [4, 8])

    def test_ilshift(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([1, 2], dtype=numpy.int32)
            arr <<= numpy.int32(2)
            numpy.testing.assert_array_equal(arr, [4, 8])

    def test_rrshift(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([4, 8], dtype=numpy.int32)
            result = arr >> numpy.int32(1)
            numpy.testing.assert_array_equal(result, [2, 4])

    def test_irshift(self):
        import whest as we

        with BudgetContext(flop_budget=10**9):
            arr = we.array([4, 8], dtype=numpy.int32)
            arr >>= numpy.int32(1)
            numpy.testing.assert_array_equal(arr, [2, 4])


class TestArrayWrapReturnScalar:
    """Cover line 62-63: __array_wrap__ with return_scalar=True."""

    def test_return_scalar(self):
        from whest._ndarray import WhestArray

        arr = numpy.array([42.0]).view(WhestArray)
        # Calling __array_wrap__ with return_scalar=True should return a scalar
        result = arr.__array_wrap__(numpy.array(42.0), return_scalar=True)
        assert isinstance(result, (int, float, numpy.floating))

    def test_return_non_scalar(self):
        from whest._ndarray import WhestArray

        arr = numpy.array([1.0, 2.0]).view(WhestArray)
        out = numpy.array([3.0, 4.0])
        result = arr.__array_wrap__(out, return_scalar=False)
        assert isinstance(result, numpy.ndarray)


# ============================================================================
# _config
# ============================================================================


class TestConfigureUnknownSetting:
    """Cover line 26: unknown setting raises ValueError."""

    def test_unknown_setting(self):
        from whest._config import configure

        with pytest.raises(ValueError, match="Unknown setting"):
            configure(nonexistent_option=True)


# ============================================================================
# _docstrings
# ============================================================================


class TestAttachDocstringEmpty:
    """Cover lines 18-21: attach_docstring when np_func has empty docstring."""

    def test_empty_docstring(self):
        from whest._docstrings import attach_docstring

        def dummy():
            pass

        class FakeNpFunc:
            __name__ = "fake_func"
            __doc__ = ""

        attach_docstring(dummy, FakeNpFunc(), "test", "10 FLOPs")
        assert "FLOP Cost" in dummy.__doc__
        assert "fake_func" in dummy.__doc__

    def test_none_docstring(self):
        from whest._docstrings import attach_docstring

        def dummy():
            pass

        class FakeNpFunc:
            __name__ = "fake_func2"
            __doc__ = None

        attach_docstring(dummy, FakeNpFunc(), "test", "5 FLOPs")
        assert "FLOP Cost" in dummy.__doc__


# ============================================================================
# random
# ============================================================================


class TestRandomDimsSamplerNoArgs:
    """Cover line 79: dims sampler with no args (returns scalar)."""

    def test_rand_no_args(self):
        from whest import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.rand()
        assert isinstance(result, float)
        assert budget.flops_used == 1

    def test_randn_no_args(self):
        from whest import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.randn()
        assert isinstance(result, float)
        assert budget.flops_used == 1


class TestRandomSizeOnlySampler:
    """Cover lines 161-165: size-only samplers."""

    def test_random_with_size(self):
        from whest import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.random(size=(3, 4))
        assert result.shape == (3, 4)
        assert budget.flops_used == 12

    def test_random_no_size(self):
        from whest import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.random()
        assert budget.flops_used == 1


class TestRandomChoice:
    """Cover lines 223-234: choice with replace=False."""

    def test_choice_without_replacement(self):
        from whest import random as merandom
        from whest._flops import sort_cost

        n = 20
        with BudgetContext(flop_budget=10**6) as budget:
            merandom.choice(n, size=5, replace=False)
        assert budget.flops_used == sort_cost(n)

    def test_choice_from_array(self):
        from whest import random as merandom

        arr = numpy.arange(10)
        with BudgetContext(flop_budget=10**6) as budget:
            merandom.choice(arr, size=3, replace=True)
        assert budget.flops_used == 3


class TestRandomBytes:
    """Cover lines 242-245: bytes sampler."""

    def test_bytes(self):
        from whest import random as merandom

        with BudgetContext(flop_budget=10**6) as budget:
            result = merandom.bytes(16)
        assert len(result) == 16
        assert budget.flops_used == 16


class TestRandomShuffle:
    """Cover lines 207: shuffle with list input."""

    def test_shuffle_list(self):
        from whest import random as merandom

        lst = list(range(10))
        with BudgetContext(flop_budget=10**6) as budget:
            merandom.shuffle(lst)
        assert budget.flops_used > 0


class TestRandomGetattr:
    """Cover line 255: __getattr__ fallback for unknown attrs."""

    def test_getattr_existing(self):
        from whest import random as merandom

        # Access something that exists in numpy.random but not explicitly wrapped
        assert hasattr(merandom, "get_state")

    def test_getattr_missing(self):
        from whest import random as merandom

        with pytest.raises(AttributeError, match="does not provide"):
            _ = merandom.totally_nonexistent_attr_xyz


class TestRandomOutputSize:
    """Cover lines 35-37: _output_size with dims."""

    def test_output_size_with_dims(self):
        from whest.random import _output_size

        assert _output_size(3, 4, 5) == 60
        assert _output_size() == 1
        assert _output_size(size=(2, 3)) == 6
        assert _output_size(size=10) == 10


# ============================================================================
# linalg/_solvers
# ============================================================================


class TestLinalgSolversExtended:
    """Cover remaining linalg solver branches."""

    def test_inv_with_symmetric(self):
        from whest._symmetric import as_symmetric
        from whest.linalg._solvers import inv, inv_cost

        # Symmetric cost: n^3/3 + n^3
        n = 4
        a = numpy.eye(n) * 2.0
        sym_a = as_symmetric(a, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**9) as budget:
            result = inv(sym_a)
        expected_cost = inv_cost(n, symmetric=True)
        assert budget.flops_used == expected_cost
        assert expected_cost == max(n**3 // 3 + n**3, 1)

    def test_lstsq(self):
        from whest.linalg._solvers import lstsq

        a = numpy.random.rand(5, 3)
        b = numpy.random.rand(5)
        with BudgetContext(flop_budget=10**9) as budget:
            lstsq(a, b, rcond=None)
        assert budget.flops_used == 5 * 3 * 3  # m * n * min(m, n)

    def test_pinv(self):
        from whest.linalg._solvers import pinv

        a = numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            pinv(a)
        assert budget.flops_used == 4 * 6 * 4  # m * n * min(m, n)

    def test_tensorsolve(self):
        from whest.linalg._solvers import tensorsolve

        # a must be such that prod(a.shape[ind:]) == prod(b.shape)
        a = numpy.eye(24).reshape(6, 4, 6, 4)
        b = numpy.ones((6, 4))
        with BudgetContext(flop_budget=10**12) as budget:
            tensorsolve(a, b)
        assert budget.flops_used > 0

    def test_tensorinv(self):
        from whest.linalg._solvers import tensorinv

        a = numpy.eye(24).reshape(4, 6, 24)
        with BudgetContext(flop_budget=10**12) as budget:
            tensorinv(a, ind=2)
        assert budget.flops_used > 0


# ============================================================================
# linalg/_decompositions
# ============================================================================


class TestLinalgDecompositionsExtended:
    """Cover remaining decomposition branches."""

    def test_svdvals_with_k(self):
        from whest.linalg._decompositions import svdvals

        a = numpy.random.rand(6, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            result = svdvals(a, k=2)
        assert len(result) == 2
        assert budget.flops_used == 6 * 4 * 2

    def test_svdvals_invalid_k(self):
        from whest.linalg._decompositions import svdvals

        a = numpy.random.rand(4, 3)
        with BudgetContext(flop_budget=10**9):
            with pytest.raises(ValueError, match="k must satisfy"):
                svdvals(a, k=5)

    def test_eigvalsh(self):
        from whest.linalg._decompositions import eigvalsh

        a = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            eigvalsh(a)
        assert budget.flops_used == 8  # n^3 = 2^3

    def test_qr(self):
        from whest.linalg._decompositions import qr

        a = numpy.random.rand(5, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            qr(a)
        assert budget.flops_used == 5 * 3 * 3  # m * n * min(m, n)


# ============================================================================
# linalg/_properties
# ============================================================================


class TestLinalgPropertiesExtended:
    """Cover remaining norm, cond, and matrix_rank branches."""

    def test_norm_vector_general_ord(self):
        from whest.linalg._properties import norm

        x = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**9) as budget:
            norm(x, ord=3)
        # FMA=1: all vector norms cost numel
        assert budget.flops_used == 3

    def test_norm_matrix_2(self):
        from whest.linalg._properties import norm

        a = numpy.random.rand(4, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, ord=2)
        # SVD-based: 4 * m * n * min(m, n)
        assert budget.flops_used == 4 * 4 * 3 * 3

    def test_norm_matrix_nuc(self):
        from whest.linalg._properties import norm

        a = numpy.random.rand(3, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, ord="nuc")
        assert budget.flops_used == 4 * 3 * 5 * 3

    def test_norm_matrix_1(self):
        from whest.linalg._properties import norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, ord=1)
        assert budget.flops_used == 12  # numel

    def test_norm_with_axis_tuple(self):
        from whest.linalg._properties import norm

        a = numpy.random.rand(3, 4, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            norm(a, axis=(1, 2))
        assert budget.flops_used > 0

    def test_cond_with_p_1(self):
        from whest.linalg._properties import cond

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            cond(a, p=1)
        # LU-based: k^3 + m*n where k=min(m,n)=2
        assert budget.flops_used == 2**3 + 2 * 2

    def test_cond_with_p_inf(self):
        from whest.linalg._properties import cond

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            cond(a, p=numpy.inf)
        assert budget.flops_used == 2**3 + 2 * 2

    def test_matrix_rank(self):
        from whest.linalg._properties import matrix_rank

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_rank(a)
        assert budget.flops_used == 2 * 2 * 2  # m * n * min(m, n)

    def test_vector_norm(self):
        from whest.linalg._properties import vector_norm

        x = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**9) as budget:
            vector_norm(x, ord=3)
        # FMA=1: all norms cost numel
        assert budget.flops_used == 3

    def test_vector_norm_default(self):
        from whest.linalg._properties import vector_norm

        x = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**9) as budget:
            vector_norm(x)
        assert budget.flops_used == 2

    def test_vector_norm_with_axis(self):
        from whest.linalg._properties import vector_norm

        x = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            vector_norm(x, axis=1)
        assert budget.flops_used > 0

    def test_matrix_norm_fro(self):
        from whest.linalg._properties import matrix_norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_norm(a)
        # FMA=1: Frobenius norm costs numel
        assert budget.flops_used == 3 * 4

    def test_matrix_norm_2(self):
        from whest.linalg._properties import matrix_norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_norm(a, ord=2)
        assert budget.flops_used == 4 * 3 * 4 * 3

    def test_matrix_norm_nuc(self):
        from whest.linalg._properties import matrix_norm

        a = numpy.random.rand(3, 4)
        with BudgetContext(flop_budget=10**9) as budget:
            matrix_norm(a, ord="nuc")
        assert budget.flops_used == 4 * 3 * 4 * 3


# ============================================================================
# linalg/_aliases
# ============================================================================


class TestLinalgAliases:
    """Cover linalg alias functions that delegate to top-level whest."""

    def test_cross(self):
        from whest.linalg._aliases import cross

        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**9):
            result = cross(a, b)
        numpy.testing.assert_array_almost_equal(result, numpy.cross(a, b))

    def test_outer(self):
        from whest.linalg._aliases import outer

        a = numpy.array([1.0, 2.0])
        b = numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**9):
            result = outer(a, b)
        numpy.testing.assert_array_almost_equal(result, numpy.outer(a, b))

    def test_tensordot(self):
        from whest.linalg._aliases import tensordot

        a = numpy.random.rand(3, 4)
        b = numpy.random.rand(4, 5)
        with BudgetContext(flop_budget=10**9):
            result = tensordot(a, b, axes=1)
        numpy.testing.assert_array_almost_equal(result, numpy.tensordot(a, b, axes=1))

    def test_vecdot(self):
        from whest.linalg._aliases import vecdot

        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**9):
            result = vecdot(a, b)
        assert float(result) == pytest.approx(numpy.vecdot(a, b))

    def test_diagonal(self):
        from whest.linalg._aliases import diagonal

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9):
            result = diagonal(a)
        numpy.testing.assert_array_equal(result, [1.0, 4.0])

    def test_matrix_transpose(self):
        from whest.linalg._aliases import matrix_transpose

        a = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**9):
            result = matrix_transpose(a)
        numpy.testing.assert_array_equal(result, a.T)


# ============================================================================
# fft/_transforms
# ============================================================================


class TestFFTExtended:
    """Cover remaining FFT transform branches."""

    def test_hfft(self):
        from whest.fft._transforms import hfft

        a = numpy.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        with BudgetContext(flop_budget=10**9) as budget:
            hfft(a)
        assert budget.flops_used > 0

    def test_ihfft(self):
        from whest.fft._transforms import ihfft

        a = numpy.array([1.0, 2.0, 3.0, 4.0])
        with BudgetContext(flop_budget=10**9) as budget:
            ihfft(a)
        assert budget.flops_used > 0

    def test_irfft2(self):
        from whest.fft._transforms import irfft2

        a = numpy.random.rand(4, 3) + 1j * numpy.random.rand(4, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            irfft2(a)
        assert budget.flops_used > 0

    def test_irfftn(self):
        from whest.fft._transforms import irfftn

        a = numpy.random.rand(4, 3, 5) + 1j * numpy.random.rand(4, 3, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            irfftn(a)
        assert budget.flops_used > 0

    def test_irfftn_with_axes(self):
        from whest.fft._transforms import irfftn

        a = numpy.random.rand(4, 3, 5) + 1j * numpy.random.rand(4, 3, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            irfftn(a, axes=(0, 1, 2))
        assert budget.flops_used > 0

    def test_fft2_with_s(self):
        from whest.fft._transforms import fft2

        a = numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            fft2(a, s=(8, 8))
        assert budget.flops_used > 0

    def test_ifft2_with_s(self):
        from whest.fft._transforms import ifft2

        a = numpy.random.rand(4, 6) + 1j * numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            ifft2(a, s=(4, 6))
        assert budget.flops_used > 0

    def test_rfft2_with_s(self):
        from whest.fft._transforms import rfft2

        a = numpy.random.rand(4, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            rfft2(a, s=(8, 8))
        assert budget.flops_used > 0

    def test_irfft2_with_s(self):
        from whest.fft._transforms import irfft2

        a = numpy.random.rand(4, 5) + 1j * numpy.random.rand(4, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            irfft2(a, s=(4, 8))
        assert budget.flops_used > 0

    def test_fftn_with_s_and_axes(self):
        from whest.fft._transforms import fftn

        a = numpy.random.rand(4, 5, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            fftn(a, s=(4, 5), axes=(0, 1))
        assert budget.flops_used > 0

    def test_ifftn_with_s(self):
        from whest.fft._transforms import ifftn

        a = numpy.random.rand(4, 5) + 1j * numpy.random.rand(4, 5)
        with BudgetContext(flop_budget=10**9) as budget:
            ifftn(a, s=(4, 5))
        assert budget.flops_used > 0

    def test_rfftn_with_s(self):
        from whest.fft._transforms import rfftn

        a = numpy.random.rand(4, 5, 6)
        with BudgetContext(flop_budget=10**9) as budget:
            rfftn(a, s=(4, 5, 6))
        assert budget.flops_used > 0

    def test_irfftn_with_s(self):
        from whest.fft._transforms import irfftn

        a = numpy.random.rand(4, 5, 3) + 1j * numpy.random.rand(4, 5, 3)
        with BudgetContext(flop_budget=10**9) as budget:
            irfftn(a, s=(4, 5, 4))
        assert budget.flops_used > 0


# ============================================================================
# _sorting_ops
# ============================================================================


class TestSortingOpsExtended:
    """Cover remaining sorting/search branches."""

    def test_lexsort(self):
        from whest._sorting_ops import lexsort

        a = numpy.array([1, 2, 3, 1, 2])
        b = numpy.array([5, 4, 3, 2, 1])
        with BudgetContext(flop_budget=10**9) as budget:
            lexsort((a, b))
        assert budget.flops_used > 0

    def test_lexsort_single_key(self):
        from whest._sorting_ops import lexsort

        a = numpy.array([3, 1, 2])
        with BudgetContext(flop_budget=10**9) as budget:
            lexsort((a,))
        assert budget.flops_used > 0

    def test_partition(self):
        from whest._sorting_ops import partition

        a = numpy.array([3, 1, 2, 5, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            partition(a, 2)
        assert budget.flops_used > 0

    def test_partition_from_list(self):
        from whest._sorting_ops import partition

        a = [3, 1, 2, 5, 4]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            partition(a, 2)
        assert budget.flops_used > 0

    def test_argpartition(self):
        from whest._sorting_ops import argpartition

        a = numpy.array([3, 1, 2, 5, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            argpartition(a, 2)
        assert budget.flops_used > 0

    def test_argpartition_from_list(self):
        from whest._sorting_ops import argpartition

        a = [3, 1, 2, 5, 4]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            argpartition(a, 2)
        assert budget.flops_used > 0

    def test_digitize(self):
        from whest._sorting_ops import digitize

        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0, 10.0])
        with BudgetContext(flop_budget=10**9) as budget:
            digitize(x, bins)
        assert budget.flops_used > 0

    def test_setdiff1d(self):
        from whest._sorting_ops import setdiff1d

        a = numpy.array([1, 2, 3, 4, 5])
        b = numpy.array([3, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            result = setdiff1d(a, b)
        numpy.testing.assert_array_equal(result, [1, 2, 5])
        assert budget.flops_used > 0

    def test_setxor1d(self):
        from whest._sorting_ops import setxor1d

        a = numpy.array([1, 2, 3])
        b = numpy.array([2, 3, 4])
        with BudgetContext(flop_budget=10**9) as budget:
            result = setxor1d(a, b)
        numpy.testing.assert_array_equal(result, [1, 4])
        assert budget.flops_used > 0

    def test_sort_axis_none(self):
        from whest._sorting_ops import sort

        a = numpy.array([[3, 1], [2, 4]])
        with BudgetContext(flop_budget=10**9) as budget:
            sort(a, axis=None)
        assert budget.flops_used > 0

    def test_argsort_axis_none(self):
        from whest._sorting_ops import argsort

        a = numpy.array([[3, 1], [2, 4]])
        with BudgetContext(flop_budget=10**9) as budget:
            argsort(a, axis=None)
        assert budget.flops_used > 0

    def test_sort_from_list(self):
        from whest._sorting_ops import sort

        a = [3, 1, 2]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            sort(a)
        assert budget.flops_used > 0

    def test_argsort_from_list(self):
        from whest._sorting_ops import argsort

        a = [3, 1, 2]  # non-ndarray input triggers coercion
        with BudgetContext(flop_budget=10**9) as budget:
            argsort(a)
        assert budget.flops_used > 0


# ============================================================================
# _opt_einsum/_contract
# ============================================================================


class TestContractPathMemoryLimit:
    """Cover _choose_memory_arg branches."""

    def test_memory_limit_max_input(self):
        from whest._opt_einsum._contract import contract_path

        path, info = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            memory_limit="max_input",
        )
        assert len(path) > 0

    def test_memory_limit_explicit(self):
        from whest._opt_einsum._contract import contract_path

        path, info = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            memory_limit=100,
        )
        assert len(path) > 0

    def test_memory_limit_invalid_string(self):
        from whest._opt_einsum._contract import _choose_memory_arg

        with pytest.raises(ValueError, match="memory_limit must be"):
            _choose_memory_arg("invalid", [10, 20])

    def test_memory_limit_negative(self):
        from whest._opt_einsum._contract import _choose_memory_arg

        result = _choose_memory_arg(-1, [10, 20])
        assert result is None

    def test_memory_limit_negative_invalid(self):
        from whest._opt_einsum._contract import _choose_memory_arg

        with pytest.raises(ValueError, match="Memory limit must be larger"):
            _choose_memory_arg(-2, [10, 20])


class TestPathInfoFormatTable:
    """Cover PathInfo.format_table and __repr__."""

    def test_format_table(self):
        from whest._opt_einsum._contract import contract_path

        path, info = contract_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), shapes=True)
        table = info.format_table()
        assert "Naive cost" in table
        assert "Optimized cost" in table

    def test_format_table_verbose(self):
        from whest._opt_einsum._contract import contract_path

        path, info = contract_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), shapes=True)
        table = info.format_table(verbose=True)
        assert "subset=" in table
        assert "cumulative=" in table

    def test_repr(self):
        from whest._opt_einsum._contract import contract_path

        path, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
        repr_str = repr(info)
        assert "contraction" in repr_str.lower() or "cost" in repr_str.lower()

    def test_optimize_false(self):
        from whest._opt_einsum._contract import contract_path

        path, info = contract_path(
            "ij,jk->ik", (3, 4), (4, 5), shapes=True, optimize=False
        )
        assert len(path) == 1

    def test_opt_cost_property(self):
        from whest._opt_einsum._contract import contract_path

        _, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
        assert info.opt_cost >= 0

    def test_eq_property(self):
        from whest._opt_einsum._contract import contract_path

        _, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
        assert info.eq == "ij,jk->ik"


# ============================================================================
# _opt_einsum/_blas
# ============================================================================


def _make_sym_group(labels):
    """Create a symmetric PermutationGroup with the given labels."""
    from whest._perm_group import PermutationGroup

    pg = PermutationGroup.symmetric(len(labels))
    pg._labels = labels
    return pg


class TestBLASClassification:
    """Cover remaining BLAS classification branches."""

    def test_outer_einsum(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ab", "cd"], "abcd", set())
        assert result == "OUTER/EINSUM"

    def test_dot(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ij", "ij"], "", set("ij"))
        assert result == "DOT"

    def test_dot_einsum(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ij", "ji"], "", set("ij"))
        assert result == "DOT/EINSUM"

    def test_gemm_transpose_both(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ji", "ik"], "jk", set("i"))
        assert result == "GEMM"

    def test_gemm_transpose_right(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ij", "kj"], "ik", set("j"))
        assert result == "GEMM"

    def test_gemm_transpose_left(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ji", "jk"], "ik", set("j"))
        assert result == "GEMM"

    def test_gemv_einsum(self):
        from whest._opt_einsum._blas import can_blas

        # "j,ijk->ik": keep_left is empty, keep_right={i,k}
        # No GEMM pattern matches because removed index is not at edges
        result = can_blas(["j", "ijk"], "ik", set("j"))
        assert result == "GEMV/EINSUM"

    def test_tdot(self):
        from whest._opt_einsum._blas import can_blas

        # Removed indices not at edges of either input -> TDOT
        result = can_blas(["ikj", "jlk"], "il", set("jk"))
        assert result == "TDOT"

    def test_repeated_index_false(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ijj", "jk"], "ik", set("j"))
        assert result is False

    def test_three_inputs_false(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ij", "jk", "kl"], "il", set("jk"))
        assert result is False

    def test_broadcast_dims_false(self):
        from whest._opt_einsum._blas import can_blas

        result = can_blas(["ij", "jk"], "ik", set("j"), shapes=[(4, 1), (5, 6)])
        assert result is False

    def test_symm_classification(self):
        from whest._opt_einsum._blas import can_blas

        # Create a symmetric group on indices i,j for left input
        sym = _make_sym_group(("i", "j"))
        result = can_blas(["ij", "jk"], "ik", set("j"), input_groups=[sym, None])
        assert result == "SYMM"

    def test_symv_classification(self):
        from whest._opt_einsum._blas import can_blas

        sym = _make_sym_group(("j", "i", "k"))
        # "j,ijk->ik" is GEMV/EINSUM, with symmetric right input -> SYMV
        result = can_blas(["j", "ijk"], "ik", set("j"), input_groups=[None, sym])
        assert result == "SYMV"

    def test_sydt_classification(self):
        from whest._opt_einsum._blas import can_blas

        sym = _make_sym_group(("i", "j"))
        result = can_blas(["ij", "ij"], "", set("ij"), input_groups=[sym, None])
        assert result == "SYDT"
