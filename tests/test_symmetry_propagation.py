"""Tests for symmetry propagation through slices, reductions, and binary ops."""

import warnings

import numpy as np
import pytest

import mechestim as me
from mechestim._budget import BudgetContext
from mechestim._symmetric import (
    SymmetricTensor,
    as_symmetric,
    intersect_symmetry,
    propagate_symmetry_reduce,
    propagate_symmetry_slice,
)
from mechestim.errors import SymmetryLossWarning


def _sym4d(n=6):
    """Create a 4D fully-symmetric tensor with shape (n,n,n,n)."""
    data = np.random.RandomState(42).randn(n, n, n, n)
    # Symmetrize over all 4! permutations.
    from itertools import permutations

    sym = np.zeros_like(data)
    for perm in permutations(range(4)):
        sym += data.transpose(perm)
    sym /= 24
    return as_symmetric(sym, [(0, 1, 2, 3)])


def _sym2d(n=5):
    """Create a symmetric 2D matrix (n, n)."""
    data = np.random.RandomState(0).randn(n, n)
    data = (data + data.T) / 2
    return as_symmetric(data, (0, 1))


# -----------------------------------------------------------------------
# propagate_symmetry_slice (unit tests)
# -----------------------------------------------------------------------


class TestPropagateSymmetrySlice:
    def test_integer_index_removes_dim(self):
        # A[0] on ((0,1,2,3)) shape (6,6,6,6) → ((0,1,2)) on 3D
        result = propagate_symmetry_slice([(0, 1, 2, 3)], (6, 6, 6, 6), 0)
        assert result == [(0, 1, 2)]

    def test_integer_index_last_dim(self):
        # A[:,:,:,0] on ((0,1,2,3)) shape (6,6,6,6) → ((0,1,2))
        result = propagate_symmetry_slice(
            [(0, 1, 2, 3)], (6, 6, 6, 6), (slice(None), slice(None), slice(None), 0)
        )
        assert result == [(0, 1, 2)]

    def test_slice_same_size_preserves(self):
        # A[0:6] on ((0,1,2,3)) shape (6,6,6,6) → still ((0,1,2,3))
        result = propagate_symmetry_slice([(0, 1, 2, 3)], (6, 6, 6, 6), slice(0, 6))
        assert result == [(0, 1, 2, 3)]

    def test_slice_changes_size_pulls_dim(self):
        # A[0:3] on ((0,1,2,3)) shape (6,6,6,6) → dim 0 pulled → ((1,2,3))
        result = propagate_symmetry_slice([(0, 1, 2, 3)], (6, 6, 6, 6), slice(0, 3))
        assert result == [(1, 2, 3)]

    def test_two_slices_different_sizes_breaks_group(self):
        # A[0:3, 0:4] on ((0,1,2,3)) shape (6,6,6,6) → dims 0,1 pulled → ((2,3))
        result = propagate_symmetry_slice(
            [(0, 1, 2, 3)], (6, 6, 6, 6), (slice(0, 3), slice(0, 4))
        )
        assert result == [(2, 3)]

    def test_conservative_same_slice_both_dims(self):
        # Wilson's edge case: A[0:3, 0:3] — conservative, pulls both → ((2,3))
        result = propagate_symmetry_slice(
            [(0, 1, 2, 3)], (6, 6, 6, 6), (slice(0, 3), slice(0, 3))
        )
        # Both dims 0,1 are resized to 3, peers (dims 2,3) are size 6.
        # Mixed sizes → group broken. Only dims 2,3 survive.
        assert result == [(2, 3)]

    def test_advanced_indexing_strips_all(self):
        result = propagate_symmetry_slice([(0, 1)], (5, 5), np.array([0, 1, 2]))
        assert result is None

    def test_boolean_indexing_strips_all(self):
        mask = [True, False, True, False, True]
        result = propagate_symmetry_slice([(0, 1)], (5, 5), mask)
        assert result is None

    def test_ellipsis_expansion(self):
        # A[..., 0] on ((0,1,2,3)) shape (6,6,6,6) → removes dim 3 → ((0,1,2))
        result = propagate_symmetry_slice([(0, 1, 2, 3)], (6, 6, 6, 6), (Ellipsis, 0))
        assert result == [(0, 1, 2)]

    def test_multiple_groups_partial_survival(self):
        # ((0,1), (2,3)) on shape (5,5,4,4). A[0] removes dim 0.
        # Group (0,1) loses dim 0 → only dim 1 left → dropped.
        # Group (2,3) → renumbered to (1,2).
        result = propagate_symmetry_slice([(0, 1), (2, 3)], (5, 5, 4, 4), 0)
        assert result == [(1, 2)]

    def test_no_symmetry_survives_returns_none(self):
        # 2D ((0,1)). A[0] → 1D, group has only 1 dim → None.
        result = propagate_symmetry_slice([(0, 1)], (5, 5), 0)
        assert result is None

    def test_non_symmetric_dim_indexed(self):
        # ((0,1)) on (5,5,8). A[:,:,0] → shape (5,5). Group (0,1) intact.
        result = propagate_symmetry_slice(
            [(0, 1)], (5, 5, 8), (slice(None), slice(None), 0)
        )
        assert result == [(0, 1)]


# -----------------------------------------------------------------------
# propagate_symmetry_reduce (unit tests)
# -----------------------------------------------------------------------


class TestPropagateSymmetryReduce:
    def test_reduce_one_dim(self):
        # ((0,1,2,3)) reduce axis=0 → ((0,1,2))
        result = propagate_symmetry_reduce([(0, 1, 2, 3)], 4, axis=0)
        assert result == [(0, 1, 2)]

    def test_reduce_two_dims(self):
        # ((0,1,2,3)) reduce axis=(0,1) → ((0,1))
        result = propagate_symmetry_reduce([(0, 1, 2, 3)], 4, axis=(0, 1))
        assert result == [(0, 1)]

    def test_reduce_all_dims(self):
        # axis=None → scalar → None
        result = propagate_symmetry_reduce([(0, 1, 2, 3)], 4, axis=None)
        assert result is None

    def test_reduce_three_of_four(self):
        # ((0,1,2,3)) reduce (0,1,2) → 1D with dim 0, only 1 dim in group → None
        result = propagate_symmetry_reduce([(0, 1, 2, 3)], 4, axis=(0, 1, 2))
        assert result is None

    def test_keepdims_true(self):
        # ((0,1,2,3)) reduce axis=0, keepdims=True → dim 0 size 1, pulled.
        # Group becomes (1,2,3) on 4D output.
        result = propagate_symmetry_reduce([(0, 1, 2, 3)], 4, axis=0, keepdims=True)
        assert result == [(1, 2, 3)]

    def test_reduce_non_symmetric_dim(self):
        # ((0,1)) on 3D (5,5,8). reduce axis=2 → ((0,1)) still.
        result = propagate_symmetry_reduce([(0, 1)], 3, axis=2)
        assert result == [(0, 1)]

    def test_negative_axis(self):
        # ((0,1,2,3)) reduce axis=-1 (=3) → ((0,1,2))
        result = propagate_symmetry_reduce([(0, 1, 2, 3)], 4, axis=-1)
        assert result == [(0, 1, 2)]

    def test_multiple_groups(self):
        # ((0,1), (2,3)) reduce axis=0 → group (0,1) loses 0 → only dim 0(=1-1) → dropped.
        # group (2,3) renumbered to (1,2).
        result = propagate_symmetry_reduce([(0, 1), (2, 3)], 4, axis=0)
        assert result == [(1, 2)]


# -----------------------------------------------------------------------
# intersect_symmetry (unit tests)
# -----------------------------------------------------------------------


class TestIntersectSymmetry:
    def test_same_groups(self):
        result = intersect_symmetry(
            [(0, 1, 2)],
            [(0, 1, 2)],
            (5, 5, 5),
            (5, 5, 5),
            (5, 5, 5),
        )
        assert result == [(0, 1, 2)]

    def test_subset_groups(self):
        # A has ((0,1,2)), B has ((0,1)). Intersection = ((0,1)).
        result = intersect_symmetry(
            [(0, 1, 2)],
            [(0, 1)],
            (5, 5, 5),
            (5, 5),
            (5, 5, 5),
        )
        # B shape (5,5) gets aligned to output (5,5,5) as dims (1,2).
        # B's group (0,1) becomes (1,2) in output space.
        # A's group (0,1,2) vs B's (1,2) → no exact match.
        assert result is None

    def test_both_same_ndim_subset(self):
        # A has ((0,1,2)), B has ((0,1)). Both 3D.
        result = intersect_symmetry(
            [(0, 1, 2)],
            [(0, 1)],
            (5, 5, 5),
            (5, 5, 5),
            (5, 5, 5),
        )
        # A: (0,1,2). B: (0,1). Intersection needs exact match → None.
        assert result is None

    def test_multiple_groups_partial_match(self):
        # A: ((0,1), (2,3)). B: ((0,1)). Both 4D.
        result = intersect_symmetry(
            [(0, 1), (2, 3)],
            [(0, 1)],
            (5, 5, 4, 4),
            (5, 5, 4, 4),
            (5, 5, 4, 4),
        )
        assert result == [(0, 1)]

    def test_no_overlap(self):
        result = intersect_symmetry(
            [(0, 1)],
            [(2, 3)],
            (5, 5, 4, 4),
            (5, 5, 4, 4),
            (5, 5, 4, 4),
        )
        assert result is None

    def test_broadcast_breaks_symmetry(self):
        # A shape (5,5) with ((0,1)), B shape (1,5) → dim 0 of B stretched.
        # B has no groups that include dim 0 in output anyway.
        # A's ((0,1)) in output space is (0,1). B has no groups → intersection empty.
        result = intersect_symmetry(
            [(0, 1)],
            None,
            (5, 5),
            (1, 5),
            (5, 5),
        )
        assert result is None

    def test_one_none(self):
        result = intersect_symmetry(None, [(0, 1)], (5, 5), (5, 5), (5, 5))
        assert result is None


# -----------------------------------------------------------------------
# Integration tests: SymmetricTensor.__getitem__
# -----------------------------------------------------------------------


class TestSlicePropagation:
    def test_integer_index_propagates(self):
        st = _sym4d(6)
        result = st[0]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1, 2)]
        assert result.shape == (6, 6, 6)

    def test_full_slice_preserves(self):
        st = _sym4d(6)
        result = st[0:6]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1, 2, 3)]

    def test_partial_slice_propagates(self):
        st = _sym4d(6)
        result = st[0:3]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(1, 2, 3)]
        assert result.shape == (3, 6, 6, 6)

    def test_2d_integer_index_no_symmetry(self):
        st = _sym2d(5)
        result = st[0]
        assert not isinstance(result, SymmetricTensor)
        assert result.shape == (5,)

    def test_2d_full_slice_preserves(self):
        st = _sym2d(5)
        result = st[:]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_warning_on_symmetry_loss(self):
        st = _sym2d(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = st[0]
            sym_warnings = [x for x in w if issubclass(x.category, SymmetryLossWarning)]
            assert len(sym_warnings) >= 1

    def test_warning_suppressed_by_configure(self):
        st = _sym2d(5)
        me.configure(symmetry_warnings=False)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = st[0]
                sym_warnings = [
                    x for x in w if issubclass(x.category, SymmetryLossWarning)
                ]
                assert len(sym_warnings) == 0
        finally:
            me.configure(symmetry_warnings=True)

    def test_scalar_result_no_crash(self):
        st = _sym2d(5)
        result = st[0, 0]
        assert not isinstance(result, SymmetricTensor)


# -----------------------------------------------------------------------
# Integration tests: reductions with symmetry propagation
# -----------------------------------------------------------------------


class TestReductionPropagation:
    def test_sum_axis0_propagates(self):
        st = _sym4d(4)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.sum(st, axis=0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1, 2)]
        assert result.shape == (4, 4, 4)

    def test_sum_two_axes_propagates(self):
        st = _sym4d(4)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.sum(st, axis=(0, 1))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        assert result.shape == (4, 4)

    def test_sum_all_axes_no_symmetry(self):
        st = _sym4d(4)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.sum(st)
        assert not isinstance(result, SymmetricTensor)

    def test_sum_keepdims(self):
        st = _sym4d(4)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.sum(st, axis=0, keepdims=True)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(1, 2, 3)]
        assert result.shape == (1, 4, 4, 4)

    def test_mean_propagates(self):
        st = _sym4d(4)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.mean(st, axis=0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1, 2)]

    def test_sum_on_2d_axis0_loses_symmetry(self):
        st = _sym2d(5)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.sum(st, axis=0)
        # 1D result, group had 2 dims, one removed → only 1 left → no symmetry.
        assert not isinstance(result, SymmetricTensor)

    def test_reduction_warning_emitted(self):
        st = _sym2d(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with BudgetContext(flop_budget=10**8, quiet=True):
                _ = me.sum(st, axis=0)
            sym_warnings = [x for x in w if issubclass(x.category, SymmetryLossWarning)]
            assert len(sym_warnings) >= 1


# -----------------------------------------------------------------------
# Integration tests: binary ops with intersection
# -----------------------------------------------------------------------


class TestBinaryIntersection:
    def test_same_groups_preserved(self):
        A = _sym2d(5)
        B = _sym2d(5)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.add(A, B)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_scalar_preserves(self):
        A = _sym2d(5)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.multiply(A, np.asarray(3.0))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_no_shared_groups_strips(self):
        A = _sym2d(5)
        B = np.ones((5, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.add(A, B)
        assert not isinstance(result, SymmetricTensor)

    def test_partial_group_match(self):
        # A: ((0,1), (2,3)) on 4D. B: ((0,1)) on 4D.
        A_data = np.random.RandomState(0).randn(3, 3, 3, 3)
        A_sym = (A_data + A_data.transpose(1, 0, 2, 3)) / 2
        A_sym = (A_sym + A_sym.transpose(0, 1, 3, 2)) / 2
        A_st = as_symmetric(A_sym, [(0, 1), (2, 3)])

        B_data = np.random.RandomState(1).randn(3, 3, 3, 3)
        B_sym = (B_data + B_data.transpose(1, 0, 2, 3)) / 2
        B_st = as_symmetric(B_sym, [(0, 1)])

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.add(A_st, B_st)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]


# -----------------------------------------------------------------------
# Configuration tests
# -----------------------------------------------------------------------


class TestConfigure:
    def test_configure_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown setting"):
            me.configure(nonexistent_key=True)

    def test_configure_roundtrip(self):
        me.configure(symmetry_warnings=False)
        from mechestim._config import get_setting

        assert get_setting("symmetry_warnings") is False
        me.configure(symmetry_warnings=True)
        assert get_setting("symmetry_warnings") is True
