"""Coverage-focused tests for whest._symmetric.

Targets uncovered lines to push coverage from ~74% toward ~95%.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from whest._perm_group import PermutationGroup
from whest._symmetric import (
    SymmetricTensor,
    SymmetryInfo,
    _warn_symmetry_loss,
    as_symmetric,
    intersect_symmetry,
    is_symmetric,
    propagate_symmetry_reduce,
    propagate_symmetry_slice,
    validate_symmetry,
    validate_symmetry_groups,
)
from whest.errors import SymmetryError, SymmetryLossWarning

# ============================================================================
# Helpers
# ============================================================================


def _sym_matrix(n: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2


def _sym_3d(n: int = 4, seed: int = 7) -> np.ndarray:
    """3D tensor symmetric under full permutation of all 3 axes."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n, n))
    # Symmetrise over all permutations of (0,1,2)
    from itertools import permutations

    result = sum(a.transpose(p) for p in permutations(range(3)))
    return result / 6.0


def _sym_4d(n1: int = 3, n2: int = 4, seed: int = 11) -> np.ndarray:
    """4D tensor with groups (0,1) and (2,3)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n1, n1, n2, n2))
    a = (a + a.transpose(1, 0, 2, 3)) / 2
    a = (a + a.transpose(0, 1, 3, 2)) / 2
    return a


# ============================================================================
# validate_symmetry (lines 100-114)
# ============================================================================


class TestValidateSymmetry:
    def test_valid_symmetric_matrix(self):
        """No error for genuinely symmetric data."""
        validate_symmetry(_sym_matrix(), [(0, 1)])

    def test_non_symmetric_raises(self):
        """Clearly non-symmetric data raises SymmetryError."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 5))
        with pytest.raises(SymmetryError) as exc_info:
            validate_symmetry(data, [(0, 1)])
        assert exc_info.value.max_deviation < float("inf")

    def test_unequal_sizes_raises(self):
        """Mismatched dimension sizes raise SymmetryError with inf deviation."""
        data = np.zeros((3, 4))
        with pytest.raises(SymmetryError) as exc_info:
            validate_symmetry(data, [(0, 1)])
        assert exc_info.value.max_deviation == float("inf")

    def test_close_to_symmetric_within_tolerance(self):
        """Deviation within atol/rtol passes."""
        data = _sym_matrix()
        data[0, 1] += 5e-7
        data[1, 0] -= 5e-7
        # Default tolerance is atol=1e-6, rtol=1e-5 so 1e-6 deviation passes
        validate_symmetry(data, [(0, 1)])

    def test_close_to_symmetric_exceeds_tolerance(self):
        """Deviation exceeding tolerance raises."""
        data = _sym_matrix()
        data[0, 1] += 0.01
        with pytest.raises(SymmetryError):
            validate_symmetry(data, [(0, 1)])

    def test_single_dim_group_skipped(self):
        """Groups with < 2 dims are skipped silently."""
        validate_symmetry(np.zeros((3,)), [(0,)])

    def test_three_way_symmetric(self):
        """Validates all 3 pairwise transpositions for a 3-way group."""
        data = _sym_3d()
        validate_symmetry(data, [(0, 1, 2)])

    def test_three_way_non_symmetric_pair(self):
        """Breaks symmetry between one pair in a 3-way group."""
        data = _sym_3d()
        data[0, 1, 2] += 1.0  # breaks (0,1) transposition
        with pytest.raises(SymmetryError):
            validate_symmetry(data, [(0, 1, 2)])

    def test_multiple_groups(self):
        data = _sym_4d()
        validate_symmetry(data, [(0, 1), (2, 3)])


# ============================================================================
# validate_symmetry_groups (lines 128-146)
# ============================================================================


class TestValidateSymmetryGroups:
    def test_valid_s2_group(self):
        data = _sym_matrix()
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        validate_symmetry_groups(data, [g])

    def test_no_axes_defaults_to_identity(self):
        """Group without axes defaults to axes=(0, 1, ..., degree-1)."""
        data = _sym_matrix()
        g = PermutationGroup.symmetric(2)  # no axes
        validate_symmetry_groups(data, [g])
        assert g.axes == (0, 1)

    def test_mismatched_orbit_sizes_raises(self):
        """Dimensions in same orbit with different sizes raise SymmetryError."""
        data = np.zeros((3, 4))
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        with pytest.raises(SymmetryError) as exc_info:
            validate_symmetry_groups(data, [g])
        assert exc_info.value.max_deviation == float("inf")

    def test_non_symmetric_data_raises(self):
        """Non-symmetric data fails generator check."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 4))
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        with pytest.raises(SymmetryError):
            validate_symmetry_groups(data, [g])

    def test_cyclic_group_valid(self):
        """Data symmetric under C3 passes validation with C3 group."""
        n = 3
        rng = np.random.default_rng(5)
        a = rng.standard_normal((n, n, n))
        sym = (a + a.transpose(1, 2, 0) + a.transpose(2, 0, 1)) / 3.0
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        validate_symmetry_groups(sym, [g])

    def test_cyclic_group_invalid(self):
        """Data not symmetric under C3 fails."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 3, 3))
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        with pytest.raises(SymmetryError):
            validate_symmetry_groups(data, [g])

    def test_identity_generator_skipped(self):
        """Identity generator is skipped in the loop."""
        # Trivial group on 1 element - only has the identity generator
        g = PermutationGroup.symmetric(1, axes=(0,))
        data = np.array([1.0, 2.0, 3.0])
        # Should pass - trivial group imposes no constraints
        validate_symmetry_groups(data, [g])

    def test_multiple_groups(self):
        data = _sym_4d()
        g1 = PermutationGroup.symmetric(2, axes=(0, 1))
        g2 = PermutationGroup.symmetric(2, axes=(2, 3))
        validate_symmetry_groups(data, [g1, g2])


# ============================================================================
# is_symmetric (lines 180-193)
# ============================================================================


class TestIsSymmetric:
    def test_symmetric_returns_true(self):
        assert is_symmetric(_sym_matrix(), (0, 1)) is True

    def test_non_symmetric_returns_false(self):
        rng = np.random.default_rng(0)
        assert is_symmetric(rng.standard_normal((4, 4)), (0, 1)) is False

    def test_unequal_sizes_returns_false(self):
        assert is_symmetric(np.zeros((3, 4)), (0, 1)) is False

    def test_single_tuple_input(self):
        """Single tuple (not list of tuples) is accepted."""
        assert is_symmetric(_sym_matrix(), (0, 1)) is True

    def test_list_of_tuples_input(self):
        data = _sym_4d()
        assert is_symmetric(data, [(0, 1), (2, 3)]) is True

    def test_single_dim_group_always_true(self):
        """A group with < 2 dims is trivially symmetric."""
        assert is_symmetric(np.array([1, 2, 3]), [(0,)]) is True

    def test_custom_tolerance(self):
        data = _sym_matrix()
        data[0, 1] += 0.01
        # Default tolerance rejects
        assert is_symmetric(data, (0, 1)) is False
        # Very large atol accepts
        assert is_symmetric(data, (0, 1), atol=0.1) is True

    def test_broadcasting_shape(self):
        """is_symmetric on a 3D tensor with broadcasting-like shape."""
        rng = np.random.default_rng(3)
        a = rng.standard_normal((2, 4, 4))
        a = (a + a.transpose(0, 2, 1)) / 2  # symmetric in (1,2)
        assert is_symmetric(a, (1, 2)) is True
        assert is_symmetric(a, (0, 1)) is False

    def test_method_on_symmetric_tensor(self):
        """SymmetricTensor.is_symmetric() method."""
        data = _sym_matrix()
        st = as_symmetric(data, (0, 1))
        assert st.is_symmetric() is True
        # With explicit axes
        assert st.is_symmetric((0, 1)) is True

    def test_three_way_partial_failure(self):
        """3-way group where one pair fails."""
        n = 3
        rng = np.random.default_rng(5)
        a = rng.standard_normal((n, n, n))
        # Only symmetric in (0,1), not full 3-way
        a = (a + a.transpose(1, 0, 2)) / 2
        assert is_symmetric(a, [(0, 1)]) is True
        assert is_symmetric(a, [(0, 1, 2)]) is False


# ============================================================================
# propagate_symmetry_slice (lines 245-362)
# ============================================================================


class TestPropagateSymmetrySlice:
    def test_full_slice_preserves(self):
        """Slicing with [:, :] preserves symmetry."""
        result = propagate_symmetry_slice([(0, 1)], (5, 5), (slice(None), slice(None)))
        assert result == [(0, 1)]

    def test_integer_index_removes_dim(self):
        """Integer index removes a dim from the group."""
        result = propagate_symmetry_slice([(0, 1)], (5, 5), (0,))
        # Dim 0 removed, only dim 1 survives -> group too small
        assert result is None

    def test_partial_slice_resizes(self):
        """Partial slice that changes size but keeps both dims."""
        result = propagate_symmetry_slice([(0, 1)], (5, 5), (slice(0, 3), slice(0, 3)))
        assert result == [(0, 1)]

    def test_partial_slice_different_sizes_breaks(self):
        """Slicing dims to different sizes breaks symmetry."""
        result = propagate_symmetry_slice([(0, 1)], (5, 5), (slice(0, 3), slice(0, 2)))
        # Dims have different sizes, group is lost
        assert result is None

    def test_ellipsis_expansion(self):
        """Ellipsis expands to fill missing dims."""
        result = propagate_symmetry_slice([(0, 1)], (5, 5), (Ellipsis,))
        assert result == [(0, 1)]

    def test_ellipsis_with_trailing_index(self):
        """Ellipsis + integer index at the end."""
        # shape (3, 5, 5), group (1, 2), key (..., 0)
        result = propagate_symmetry_slice([(1, 2)], (3, 5, 5), (Ellipsis, 0))
        # Dim 2 removed via integer, dim 1 survives -> group has 1 element
        assert result is None

    def test_double_ellipsis_raises(self):
        with pytest.raises(IndexError, match="only one Ellipsis"):
            propagate_symmetry_slice([(0, 1)], (5, 5), (Ellipsis, Ellipsis))

    def test_newaxis_handling(self):
        """np.newaxis adds a dim but doesn't consume an original dim."""
        # shape (5, 5), group (0, 1), key (None, :, :)
        result = propagate_symmetry_slice(
            [(0, 1)], (5, 5), (None, slice(None), slice(None))
        )
        # newaxis shifts dims: original 0->1, original 1->2
        assert result == [(1, 2)]

    def test_newaxis_between_symmetric_dims(self):
        """newaxis inserted between symmetric dims."""
        result = propagate_symmetry_slice(
            [(0, 1)], (5, 5), (slice(None), None, slice(None))
        )
        # Original 0->0, newaxis at 1, original 1->2
        assert result == [(0, 2)]

    def test_advanced_indexing_bails(self):
        """Array or list indexing returns None."""
        assert propagate_symmetry_slice([(0, 1)], (5, 5), (np.array([0, 1]),)) is None
        assert propagate_symmetry_slice([(0, 1)], (5, 5), ([0, 1],)) is None

    def test_non_tuple_key_normalized(self):
        """Non-tuple key is wrapped in a tuple."""
        result = propagate_symmetry_slice([(0, 1)], (5, 5), slice(None))
        # Single slice(None) -> only touches dim 0, rest padded
        assert result == [(0, 1)]

    def test_three_way_one_dim_removed(self):
        """Removing one dim from a 3-way group leaves a 2-way group."""
        # shape (4, 4, 4), group (0, 1, 2), key (0, :, :)
        result = propagate_symmetry_slice([(0, 1, 2)], (4, 4, 4), (0,))
        # Dim 0 removed, dims 1,2 become 0,1
        assert result == [(0, 1)]

    def test_multiple_groups_partial_survival(self):
        """One group survives, the other doesn't."""
        # shape (3, 3, 4, 4), groups (0,1) and (2,3)
        # Index dim 0 with int, keep dims 1,2,3
        result = propagate_symmetry_slice([(0, 1), (2, 3)], (3, 3, 4, 4), (0,))
        # Group (0,1): dim 0 removed -> only dim 1 survives -> lost
        # Group (2,3): both survive, remapped to (1, 2)
        assert result == [(1, 2)]

    def test_all_resized_picks_most_common(self):
        """When all dims are resized, picks the most common size."""
        # shape (10, 10, 10), group (0, 1, 2)
        # Slice all to size 3
        result = propagate_symmetry_slice(
            [(0, 1, 2)], (10, 10, 10), (slice(0, 3), slice(0, 3), slice(0, 3))
        )
        assert result == [(0, 1, 2)]

    def test_ellipsis_with_newaxis(self):
        """Ellipsis combined with newaxis."""
        # shape (5, 5), group (0,1), key (Ellipsis, None)
        result = propagate_symmetry_slice([(0, 1)], (5, 5), (Ellipsis, None))
        assert result == [(0, 1)]

    def test_step_slice(self):
        """Slice with step > 1 changes size."""
        # shape (6, 6), group (0,1), key (::2, ::2) -> size 3 each
        result = propagate_symmetry_slice(
            [(0, 1)], (6, 6), (slice(None, None, 2), slice(None, None, 2))
        )
        assert result == [(0, 1)]

    def test_step_slice_different(self):
        """Different step slices produce different sizes."""
        # shape (6, 6), key (::2, ::3) -> sizes 3, 2
        result = propagate_symmetry_slice(
            [(0, 1)], (6, 6), (slice(None, None, 2), slice(None, None, 3))
        )
        assert result is None


# ============================================================================
# propagate_symmetry_reduce (lines 385-410)
# ============================================================================


class TestPropagateSymmetryReduce:
    def test_reduce_none_axis(self):
        """axis=None reduces all dims -> no symmetry."""
        assert propagate_symmetry_reduce([(0, 1)], 2, None) is None

    def test_reduce_non_symmetric_axis_keepdims_false(self):
        """Reducing a non-symmetric axis with keepdims=False renumbers dims."""
        # shape (3, 5, 5), groups [(1,2)], reduce axis=0
        result = propagate_symmetry_reduce([(1, 2)], 3, 0, keepdims=False)
        # Dim 0 removed, dims 1,2 become 0,1
        assert result == [(0, 1)]

    def test_reduce_symmetric_axis_keepdims_false(self):
        """Reducing one symmetric axis breaks that group to 1 dim."""
        # shape (5, 5), groups [(0,1)], reduce axis=0
        result = propagate_symmetry_reduce([(0, 1)], 2, 0, keepdims=False)
        # Only dim 1 survives -> too small
        assert result is None

    def test_reduce_non_symmetric_axis_keepdims_true(self):
        """Reducing non-symmetric axis with keepdims=True keeps numbering."""
        # shape (3, 5, 5), groups [(1,2)], reduce axis=0, keepdims=True
        result = propagate_symmetry_reduce([(1, 2)], 3, 0, keepdims=True)
        assert result == [(1, 2)]

    def test_reduce_symmetric_axis_keepdims_true(self):
        """Reducing one symmetric axis with keepdims removes it from group."""
        # shape (5, 5, 5), groups [(0,1,2)], reduce axis=0, keepdims=True
        result = propagate_symmetry_reduce([(0, 1, 2)], 3, 0, keepdims=True)
        assert result == [(1, 2)]

    def test_reduce_all_symmetric_axes_keepdims_true(self):
        """Reducing all symmetric axes with keepdims=True -> no group."""
        result = propagate_symmetry_reduce([(0, 1)], 2, (0, 1), keepdims=True)
        assert result is None

    def test_reduce_tuple_axis(self):
        """Reducing multiple axes at once."""
        # shape (3, 4, 4, 3), groups [(1,2), (0,3)]
        result = propagate_symmetry_reduce([(1, 2), (0, 3)], 4, (0, 3), keepdims=False)
        # Dims 0,3 removed. Dims 1,2 become 0,1.
        assert result == [(0, 1)]

    def test_reduce_negative_axis(self):
        """Negative axis is normalized."""
        # ndim=3, axis=-1 -> axis=2
        result = propagate_symmetry_reduce([(0, 1)], 3, -1, keepdims=False)
        # Dim 2 removed, dims 0,1 unchanged
        assert result == [(0, 1)]

    def test_reduce_all_axes_keepdims_false(self):
        """Reduce both axes of a 2-group without keepdims."""
        result = propagate_symmetry_reduce([(0, 1)], 2, (0, 1), keepdims=False)
        assert result is None


# ============================================================================
# intersect_symmetry (lines 439-465)
# ============================================================================


class TestIntersectSymmetry:
    def test_both_none(self):
        assert intersect_symmetry(None, None, (3,), (3,), (3,)) is None

    def test_one_none(self):
        assert intersect_symmetry([(0, 1)], None, (3, 3), (3, 3), (3, 3)) is None
        assert intersect_symmetry(None, [(0, 1)], (3, 3), (3, 3), (3, 3)) is None

    def test_same_groups(self):
        """Identical groups produce the intersection."""
        result = intersect_symmetry([(0, 1)], [(0, 1)], (3, 3), (3, 3), (3, 3))
        assert result == [(0, 1)]

    def test_different_groups(self):
        """Non-overlapping groups produce empty intersection."""
        result = intersect_symmetry(
            [(0, 1)], [(2, 3)], (3, 3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3)
        )
        assert result is None

    def test_broadcast_alignment(self):
        """Shapes of different ndim are right-aligned."""
        # a: shape (3, 3), groups [(0,1)]
        # b: shape (1, 3, 3), groups [(1,2)]
        # output: (1, 3, 3)
        # a's dims 0,1 -> output dims 1,2
        # b's dims 1,2 -> output dims 1,2
        result = intersect_symmetry([(0, 1)], [(1, 2)], (3, 3), (1, 3, 3), (1, 3, 3))
        assert result == [(1, 2)]

    def test_broadcast_stretched_dim(self):
        """A dim that is broadcast-stretched (1 -> n) is removed from groups."""
        # a: shape (1, 3), group [(0,1)] - dim 0 is size 1
        # output: (3, 3) - dim 0 stretched from 1->3
        # b: shape (3, 3), group [(0,1)]
        result = intersect_symmetry([(0, 1)], [(0, 1)], (1, 3), (3, 3), (3, 3))
        # a's dim 0 is stretched, so (0,1) group loses dim 0 -> too small
        # b's (0,1) survives but a's doesn't match -> no intersection
        assert result is None

    def test_multiple_groups_partial_intersection(self):
        """Only matching groups survive intersection."""
        result = intersect_symmetry(
            [(0, 1), (2, 3)],
            [(0, 1)],
            (3, 3, 4, 4),
            (3, 3, 4, 4),
            (3, 3, 4, 4),
        )
        assert result == [(0, 1)]


# ============================================================================
# SymmetricTensor.__array_finalize__ (lines 507-519)
# ============================================================================


class TestArrayFinalize:
    def test_shape_mismatch_filters_invalid_groups(self):
        """When shape changes, invalid groups are filtered out."""
        data = _sym_matrix(5)
        st = as_symmetric(data, (0, 1))
        # Flatten changes shape so axes (0,1) no longer valid
        flat = st.ravel()
        # Should not carry symmetry through ravel
        if isinstance(flat, SymmetricTensor):
            assert flat.symmetric_axes == []
        else:
            assert isinstance(flat, np.ndarray)

    def test_view_same_shape_preserves(self):
        """View with same shape preserves metadata."""
        data = _sym_matrix(3)
        st = as_symmetric(data, (0, 1))
        v = st.view(SymmetricTensor)
        assert v.symmetric_axes == [(0, 1)]

    def test_reshape_invalidates(self):
        """Reshape to incompatible shape loses symmetry info."""
        data = _sym_matrix(4)
        st = as_symmetric(data, (0, 1))
        reshaped = st.reshape(2, 8)
        # (0,1) on shape (4,4) -> on (2,8) dim 0 has size 2, dim 1 has size 8
        # sizes differ -> group invalid
        if isinstance(reshaped, SymmetricTensor):
            assert reshaped.symmetric_axes == []

    def test_finalize_from_none(self):
        """__array_finalize__ with None obj (explicit construction)."""
        # Direct construction through __new__ without prior obj
        st = SymmetricTensor(np.eye(3), [(0, 1)])
        assert st.symmetric_axes == [(0, 1)]


# ============================================================================
# SymmetricTensor.__setstate__ / pickle (lines 623-647)
# ============================================================================


class TestPickle:
    def test_roundtrip_new_format(self):
        """Standard pickle roundtrip (new format with groups)."""
        data = _sym_matrix()
        st = as_symmetric(data, (0, 1))
        loaded = pickle.loads(pickle.dumps(st))
        assert isinstance(loaded, SymmetricTensor)
        assert loaded.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(loaded, st)
        # Groups should survive
        assert len(loaded.symmetry_info.groups) == 1

    def test_roundtrip_multiple_groups(self):
        data = _sym_4d()
        st = as_symmetric(data, [(0, 1), (2, 3)])
        loaded = pickle.loads(pickle.dumps(st))
        assert loaded.symmetric_axes == [(0, 1), (2, 3)]
        assert len(loaded.symmetry_info.groups) == 2

    def test_old_format_compat(self):
        """Simulate old pickle format where state[-1] is symmetric_axes (list of tuples)."""
        data = _sym_matrix()
        st = as_symmetric(data, (0, 1))
        # Get the normal reduce tuple
        reconstruct, args, state = st.__reduce__()
        # Old format: state = ndarray_state + (symmetric_axes,) -- no groups
        ndarray_state = state[:-2]
        old_state = ndarray_state + ([(0, 1)],)
        # Reconstruct and apply old state
        new_obj = reconstruct(*args)
        new_obj.__setstate__(old_state)
        assert isinstance(new_obj, SymmetricTensor)
        assert new_obj._symmetric_axes == [(0, 1)]
        # Groups should be auto-rebuilt
        assert len(new_obj._symmetry_groups) == 1
        assert isinstance(new_obj._symmetry_groups[0], PermutationGroup)

    def test_new_format_empty_groups(self):
        """New format with empty groups list."""
        data = np.array([1.0, 2.0, 3.0])
        st = SymmetricTensor(data, [])
        loaded = pickle.loads(pickle.dumps(st))
        assert isinstance(loaded, SymmetricTensor)
        assert loaded.symmetric_axes == []

    def test_pickle_highest_protocol(self):
        """Test with highest pickle protocol."""
        data = _sym_matrix(3)
        st = as_symmetric(data, (0, 1))
        loaded = pickle.loads(pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL))
        assert isinstance(loaded, SymmetricTensor)
        assert loaded.symmetric_axes == [(0, 1)]


# ============================================================================
# as_symmetric (lines 708-717): legacy path
# ============================================================================


class TestAsSymmetric:
    def test_single_tuple(self):
        """Single tuple is treated as one group."""
        data = _sym_matrix()
        st = as_symmetric(data, (0, 1))
        assert st.symmetric_axes == [(0, 1)]

    def test_list_of_tuples(self):
        data = _sym_4d()
        st = as_symmetric(data, [(0, 1), (2, 3)])
        assert st.symmetric_axes == [(0, 1), (2, 3)]

    def test_no_args_raises(self):
        """Neither symmetric_axes nor symmetry -> ValueError."""
        with pytest.raises(ValueError, match="Either symmetric_axes or symmetry"):
            as_symmetric(np.eye(3))

    def test_both_args_raises(self):
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        with pytest.raises(ValueError, match="mutually exclusive"):
            as_symmetric(np.eye(3), symmetric_axes=(0, 1), symmetry=g)

    def test_symmetry_single_group(self):
        """Pass a single PermutationGroup (not list)."""
        data = _sym_matrix(3)
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        st = as_symmetric(data, symmetry=g)
        assert isinstance(st, SymmetricTensor)

    def test_symmetry_list_of_groups(self):
        data = _sym_4d()
        groups = [
            PermutationGroup.symmetric(2, axes=(0, 1)),
            PermutationGroup.symmetric(2, axes=(2, 3)),
        ]
        st = as_symmetric(data, symmetry=groups)
        assert st.symmetric_axes == [(0, 1), (2, 3)]


# ============================================================================
# Slicing SymmetricTensor end-to-end
# ============================================================================


class TestSlicingEndToEnd:
    def test_ellipsis_slice(self):
        """Slicing with ellipsis preserves symmetry."""
        data = _sym_matrix(4)
        st = as_symmetric(data, (0, 1))
        result = st[...]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_newaxis_slice(self):
        """Slicing with np.newaxis."""
        data = _sym_matrix(4)
        st = as_symmetric(data, (0, 1))
        result = st[np.newaxis, :, :]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(1, 2)]

    def test_integer_index_loses_symmetry(self):
        data = _sym_matrix(4)
        st = as_symmetric(data, (0, 1))
        row = st[0]
        assert not isinstance(row, SymmetricTensor)

    def test_advanced_indexing_loses_symmetry(self):
        data = _sym_matrix(4)
        st = as_symmetric(data, (0, 1))
        result = st[np.array([0, 1])]
        assert not isinstance(result, SymmetricTensor)

    def test_scalar_result(self):
        """Indexing to a scalar returns plain value."""
        data = _sym_matrix(3)
        st = as_symmetric(data, (0, 1))
        val = st[0, 0]
        assert not isinstance(val, SymmetricTensor)

    def test_partial_slice_preserves(self):
        """Slicing both dims equally preserves symmetry."""
        data = _sym_matrix(6)
        st = as_symmetric(data, (0, 1))
        result = st[:3, :3]
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_4d_slice_one_group_lost(self):
        """Slicing that destroys one group but keeps another."""
        data = _sym_4d(4, 5)
        st = as_symmetric(data, [(0, 1), (2, 3)])
        # Remove dim 0 via integer index
        result = st[0]
        # Group (0,1) lost, group (2,3) remapped to (1,2)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(1, 2)]

    def test_no_symmetric_axes_returns_plain(self):
        """SymmetricTensor with no axes returns plain array on slice."""
        st = SymmetricTensor(np.arange(6).reshape(2, 3), [])
        result = st[:, 0]
        assert not isinstance(result, SymmetricTensor)


# ============================================================================
# Reductions with keepdims
# ============================================================================


class TestReductionsKeepdims:
    """Test propagate_symmetry_reduce keepdims branching via direct calls."""

    def test_keepdims_true_removes_from_group(self):
        result = propagate_symmetry_reduce([(0, 1, 2)], 3, 0, keepdims=True)
        assert result == [(1, 2)]

    def test_keepdims_false_renumbers(self):
        result = propagate_symmetry_reduce([(1, 2)], 3, 0, keepdims=False)
        assert result == [(0, 1)]

    def test_keepdims_true_non_symmetric_axis(self):
        result = propagate_symmetry_reduce([(0, 1)], 3, 2, keepdims=True)
        assert result == [(0, 1)]

    def test_keepdims_false_non_symmetric_axis(self):
        result = propagate_symmetry_reduce([(0, 1)], 3, 2, keepdims=False)
        assert result == [(0, 1)]


# ============================================================================
# _warn_symmetry_loss
# ============================================================================


class TestWarnSymmetryLoss:
    def test_warning_emitted_when_enabled(self):
        from whest._config import configure

        configure(symmetry_warnings=True)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _warn_symmetry_loss([(0, 1)], "test reason", stacklevel=1)
                assert len(w) == 1
                assert issubclass(w[0].category, SymmetryLossWarning)
        finally:
            configure(symmetry_warnings=False)

    def test_no_warning_when_disabled(self):
        from whest._config import configure

        configure(symmetry_warnings=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_symmetry_loss([(0, 1)], "test reason", stacklevel=1)
            symmetry_warnings = [
                x for x in w if issubclass(x.category, SymmetryLossWarning)
            ]
            assert len(symmetry_warnings) == 0


# ============================================================================
# SymmetricTensor.copy
# ============================================================================


class TestCopy:
    def test_copy_preserves_groups(self):
        data = _sym_matrix()
        st = as_symmetric(data, (0, 1))
        cp = st.copy()
        assert isinstance(cp, SymmetricTensor)
        assert cp.symmetric_axes == [(0, 1)]
        assert len(cp.symmetry_info.groups) == 1

    def test_copy_is_independent(self):
        data = _sym_matrix()
        st = as_symmetric(data, (0, 1))
        cp = st.copy()
        cp[0, 0] = 999.0
        assert st[0, 0] != 999.0


# ============================================================================
# SymmetryInfo edge cases
# ============================================================================


class TestSymmetryInfoEdgeCases:
    def test_no_groups(self):
        info = SymmetryInfo(symmetric_axes=[], shape=(3, 3))
        assert info.unique_elements == 9
        assert info.symmetry_factor == 1

    def test_groups_auto_built(self):
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(3, 3))
        assert info.groups is not None
        assert len(info.groups) == 1

    def test_groups_explicit(self):
        g = PermutationGroup.cyclic(2, axes=(0, 1))
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(3, 3), groups=[g])
        assert info.groups[0] is g

    def test_single_dim_group_ignored_for_auto_groups(self):
        """Groups with < 2 dims don't produce auto PermutationGroup."""
        info = SymmetryInfo(symmetric_axes=[(0,)], shape=(3,))
        assert len(info.groups) == 0


class TestPropagateSliceGeneralGroups:
    """Test propagate_symmetry_slice with non-S_k groups."""

    def test_c3_slice_one_axis_no_symmetry(self):
        """C_3 on {0,1,2}, slice axis 2 → no symmetry survives."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_slice([g], (5, 5, 5), (slice(None), slice(None), 0))
        assert result is None

    def test_c4_slice_two_axes_no_symmetry(self):
        """C_4 on {0,1,2,3}, slice axes {1,3} → pointwise stab of {1,3} = trivial."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        # Pointwise stabilizer: each of 1,3 must map to itself. Only identity does.
        result = propagate_symmetry_slice([g], (5, 5, 5, 5), (slice(None), 0, slice(None), 0))
        assert result is None

    def test_s3_slice_one_axis_s2_survives(self):
        """S_3 on {0,1,2}, slice axis 2 → S_2 on {0,1}. Same as old behavior."""
        g = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        result = propagate_symmetry_slice([g], (5, 5, 5), (slice(None), slice(None), 0))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2

    def test_d4_slice_one_axis(self):
        """D_4 on {0,1,2,3}, slice axis 0 → pointwise stab of {0}, restricted."""
        g = PermutationGroup.dihedral(4, axes=(0, 1, 2, 3))
        result = propagate_symmetry_slice([g], (5, 5, 5, 5), (0, slice(None), slice(None), slice(None)))
        assert result is not None
        assert len(result) == 1
        stab = result[0]
        assert stab.degree == 3
        for elem in stab.elements():
            assert elem.size == 3

    def test_multiple_groups_independent(self):
        """Two independent groups, slice removes axis from one."""
        g1 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        g2 = PermutationGroup.symmetric(2, axes=(3, 4))
        result = propagate_symmetry_slice(
            [g1, g2], (5, 5, 5, 5, 5), (0, slice(None), slice(None), slice(None), slice(None))
        )
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2

    def test_no_axes_removed_group_unchanged(self):
        """Full slice preserves group unchanged."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_slice([g], (5, 5, 5), (slice(None), slice(None), slice(None)))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3


class TestPropagateReduceGeneralGroups:
    """Test propagate_symmetry_reduce with non-S_k groups."""

    def test_c4_reduce_13_c2_survives(self):
        """C_4 on {0,1,2,3}, reduce axes {1,3} → C_2 on output {0,1}."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        result = propagate_symmetry_reduce([g], 4, (1, 3), keepdims=False)
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2
        assert result[0].axes == (0, 1)

    def test_c3_reduce_one_axis_trivial(self):
        """C_3 on {0,1,2}, reduce axis 2 → only identity survives → no group."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_reduce([g], 3, 2, keepdims=False)
        assert result is None

    def test_s3_reduce_one_axis_s2(self):
        """S_3 on {0,1,2}, reduce axis 2 → S_2 on {0,1}. Matches old behavior."""
        g = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        result = propagate_symmetry_reduce([g], 3, 2, keepdims=False)
        assert result is not None
        assert result[0].order() == 2

    def test_reduce_none_returns_none(self):
        """axis=None reduces everything → no symmetry."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_reduce([g], 3, None)
        assert result is None

    def test_c4_reduce_keepdims(self):
        """C_4 on {0,1,2,3}, reduce axes {1,3} keepdims=True → C_2."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        result = propagate_symmetry_reduce([g], 4, (1, 3), keepdims=True)
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2
        assert result[0].axes == (0, 2)

    def test_reduce_disjoint_axis(self):
        """Reduce an axis not in the group → group unchanged, axes renumbered."""
        g = PermutationGroup.cyclic(3, axes=(1, 2, 3))
        result = propagate_symmetry_reduce([g], 4, 0, keepdims=False)
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3


class TestIntersectSymmetryGeneralGroups:
    """Test intersect_symmetry with PermutationGroup objects."""

    def test_same_group_returns_group(self):
        """Intersecting a group with itself returns same group."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = intersect_symmetry([g], [g], (5, 5, 5), (5, 5, 5), (5, 5, 5))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3

    def test_s3_intersect_c3(self):
        """S_3 ∩ C_3 = C_3 (C_3 is a subgroup of S_3)."""
        s3 = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        c3 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = intersect_symmetry([s3], [c3], (5, 5, 5), (5, 5, 5), (5, 5, 5))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3

    def test_none_input_returns_none(self):
        """If either input is None, result is None."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        assert intersect_symmetry(None, [g], (5, 5, 5), (5, 5, 5), (5, 5, 5)) is None
        assert intersect_symmetry([g], None, (5, 5, 5), (5, 5, 5), (5, 5, 5)) is None

    def test_disjoint_axes_dropped(self):
        """Groups on different axes don't intersect → None."""
        g1 = PermutationGroup.symmetric(2, axes=(0, 1))
        g2 = PermutationGroup.symmetric(2, axes=(2, 3))
        result = intersect_symmetry([g1], [g2], (5, 5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5))
        assert result is None


class TestSymmetricTensorGeneralGroups:
    """End-to-end tests: SymmetricTensor with general groups through slice/reduce."""

    def test_c3_tensor_slice_loses_symmetry(self):
        """SymmetricTensor with C_3, slicing one axis -> no symmetry."""
        arr = np.zeros((3, 3, 3))
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        t = SymmetricTensor(arr, symmetric_axes=[(0, 1, 2)], perm_groups=[g])
        result = t[:, :, 0]
        assert not isinstance(result, SymmetricTensor) or not result.symmetric_axes

    def test_s3_tensor_slice_keeps_s2(self):
        """SymmetricTensor with S_3, slicing one axis -> S_2 survives."""
        arr = np.zeros((3, 3, 3))
        g = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        t = SymmetricTensor(arr, symmetric_axes=[(0, 1, 2)], perm_groups=[g])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = t[:, :, 0]
        assert isinstance(result, SymmetricTensor)
        assert len(result._symmetry_groups) == 1
        assert result._symmetry_groups[0].order() == 2
