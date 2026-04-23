"""Focused coverage tests for current symmetry tensor helpers."""

from __future__ import annotations

import numpy as np
import pytest

from whest._perm_group import SymmetryGroup
from whest._symmetric import (
    SymmetricTensor,
    as_symmetric,
    intersect_symmetry,
    is_symmetric,
    propagate_symmetry_reduce,
    propagate_symmetry_slice,
    symmetrize,
    validate_symmetry_groups,
)
from whest.errors import SymmetryError


def _sg(*axes: int) -> SymmetryGroup:
    return SymmetryGroup.symmetric(axes=axes)


def _assert_groups_equal(result, expected_axes_list):
    assert result is not None
    assert [group.axes for group in result] == [
        tuple(axes) for axes in expected_axes_list
    ]


def _sym_matrix(n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, n))
    return (data + data.T) / 2


def _sym_4d() -> np.ndarray:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 3, 4, 4))
    data = (data + data.transpose(1, 0, 2, 3)) / 2
    data = (data + data.transpose(0, 1, 3, 2)) / 2
    return data


class TestValidateSymmetryGroups:
    def test_valid_s2_group(self):
        validate_symmetry_groups(_sym_matrix(), [_sg(0, 1)])

    def test_cyclic_group_valid(self):
        n = 3
        rng = np.random.default_rng(5)
        a = rng.standard_normal((n, n, n))
        sym = (a + a.transpose(1, 2, 0) + a.transpose(2, 0, 1)) / 3.0
        validate_symmetry_groups(sym, [SymmetryGroup.cyclic(axes=(0, 1, 2))])

    def test_invalid_data_raises(self):
        with pytest.raises(SymmetryError):
            validate_symmetry_groups(
                np.random.default_rng(0).standard_normal((4, 4)), [_sg(0, 1)]
            )

    def test_mismatched_orbit_sizes_raise(self):
        with pytest.raises(SymmetryError):
            validate_symmetry_groups(np.zeros((3, 4)), [_sg(0, 1)])


class TestIsSymmetricAndWrapping:
    def test_is_symmetric_accepts_group_and_shorthand(self):
        data = _sym_matrix()
        assert is_symmetric(data, symmetry=_sg(0, 1)) is True
        assert is_symmetric(data, symmetry=(0, 1)) is True
        assert is_symmetric(data, symmetry=((0, 1),)) is True

    def test_as_symmetric_and_symmetrize_expose_only_symmetry(self):
        data = _sym_matrix()
        tensor = as_symmetric(data, symmetry=(0, 1))
        projected = symmetrize(np.arange(16.0).reshape(4, 4), symmetry=_sg(0, 1))
        assert isinstance(tensor, SymmetricTensor)
        assert tensor.symmetry == _sg(0, 1)
        assert not hasattr(tensor, "symmetry" + "_info")
        assert not hasattr(tensor, "symmetric_" + "axes")
        assert projected.symmetry == _sg(0, 1)

    def test_as_symmetric_rejects_non_symmetric_input(self):
        with pytest.raises(SymmetryError):
            as_symmetric(
                np.random.default_rng(9).standard_normal((4, 4)), symmetry=(0, 1)
            )


class TestPropagateSymmetrySlice:
    def test_full_slice_preserves(self):
        result = propagate_symmetry_slice(
            [_sg(0, 1)], (5, 5), (slice(None), slice(None))
        )
        _assert_groups_equal(result, [(0, 1)])

    def test_integer_index_on_s3_keeps_remaining_swap(self):
        result = propagate_symmetry_slice(
            [SymmetryGroup.symmetric(axes=(0, 1, 2))],
            (4, 4, 4),
            (0, slice(None), slice(None)),
        )
        _assert_groups_equal(result, [(0, 1)])

    def test_integer_index_removes_group(self):
        assert propagate_symmetry_slice([_sg(0, 1)], (5, 5), (0,)) is None

    def test_repeated_integer_indices_use_pointwise_stabilizer(self):
        result = propagate_symmetry_slice(
            [SymmetryGroup.symmetric(axes=(0, 1, 2))],
            (4, 4, 4),
            (0, 0, slice(None)),
        )
        assert result is None

    def test_newaxis_renumbers_group(self):
        result = propagate_symmetry_slice(
            [_sg(0, 1)], (5, 5), (None, slice(None), slice(None))
        )
        _assert_groups_equal(result, [(1, 2)])

    def test_multiple_groups_partial_survival(self):
        result = propagate_symmetry_slice([_sg(0, 1), _sg(2, 3)], (3, 3, 4, 4), (0,))
        _assert_groups_equal(result, [(1, 2)])

    def test_resized_kept_axis_drops_surviving_subgroup(self):
        result = propagate_symmetry_slice(
            [SymmetryGroup.symmetric(axes=(0, 1, 2))],
            (4, 4, 4),
            (0, slice(0, 3), slice(None)),
        )
        assert result is None


class TestPropagateSymmetryReduce:
    def test_reduce_non_symmetric_axis_keepdims_false(self):
        result = propagate_symmetry_reduce([_sg(1, 2)], 3, 0, keepdims=False)
        _assert_groups_equal(result, [(0, 1)])

    def test_reduce_symmetric_axis_breaks_group(self):
        assert propagate_symmetry_reduce([_sg(0, 1)], 2, 0, keepdims=False) is None

    def test_reduce_tuple_axis_keepdims_true(self):
        result = propagate_symmetry_reduce([_sg(0, 1, 2)], 3, 0, keepdims=True)
        _assert_groups_equal(result, [(1, 2)])


class TestIntersectSymmetry:
    def test_same_groups_intersect(self):
        result = intersect_symmetry([_sg(0, 1)], [_sg(0, 1)], (3, 3), (3, 3), (3, 3))
        _assert_groups_equal(result, [(0, 1)])

    def test_broadcast_alignment(self):
        result = intersect_symmetry(
            [_sg(0, 1)], [_sg(1, 2)], (3, 3), (1, 3, 3), (1, 3, 3)
        )
        _assert_groups_equal(result, [(1, 2)])

    def test_broadcast_stretched_dim_drops_group(self):
        result = intersect_symmetry([_sg(0, 1)], [_sg(0, 1)], (1, 1), (3, 3), (3, 3))
        assert result is None
