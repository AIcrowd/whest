"""Tests for SymmetryInfo dataclass."""
import numpy as np
import pytest

from mechestim._symmetric import SymmetryInfo


class TestSymmetryInfo:
    """Tests for the SymmetryInfo frozen dataclass."""

    def test_single_group_unique_elements(self):
        """Single group (0,1) on (5,5): C(5+2-1, 2) = C(6,2) = 15."""
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
        assert info.unique_elements == 15

    def test_single_group_symmetry_factor(self):
        """Single group (0,1) on (5,5): 2! = 2."""
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
        assert info.symmetry_factor == 2

    def test_partial_symmetry(self):
        """Two groups [(0,1),(2,3)] on (4,4,3,3): C(4+1,2)*C(3+1,2) = 10*6 = 60."""
        info = SymmetryInfo(symmetric_dims=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
        assert info.unique_elements == 60
        assert info.symmetry_factor == 4  # 2! * 2! = 4

    def test_three_way_symmetry(self):
        """Three-way (0,1,2) on (3,3,3): 3! = 6, C(3+2, 3) = C(5,3) = 10."""
        info = SymmetryInfo(symmetric_dims=[(0, 1, 2)], shape=(3, 3, 3))
        assert info.symmetry_factor == 6
        assert info.unique_elements == 10

    def test_mixed_symmetric_and_free(self):
        """(0,1) on (5,5,8): C(6,2) * 8 = 15 * 8 = 120."""
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5, 8))
        assert info.unique_elements == 120

    def test_frozen(self):
        """SymmetryInfo is frozen; reassignment raises."""
        info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
        with pytest.raises(AttributeError):
            info.shape = (3, 3)

    def test_post_init_normalizes_dims(self):
        """Dims are normalized to sorted tuples."""
        info = SymmetryInfo(symmetric_dims=[(1, 0)], shape=(5, 5))
        assert info.symmetric_dims == [(0, 1)]
