"""Focused coverage tests for private permutation helpers and exact groups."""

from __future__ import annotations

import pytest

from whest._perm_group import SymmetryGroup, _dimino, _Permutation


class TestValidation:
    def test_group_requires_generators(self):
        with pytest.raises(ValueError, match="At least one generator"):
            SymmetryGroup()

    def test_mismatched_generator_sizes_raise(self):
        with pytest.raises(ValueError, match="same size"):
            SymmetryGroup(_Permutation([1, 0]), _Permutation([1, 0, 2]))

    def test_from_generators_rejects_non_bijection(self):
        with pytest.raises(ValueError, match="bijection"):
            SymmetryGroup.from_generators([[1, 1]], axes=(0, 1))

    def test_from_generators_rejects_wrong_degree(self):
        with pytest.raises(ValueError, match="degree"):
            SymmetryGroup.from_generators([[1, 0, 2]], axes=(0, 1))


class TestGroupOperations:
    def test_restrict(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        restricted = g.pointwise_stabilizer({0}).restrict((1, 2))
        assert restricted.axes == (1, 2)
        assert restricted.order() == 2

    def test_pointwise_and_setwise_stabilizers(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        assert g.pointwise_stabilizer({0}).order() == 2
        assert g.setwise_stabilizer({0, 1}).order() == 2

    def test_direct_product_requires_disjoint_support(self):
        a = SymmetryGroup.symmetric(axes=(0, 1))
        b = SymmetryGroup.symmetric(axes=(1, 2))
        with pytest.raises(ValueError, match="disjoint"):
            SymmetryGroup.direct_product(a, b)

    def test_repr_uses_symmetry_group_name(self):
        g = SymmetryGroup.cyclic(axes=(0, 1, 2))
        assert "SymmetryGroup(" in repr(g)


class TestDimino:
    def test_identity_generator_only(self):
        elems = _dimino((_Permutation.identity(3),))
        assert len(elems) == 1
        assert elems[0].is_identity

    def test_single_transposition_generates_order_two(self):
        gen = _Permutation.from_cycle(3, [0, 1])
        elems = _dimino((gen,))
        assert len(elems) == 2
        assert _Permutation.identity(3) in elems
        assert gen in elems

    def test_adjacent_transpositions_generate_s3(self):
        g1 = _Permutation.from_cycle(3, [0, 1])
        g2 = _Permutation.from_cycle(3, [1, 2])
        assert len(_dimino((g1, g2))) == 6
