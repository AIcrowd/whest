"""Tests for the current exact-group implementation."""

from __future__ import annotations

import math

import pytest

from whest._perm_group import SymmetryGroup, _Cycle, _Permutation


class TestPermutation:
    def test_identity(self):
        e = _Permutation.identity(3)
        assert e.array_form == [0, 1, 2]
        assert e.size == 3
        assert e.is_identity

    def test_from_cycle(self):
        p = _Permutation.from_cycle(4, [1, 3])
        assert p.array_form == [0, 3, 2, 1]

    def test_compose_and_inverse(self):
        p = _Permutation([1, 2, 0])
        assert (p * p).array_form == [2, 0, 1]
        assert (p * ~p).is_identity
        assert (~p * p).is_identity

    def test_cycle_views(self):
        p = _Permutation([1, 0, 3, 2])
        assert {frozenset(c) for c in p.cyclic_form} == {
            frozenset({0, 1}),
            frozenset({2, 3}),
        }
        assert p.cycle_structure == {2: 2}
        assert p.order == 2


class TestCycleBuilder:
    def test_cycle_builder_expands_to_permutation(self):
        c = _Cycle(0, 2)(1, 3)
        p = _Permutation(c)
        assert p.array_form == [2, 3, 0, 1]
        assert c.list() == [2, 3, 0, 1]


class TestSymmetryGroup:
    def test_symmetric_cyclic_dihedral_orders(self):
        assert SymmetryGroup.symmetric(axes=(0, 1)).order() == 2
        assert SymmetryGroup.symmetric(axes=(0, 1, 2)).order() == 6
        assert SymmetryGroup.cyclic(axes=(0, 1, 2)).order() == 3
        assert SymmetryGroup.dihedral(axes=(0, 1, 2, 3)).order() == 8

    def test_from_generators_and_axes(self):
        g = SymmetryGroup.from_generators([[1, 0, 2]], axes=(2, 5, 7))
        assert g.axes == (2, 5, 7)
        assert g.degree == 3
        assert g.order() == 2

    def test_orbits_and_burnside(self):
        g = SymmetryGroup.symmetric(axes=(0, 1, 2))
        assert g.orbits() == [frozenset({0, 1, 2})]
        for n in [3, 5, 10]:
            assert g.burnside_unique_count({0: n, 1: n, 2: n}) == math.comb(n + 2, 3)

    def test_direct_product(self):
        row = SymmetryGroup.symmetric(axes=(0, 1))
        col = SymmetryGroup.symmetric(axes=(2, 3))
        product = SymmetryGroup.direct_product(row, col)
        assert product.axes == (0, 1, 2, 3)
        assert product.order() == 4


try:
    import sympy as _sympy_check  # noqa: F401

    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


@pytest.mark.skipif(not _SYMPY_AVAILABLE, reason='sympy not installed')
class TestSympyBridge:
    def test_permutation_round_trip(self):
        p = _Permutation([2, 0, 1])
        p2 = _Permutation.from_sympy(p.as_sympy())
        assert p2 == p

    def test_group_round_trip(self):
        g = SymmetryGroup.cyclic(axes=(0, 1, 2, 3))
        g2 = SymmetryGroup.from_sympy(g.as_sympy(), axes=(0, 1, 2, 3))
        assert g2.axes == g.axes
        assert g2.order() == g.order()
