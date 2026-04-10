"""Tests for Permutation and PermutationGroup."""

from __future__ import annotations

import pytest

from mechestim._perm_group import Permutation


class TestPermutation:
    def test_identity(self):
        e = Permutation.identity(3)
        assert e.array_form == [0, 1, 2]
        assert e.size == 3
        assert e.is_identity

    def test_from_array(self):
        p = Permutation([1, 2, 0])
        assert p.array_form == [1, 2, 0]
        assert p.size == 3
        assert not p.is_identity

    def test_from_cycle(self):
        # (0 → 1 → 2 → 0)
        p = Permutation.from_cycle(3, [0, 1, 2])
        assert p.array_form == [1, 2, 0]

    def test_from_cycle_transposition(self):
        p = Permutation.from_cycle(4, [1, 3])
        assert p.array_form == [0, 3, 2, 1]

    def test_compose(self):
        # (0 1 2) * (0 1 2) = (0 2 1)
        p = Permutation([1, 2, 0])
        q = p * p
        assert q.array_form == [2, 0, 1]

    def test_inverse(self):
        p = Permutation([1, 2, 0])
        inv = ~p
        e = p * inv
        assert e.is_identity

    def test_eq_and_hash(self):
        a = Permutation([1, 2, 0])
        b = Permutation([1, 2, 0])
        c = Permutation([2, 0, 1])
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert len({a, b, c}) == 2

    def test_cyclic_form(self):
        p = Permutation([1, 2, 0])
        assert p.cyclic_form == [(0, 1, 2)]

    def test_cyclic_form_two_cycles(self):
        # (0 1)(2 3)
        p = Permutation([1, 0, 3, 2])
        cycles = p.cyclic_form
        assert len(cycles) == 2
        assert set(map(frozenset, cycles)) == {frozenset({0, 1}), frozenset({2, 3})}

    def test_cyclic_form_identity(self):
        assert Permutation.identity(3).cyclic_form == []

    def test_full_cyclic_form_includes_fixed_points(self):
        # (0 1) on size 4: fixed points 2, 3 become 1-cycles
        p = Permutation([1, 0, 2, 3])
        full = p.full_cyclic_form
        # Should cover every position exactly once
        all_positions = sorted(pos for cycle in full for pos in cycle)
        assert all_positions == [0, 1, 2, 3]

    def test_cycle_structure(self):
        # (0 1 2) = one 3-cycle
        p = Permutation([1, 2, 0])
        assert p.cycle_structure == {3: 1}

    def test_cycle_structure_mixed(self):
        # (0 1)(2 3 4) on size 5 → {2: 1, 3: 1}
        p = Permutation([1, 0, 3, 4, 2])
        assert p.cycle_structure == {2: 1, 3: 1}

    def test_order(self):
        # (0 1 2) has order 3
        assert Permutation([1, 2, 0]).order == 3
        # (0 1)(2 3 4) has order lcm(2,3) = 6
        assert Permutation([1, 0, 3, 4, 2]).order == 6
        # identity has order 1
        assert Permutation.identity(5).order == 1
