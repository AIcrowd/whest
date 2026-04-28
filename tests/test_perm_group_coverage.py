"""Extended coverage tests for _perm_group.py.

Targets uncovered lines to bring coverage from ~79% to ~95%:
- Permutation.from_cycle() edge cases
- Permutation.__invert__() verification
- Permutation.cyclic_form for various permutation types
- Permutation.cycle_structure and order
- PermutationGroup.__init__() validation (mismatched degrees)
- PermutationGroup.orbits() with various symmetry groups
- PermutationGroup.burnside_unique_count() with different dim dicts
- _dimino() algorithm with various generator sets
- Edge cases: identity, single-element groups, large groups
"""

from __future__ import annotations

import math

import pytest

from flopscope._perm_group import Permutation, PermutationGroup, _dimino

# ---------------------------------------------------------------------------
# Permutation.from_cycle()
# ---------------------------------------------------------------------------


class TestFromCycle:
    def test_single_element_cycle(self):
        """A 1-cycle is the identity."""
        p = Permutation.from_cycle(4, [2])
        assert p.is_identity is False or p.array_form == [0, 1, 2, 3]
        # A 1-cycle maps element to itself, so the result is identity.
        assert p.array_form == [0, 1, 2, 3]

    def test_3_cycle_on_larger_set(self):
        """A 3-cycle (1 3 4) on size 6 leaves 0, 2, 5 fixed."""
        p = Permutation.from_cycle(6, [1, 3, 4])
        assert p.array_form[0] == 0
        assert p.array_form[1] == 3
        assert p.array_form[2] == 2
        assert p.array_form[3] == 4
        assert p.array_form[4] == 1
        assert p.array_form[5] == 5

    def test_full_cycle(self):
        """Full n-cycle on n elements."""
        p = Permutation.from_cycle(5, [0, 1, 2, 3, 4])
        assert p.array_form == [1, 2, 3, 4, 0]

    def test_two_element_cycle(self):
        """A transposition (0 2) on size 3."""
        p = Permutation.from_cycle(3, [0, 2])
        assert p.array_form == [2, 1, 0]


# ---------------------------------------------------------------------------
# Permutation.__invert__()
# ---------------------------------------------------------------------------


class TestInverse:
    def test_inverse_of_identity(self):
        e = Permutation.identity(4)
        assert (~e).is_identity

    def test_inverse_of_transposition(self):
        """A transposition is its own inverse."""
        p = Permutation([1, 0, 2, 3])
        assert ~p == p

    def test_inverse_of_3_cycle(self):
        """(0 1 2)^{-1} = (0 2 1)."""
        p = Permutation([1, 2, 0])
        inv = ~p
        assert inv.array_form == [2, 0, 1]

    def test_p_times_inv_is_identity(self):
        """p * ~p == identity for various permutations."""
        perms = [
            Permutation([1, 2, 0]),
            Permutation([3, 0, 1, 2]),
            Permutation([1, 0, 3, 2, 4]),
            Permutation([4, 3, 2, 1, 0]),
        ]
        for p in perms:
            e = p * ~p
            assert e.is_identity, f"p * ~p != identity for {p}"
            e2 = ~p * p
            assert e2.is_identity, f"~p * p != identity for {p}"

    def test_inverse_of_large_permutation(self):
        p = Permutation([5, 3, 4, 1, 0, 2])
        inv = ~p
        product = p * inv
        assert product.is_identity


# ---------------------------------------------------------------------------
# Permutation.cyclic_form
# ---------------------------------------------------------------------------


class TestCyclicForm:
    def test_identity_has_no_cycles(self):
        assert Permutation.identity(5).cyclic_form == []

    def test_single_transposition(self):
        p = Permutation([0, 2, 1, 3])
        cycles = p.cyclic_form
        assert len(cycles) == 1
        assert set(cycles[0]) == {1, 2}

    def test_two_disjoint_transpositions(self):
        p = Permutation([1, 0, 3, 2])
        cycles = p.cyclic_form
        assert len(cycles) == 2
        cycle_sets = {frozenset(c) for c in cycles}
        assert cycle_sets == {frozenset({0, 1}), frozenset({2, 3})}

    def test_4_cycle(self):
        p = Permutation([1, 2, 3, 0])
        cycles = p.cyclic_form
        assert len(cycles) == 1
        assert len(cycles[0]) == 4

    def test_mixed_cycle_and_fixed(self):
        """(0 2 4) with 1, 3 fixed on size 5."""
        p = Permutation([2, 1, 4, 3, 0])
        cycles = p.cyclic_form
        # Only non-fixed-point cycles
        all_in_cycles = set()
        for c in cycles:
            all_in_cycles.update(c)
        assert 1 not in all_in_cycles
        assert 3 not in all_in_cycles
        assert {0, 2, 4}.issubset(all_in_cycles)

    def test_smallest_element_first(self):
        """Each cycle should start with its smallest element."""
        p = Permutation([3, 0, 1, 2])
        for cycle in p.cyclic_form:
            assert cycle[0] == min(cycle)


# ---------------------------------------------------------------------------
# Permutation.cycle_structure
# ---------------------------------------------------------------------------


class TestCycleStructure:
    def test_identity_empty(self):
        assert Permutation.identity(4).cycle_structure == {}

    def test_single_3_cycle(self):
        p = Permutation([1, 2, 0, 3])
        assert p.cycle_structure == {3: 1}

    def test_two_2_cycles(self):
        p = Permutation([1, 0, 3, 2])
        assert p.cycle_structure == {2: 2}

    def test_mixed_lengths(self):
        """(0 1)(2 3 4)(5 6 7 8) on size 9 -> {2:1, 3:1, 4:1}."""
        p = Permutation([1, 0, 3, 4, 2, 6, 7, 8, 5])
        assert p.cycle_structure == {2: 1, 3: 1, 4: 1}


# ---------------------------------------------------------------------------
# Permutation.order
# ---------------------------------------------------------------------------


class TestPermutationOrder:
    def test_identity_order(self):
        assert Permutation.identity(5).order == 1

    def test_transposition_order(self):
        assert Permutation([1, 0, 2]).order == 2

    def test_3_cycle_order(self):
        assert Permutation([1, 2, 0]).order == 3

    def test_product_of_disjoint_cycles(self):
        """(0 1)(2 3 4) has order lcm(2,3) = 6."""
        p = Permutation([1, 0, 3, 4, 2])
        assert p.order == 6

    def test_lcm_of_coprime_lengths(self):
        """(0 1 2)(3 4 5 6 7) has order lcm(3,5) = 15."""
        p = Permutation([1, 2, 0, 4, 5, 6, 7, 3])
        assert p.order == 15

    def test_power_equals_identity(self):
        """p^order == identity."""
        p = Permutation([1, 0, 3, 4, 2])
        result = p
        for _ in range(p.order - 1):
            result = result * p
        assert result.is_identity


# ---------------------------------------------------------------------------
# PermutationGroup.__init__() validation
# ---------------------------------------------------------------------------


class TestGroupValidation:
    def test_no_generators_raises(self):
        with pytest.raises(ValueError, match="At least one generator"):
            PermutationGroup()

    def test_mismatched_degrees_raises(self):
        g1 = Permutation([1, 0, 2])  # size 3
        g2 = Permutation([1, 0, 2, 3])  # size 4
        with pytest.raises(ValueError, match="same size"):
            PermutationGroup(g1, g2)

    def test_mismatched_three_generators(self):
        g1 = Permutation([1, 0])  # size 2
        g2 = Permutation([1, 0])  # size 2
        g3 = Permutation([1, 0, 2])  # size 3
        with pytest.raises(ValueError, match="same size"):
            PermutationGroup(g1, g2, g3)


# ---------------------------------------------------------------------------
# PermutationGroup.orbits()
# ---------------------------------------------------------------------------


class TestOrbits:
    def test_identity_group_all_singletons(self):
        """Trivial group: each element is its own orbit."""
        e = Permutation.identity(4)
        g = PermutationGroup(e)
        orbs = g.orbits()
        assert len(orbs) == 4
        for orb in orbs:
            assert len(orb) == 1

    def test_full_symmetric_single_orbit(self):
        g = PermutationGroup.symmetric(4)
        orbs = g.orbits()
        assert len(orbs) == 1
        assert orbs[0] == frozenset({0, 1, 2, 3})

    def test_two_disjoint_transpositions(self):
        """Generators (0 1) and (2 3) on size 4 give orbits {0,1} and {2,3}."""
        g1 = Permutation.from_cycle(4, [0, 1])
        g2 = Permutation.from_cycle(4, [2, 3])
        g = PermutationGroup(g1, g2)
        orbs = sorted(g.orbits(), key=lambda s: min(s))
        assert orbs == [frozenset({0, 1}), frozenset({2, 3})]

    def test_cyclic_group_single_orbit(self):
        g = PermutationGroup.cyclic(5)
        orbs = g.orbits()
        assert len(orbs) == 1
        assert orbs[0] == frozenset(range(5))

    def test_partial_orbit_with_fixed_points(self):
        """(0 1 2) on size 5 gives orbits {0,1,2}, {3}, {4}."""
        gen = Permutation.from_cycle(5, [0, 1, 2])
        g = PermutationGroup(gen)
        orbs = sorted(g.orbits(), key=lambda s: min(s))
        assert orbs[0] == frozenset({0, 1, 2})
        assert orbs[1] == frozenset({3})
        assert orbs[2] == frozenset({4})


# ---------------------------------------------------------------------------
# PermutationGroup.burnside_unique_count()
# ---------------------------------------------------------------------------


class TestBurnsideExtended:
    def test_trivial_group_all_unique(self):
        """Identity-only group: all n^k elements distinct."""
        e = Permutation.identity(2)
        g = PermutationGroup(e)
        assert g.burnside_unique_count({0: 4, 1: 4}) == 16

    def test_c4_known_formula(self):
        """C_4: unique count = (n^4 + n^2 + 2n) / 4 for dim n."""
        g = PermutationGroup.cyclic(4)
        for n in [2, 3, 5]:
            expected = (n**4 + n**2 + 2 * n) // 4
            assert g.burnside_unique_count({0: n, 1: n, 2: n, 3: n}) == expected

    def test_dihedral_4(self):
        """D_4 has 8 elements; verify count for small n."""
        g = PermutationGroup.dihedral(4)
        n = 3
        # Compute manually from Burnside:
        # Must enumerate all 8 elements and count fixed points.
        # Just verify it's between C_4 and S_4 bounds.
        result = g.burnside_unique_count({0: n, 1: n, 2: n, 3: n})
        s4_result = math.comb(n + 3, 4)
        c4_result = (n**4 + n**2 + 2 * n) // 4
        assert s4_result <= result <= c4_result

    def test_heterogeneous_dims_valid(self):
        """Non-uniform dims across different orbits are fine."""
        # (0 1) on size 3: orbits {0,1}, {2}
        gen = Permutation.from_cycle(3, [0, 1])
        g = PermutationGroup(gen)
        # {0,1} must have same size, {2} can differ
        result = g.burnside_unique_count({0: 4, 1: 4, 2: 7})
        # Group has 2 elements: identity and (0 1).
        # Identity: 4 * 4 * 7 = 112
        # (0 1): cycle form is (0 1)(2), so full_cyclic = [(0,1),(2)],
        #   product = 4 * 7 = 28
        # Total = (112 + 28) / 2 = 70
        assert result == 70

    def test_mismatched_orbit_dims_raises(self):
        """Positions in the same orbit with different sizes must raise."""
        g = PermutationGroup.symmetric(3)
        with pytest.raises(ValueError, match="same dimension size"):
            g.burnside_unique_count({0: 3, 1: 3, 2: 5})

    def test_s1_trivial(self):
        """S_1 = trivial group on 1 element."""
        g = PermutationGroup.symmetric(1)
        assert g.burnside_unique_count({0: 10}) == 10


# ---------------------------------------------------------------------------
# _dimino() algorithm
# ---------------------------------------------------------------------------


class TestDimino:
    def test_identity_generator_only(self):
        """Single identity generator produces group of order 1."""
        e = Permutation.identity(3)
        elems = _dimino((e,))
        assert len(elems) == 1
        assert elems[0].is_identity

    def test_single_transposition(self):
        """Single transposition generates Z_2."""
        gen = Permutation.from_cycle(3, [0, 1])
        elems = _dimino((gen,))
        assert len(elems) == 2
        assert Permutation.identity(3) in elems
        assert gen in elems

    def test_single_3_cycle(self):
        """Single 3-cycle generates C_3."""
        gen = Permutation([1, 2, 0])
        elems = _dimino((gen,))
        assert len(elems) == 3

    def test_two_adjacent_transpositions_generate_s3(self):
        """Adjacent transpositions (0 1) and (1 2) generate S_3."""
        g1 = Permutation.from_cycle(3, [0, 1])
        g2 = Permutation.from_cycle(3, [1, 2])
        elems = _dimino((g1, g2))
        assert len(elems) == 6

    def test_redundant_generators(self):
        """Duplicate generators don't inflate the group."""
        gen = Permutation([1, 2, 0])
        elems = _dimino((gen, gen))
        assert len(elems) == 3

    def test_s4_from_generators(self):
        """S_4 from adjacent transpositions has 24 elements."""
        gens = []
        for i in range(3):
            arr = list(range(4))
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            gens.append(Permutation(arr))
        elems = _dimino(tuple(gens))
        assert len(elems) == 24

    def test_dimino_closure(self):
        """Every product of two elements is in the group."""
        gen = Permutation([1, 2, 0])
        gen2 = Permutation([1, 0, 2])
        elems = _dimino((gen, gen2))
        elem_set = set(elems)
        for a in elems:
            for b in elems:
                assert a * b in elem_set

    def test_dihedral_generators(self):
        """Rotation + reflection for D_5 gives 10 elements."""
        rot = Permutation([1, 2, 3, 4, 0])
        refl = Permutation([0, 4, 3, 2, 1])
        elems = _dimino((rot, refl))
        assert len(elems) == 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_element_permutation(self):
        """Size-1 permutation is trivially the identity."""
        p = Permutation([0])
        assert p.is_identity
        assert p.size == 1
        assert p.cyclic_form == []
        assert p.full_cyclic_form == [(0,)]
        assert p.cycle_structure == {}
        assert p.order == 1

    def test_repr(self):
        p = Permutation([2, 0, 1])
        assert repr(p) == "Permutation([2, 0, 1])"

    def test_eq_not_implemented_for_non_permutation(self):
        p = Permutation([0, 1, 2])
        assert p != "not a permutation"
        assert p != 42
        assert p.__eq__("other") is NotImplemented

    def test_group_repr(self):
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        r = repr(g)
        assert "PermutationGroup(" in r
        assert "axes=(0, 1, 2)" in r

    def test_group_repr_no_axes(self):
        g = PermutationGroup.cyclic(2)
        r = repr(g)
        assert "axes=" not in r

    def test_symmetric_1(self):
        """S_1 is the trivial group."""
        g = PermutationGroup.symmetric(1)
        assert g.order() == 1
        assert g.is_symmetric()

    def test_cyclic_1(self):
        """C_1 is trivial."""
        g = PermutationGroup.cyclic(1)
        assert g.order() == 1

    def test_dihedral_1(self):
        """D_1 = S_1, trivial."""
        g = PermutationGroup.dihedral(1)
        assert g.order() == 1

    def test_dihedral_2(self):
        """D_2 = S_2, order 2."""
        g = PermutationGroup.dihedral(2)
        assert g.order() == 2

    def test_symmetric_k_less_than_1_raises(self):
        with pytest.raises(ValueError):
            PermutationGroup.symmetric(0)

    def test_cyclic_k_less_than_1_raises(self):
        with pytest.raises(ValueError):
            PermutationGroup.cyclic(0)

    def test_dihedral_k_less_than_1_raises(self):
        with pytest.raises(ValueError):
            PermutationGroup.dihedral(0)

    def test_order_caching(self):
        """Calling order() twice returns cached value."""
        g = PermutationGroup.symmetric(3)
        o1 = g.order()
        o2 = g.order()
        assert o1 == o2 == 6

    def test_generators_returns_copy(self):
        g = PermutationGroup.symmetric(3)
        gens = g.generators
        assert isinstance(gens, list)
        # Modifying returned list doesn't affect internal state
        gens.clear()
        assert len(g.generators) > 0

    def test_axes_none_by_default(self):
        g = PermutationGroup.symmetric(3)
        assert g.axes is None

    def test_from_cycle_empty_cycle(self):
        """Empty cycle list produces identity."""
        p = Permutation.from_cycle(3, [])
        assert p.is_identity
