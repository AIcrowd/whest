"""Unit tests for low-level symmetry primitives in _symmetry.py.

Tests restrict_group, merge_two, pick_stronger, and merge_overlapping_groups
in isolation. These are the building blocks of propagate_symmetry used by
the path optimizer.

All test inputs use the new tuple-based IndexSymmetry format:
    frozenset({('i',), ('j',)}) is per-index S2{i,j}
    frozenset({('i', 'j'), ('k', 'l')}) is block S2 on blocks (i,j) and (k,l)
"""

from mechestim._opt_einsum._symmetry import (
    merge_overlapping_groups,
    merge_two,
    pick_stronger,
    restrict_group,
)


class TestRestrictGroupPerIndex:
    """restrict_group with per-index groups (block size 1)."""

    def test_all_surviving_returns_unchanged(self):
        group = frozenset({("i",), ("j",), ("k",)})
        result = restrict_group(group, frozenset("ijk"))
        assert result == group

    def test_one_contracted_drops_that_block(self):
        group = frozenset({("i",), ("j",), ("k",)})
        result = restrict_group(group, frozenset("ij"))  # k contracted
        assert result == frozenset({("i",), ("j",)})

    def test_only_one_surviving_returns_none(self):
        group = frozenset({("i",), ("j",), ("k",)})
        result = restrict_group(group, frozenset("i"))  # j, k contracted
        assert result is None

    def test_zero_surviving_returns_none(self):
        group = frozenset({("i",), ("j",)})
        result = restrict_group(group, frozenset())
        assert result is None

    def test_no_overlap_with_surviving_returns_none(self):
        group = frozenset({("i",), ("j",)})
        result = restrict_group(group, frozenset("xy"))
        assert result is None


class TestRestrictGroupBlock:
    """restrict_group with block groups (block size >= 2).

    Based on Section 2 of the spec — block degradation rules.
    Block group [(j,k),(l,m)] means T is invariant under swapping
    blocks (j,k) and (l,m) as units.
    """

    # Case A: nothing contracted → block sym intact
    def test_case_a_nothing_contracted(self):
        group = frozenset({("j", "k"), ("l", "m")})
        result = restrict_group(group, frozenset("jklm"))
        assert result == group

    # Case B: parallel contraction at position 1 → per-index S2
    def test_case_b_parallel_position_1(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # k and m contracted, j and l survive
        result = restrict_group(group, frozenset("jl"))
        # Should degrade to per-index S2{j,l}
        assert result == frozenset({("j",), ("l",)})

    def test_case_b_prime_parallel_position_0(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # j and l contracted, k and m survive
        result = restrict_group(group, frozenset("km"))
        # Should degrade to per-index S2{k,m}
        assert result == frozenset({("k",), ("m",)})

    # Case C: non-parallel contraction → breaks
    def test_case_c_non_parallel(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # j (pos 0 of block 0) and m (pos 1 of block 1) contracted — non-parallel
        result = restrict_group(group, frozenset("kl"))
        assert result is None

    def test_case_c_alt_non_parallel(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # k (pos 1 of block 0) and l (pos 0 of block 1) contracted — non-parallel
        result = restrict_group(group, frozenset("jm"))
        assert result is None

    # Case D: whole block contracted → breaks
    def test_case_d_whole_block_0(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # j, k both contracted → block 0 gone entirely
        result = restrict_group(group, frozenset("lm"))
        assert result is None

    def test_case_d_whole_block_1(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # l, m both contracted → block 1 gone entirely
        result = restrict_group(group, frozenset("jk"))
        assert result is None

    # Case E: only one index surviving → trivially None
    def test_case_e_single_survivor(self):
        group = frozenset({("j", "k"), ("l", "m")})
        result = restrict_group(group, frozenset("l"))
        assert result is None

    # Case F: asymmetric (one index from one block, nothing from other) → breaks
    def test_case_f_asymmetric(self):
        group = frozenset({("j", "k"), ("l", "m")})
        # Only k contracted — block 0 becomes (j,), block 1 stays (l,m) — mismatched
        result = restrict_group(group, frozenset("jlm"))
        assert result is None

    # 3-block case: parallel collapse preserves block structure
    def test_three_block_parallel_pos_1_collapse(self):
        # S3 on blocks: [(a,b),(c,d),(e,f)]
        group = frozenset({("a", "b"), ("c", "d"), ("e", "f")})
        # Contract b, d, f (parallel at position 1)
        result = restrict_group(group, frozenset("ace"))
        # Should degrade to per-index S3{a,c,e}
        assert result == frozenset({("a",), ("c",), ("e",)})

    # 3-tuple block, contract middle position → keep 2-block
    def test_3tuple_block_middle_contracted_stays_block(self):
        # Block group with 3-element tuples: [(a,b,c),(d,e,f)]
        group = frozenset({("a", "b", "c"), ("d", "e", "f")})
        # Contract b and e (parallel at position 1). Positions 0, 2 survive.
        result = restrict_group(group, frozenset("acdf"))
        # Blocks become (a,c) and (d,f) — still a 2-block group of size 2
        assert result == frozenset({("a", "c"), ("d", "f")})


class TestMergeTwoAndPickStronger:
    """merge_two and pick_stronger helpers for merge_overlapping_groups."""

    def test_merge_two_same_block_size_unions(self):
        g1 = frozenset({("i",), ("j",)})
        g2 = frozenset({("j",), ("k",)})
        result = merge_two(g1, g2)
        assert result == frozenset({("i",), ("j",), ("k",)})

    def test_merge_two_same_size_full_union(self):
        g1 = frozenset({("a",), ("b",)})
        g2 = frozenset({("b",), ("c",), ("d",)})
        result = merge_two(g1, g2)
        assert result == frozenset({("a",), ("b",), ("c",), ("d",)})

    def test_merge_two_same_block_size_2_unions(self):
        g1 = frozenset({("j", "k"), ("l", "m")})
        g2 = frozenset({("l", "m"), ("n", "o")})
        result = merge_two(g1, g2)
        assert result == frozenset({("j", "k"), ("l", "m"), ("n", "o")})

    def test_merge_two_different_sizes_returns_none(self):
        g1 = frozenset({("j",), ("k",)})  # size 1
        g2 = frozenset({("j", "l"), ("m", "n")})  # size 2
        result = merge_two(g1, g2)
        assert result is None

    def test_pick_stronger_prefers_larger_block_size(self):
        g1 = frozenset({("j",), ("k",)})  # size 1
        g2 = frozenset({("j", "l"), ("m", "n")})  # size 2
        result = pick_stronger(g1, g2)
        assert result == g2

    def test_pick_stronger_reverse_order(self):
        g1 = frozenset({("j", "l"), ("m", "n")})
        g2 = frozenset({("j",), ("k",)})
        result = pick_stronger(g1, g2)
        assert result == g1

    def test_pick_stronger_equal_sizes_returns_first(self):
        g1 = frozenset({("a",), ("b",)})
        g2 = frozenset({("c",), ("d",)})
        result = pick_stronger(g1, g2)
        assert result == g1


class TestMergeOverlappingGroups:
    """merge_overlapping_groups — connected-component merge with block awareness."""

    def test_disjoint_groups_unchanged(self):
        candidates = [
            frozenset({("a",), ("b",)}),
            frozenset({("c",), ("d",)}),
        ]
        result = merge_overlapping_groups(candidates)
        assert sorted(result, key=sorted) == sorted(candidates, key=sorted)

    def test_two_overlapping_per_index_groups_union(self):
        candidates = [
            frozenset({("a",), ("b",)}),
            frozenset({("b",), ("c",)}),
        ]
        result = merge_overlapping_groups(candidates)
        assert len(result) == 1
        assert result[0] == frozenset({("a",), ("b",), ("c",)})

    def test_three_pairwise_s2_merge_to_s3(self):
        """Classic triple product: S2{j,k}, S2{j,l}, S2{k,l} → S3{j,k,l}."""
        candidates = [
            frozenset({("j",), ("k",)}),
            frozenset({("j",), ("l",)}),
            frozenset({("k",), ("l",)}),
        ]
        result = merge_overlapping_groups(candidates)
        assert len(result) == 1
        assert result[0] == frozenset({("j",), ("k",), ("l",)})

    def test_empty_input(self):
        result = merge_overlapping_groups([])
        assert result == []

    def test_single_group_unchanged(self):
        candidates = [frozenset({("a",), ("b",)})]
        result = merge_overlapping_groups(candidates)
        assert result == candidates

    def test_mixed_size_overlap_picks_stronger(self):
        """When per-index and block groups overlap on shared chars, keep the stronger (block)."""
        per_idx = frozenset({("j",), ("k",)})  # size 1
        block = frozenset(
            {("j", "l"), ("k", "m")}
        )  # size 2, shares j and k with per_idx
        candidates = [per_idx, block]
        result = merge_overlapping_groups(candidates)
        assert len(result) == 1
        assert result[0] == block  # block is stronger

    def test_two_disjoint_block_groups(self):
        candidates = [
            frozenset({("a", "b"), ("c", "d")}),
            frozenset({("e", "f"), ("g", "h")}),
        ]
        result = merge_overlapping_groups(candidates)
        # Disjoint — both preserved
        assert len(result) == 2

    def test_two_overlapping_block_groups_same_size(self):
        """Blocks sharing a character union their block sets."""
        g1 = frozenset({("a", "b"), ("c", "d")})
        g2 = frozenset({("c", "d"), ("e", "f")})  # shares block (c,d)
        candidates = [g1, g2]
        result = merge_overlapping_groups(candidates)
        assert len(result) == 1
        # Union of block sets
        assert result[0] == frozenset({("a", "b"), ("c", "d"), ("e", "f")})


class TestUniqueElementsTuple:
    """_unique_elements_tuple with the new tuple-based IndexSymmetry format."""

    def test_no_symmetry_returns_product(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        indices = frozenset("ij")
        size_dict = {"i": 10, "j": 10}
        assert _unique_elements_tuple(indices, size_dict, None) == 100

    def test_per_index_s2(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        # S2{i,j}: C(n+1, 2) unique elements
        indices = frozenset("ij")
        size_dict = {"i": 10, "j": 10}
        sym = [frozenset({("i",), ("j",)})]
        # C(10+1, 2) = 55
        assert _unique_elements_tuple(indices, size_dict, sym) == 55

    def test_per_index_s3(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        indices = frozenset("ijk")
        size_dict = {"i": 10, "j": 10, "k": 10}
        sym = [frozenset({("i",), ("j",), ("k",)})]
        # C(10+2, 3) = 220
        assert _unique_elements_tuple(indices, size_dict, sym) == 220

    def test_per_index_with_free_dim(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        # a is free, (j,k) is S2
        indices = frozenset("ajk")
        size_dict = {"a": 10, "j": 10, "k": 10}
        sym = [frozenset({("j",), ("k",)})]
        # 10 * C(11,2) = 10 * 55 = 550
        assert _unique_elements_tuple(indices, size_dict, sym) == 550

    def test_block_s2_2x2(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        # Block S2 on (j,k) and (l,m): C(n^2 + 1, 2)
        indices = frozenset("jklm")
        size_dict = {"j": 10, "k": 10, "l": 10, "m": 10}
        sym = [frozenset({("j", "k"), ("l", "m")})]
        # n=10, s=2, k=2 → C(100+1, 2) = 5050
        assert _unique_elements_tuple(indices, size_dict, sym) == 5050

    def test_block_s2_with_free_dim(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        indices = frozenset("ajklm")
        size_dict = {"a": 5, "j": 10, "k": 10, "l": 10, "m": 10}
        sym = [frozenset({("j", "k"), ("l", "m")})]
        # 5 (free a) * 5050 (block) = 25250
        assert _unique_elements_tuple(indices, size_dict, sym) == 25250

    def test_two_independent_s2_groups(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        indices = frozenset("ijkl")
        size_dict = {"i": 10, "j": 10, "k": 10, "l": 10}
        sym = [
            frozenset({("i",), ("j",)}),
            frozenset({("k",), ("l",)}),
        ]
        # C(11,2) * C(11,2) = 55 * 55 = 3025
        assert _unique_elements_tuple(indices, size_dict, sym) == 3025

    def test_empty_indices(self):
        from mechestim._opt_einsum._symmetry import (
            unique_elements as _unique_elements_tuple,
        )

        assert _unique_elements_tuple(frozenset(), {"a": 10}, None) == 1
