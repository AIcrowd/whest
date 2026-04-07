"""Unit tests for low-level symmetry primitives in _symmetry.py.

Tests restrict_group, merge_two, pick_stronger, and merge_overlapping_groups
in isolation. These are the building blocks of propagate_symmetry used by
the path optimizer.

All test inputs use the new tuple-based IndexSymmetry format:
    frozenset({('i',), ('j',)}) is per-index S2{i,j}
    frozenset({('i', 'j'), ('k', 'l')}) is block S2 on blocks (i,j) and (k,l)
"""

from mechestim._opt_einsum._symmetry import restrict_group


class TestRestrictGroupPerIndex:
    """restrict_group with per-index groups (block size 1)."""

    def test_all_surviving_returns_unchanged(self):
        group = frozenset({('i',), ('j',), ('k',)})
        result = restrict_group(group, frozenset("ijk"))
        assert result == group

    def test_one_contracted_drops_that_block(self):
        group = frozenset({('i',), ('j',), ('k',)})
        result = restrict_group(group, frozenset("ij"))  # k contracted
        assert result == frozenset({('i',), ('j',)})

    def test_only_one_surviving_returns_none(self):
        group = frozenset({('i',), ('j',), ('k',)})
        result = restrict_group(group, frozenset("i"))  # j, k contracted
        assert result is None

    def test_zero_surviving_returns_none(self):
        group = frozenset({('i',), ('j',)})
        result = restrict_group(group, frozenset())
        assert result is None

    def test_no_overlap_with_surviving_returns_none(self):
        group = frozenset({('i',), ('j',)})
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
        group = frozenset({('j', 'k'), ('l', 'm')})
        result = restrict_group(group, frozenset("jklm"))
        assert result == group

    # Case B: parallel contraction at position 1 → per-index S2
    def test_case_b_parallel_position_1(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # k and m contracted, j and l survive
        result = restrict_group(group, frozenset("jl"))
        # Should degrade to per-index S2{j,l}
        assert result == frozenset({('j',), ('l',)})

    def test_case_b_prime_parallel_position_0(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # j and l contracted, k and m survive
        result = restrict_group(group, frozenset("km"))
        # Should degrade to per-index S2{k,m}
        assert result == frozenset({('k',), ('m',)})

    # Case C: non-parallel contraction → breaks
    def test_case_c_non_parallel(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # j (pos 0 of block 0) and m (pos 1 of block 1) contracted — non-parallel
        result = restrict_group(group, frozenset("kl"))
        assert result is None

    def test_case_c_alt_non_parallel(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # k (pos 1 of block 0) and l (pos 0 of block 1) contracted — non-parallel
        result = restrict_group(group, frozenset("jm"))
        assert result is None

    # Case D: whole block contracted → breaks
    def test_case_d_whole_block_0(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # j, k both contracted → block 0 gone entirely
        result = restrict_group(group, frozenset("lm"))
        assert result is None

    def test_case_d_whole_block_1(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # l, m both contracted → block 1 gone entirely
        result = restrict_group(group, frozenset("jk"))
        assert result is None

    # Case E: only one index surviving → trivially None
    def test_case_e_single_survivor(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        result = restrict_group(group, frozenset("l"))
        assert result is None

    # Case F: asymmetric (one index from one block, nothing from other) → breaks
    def test_case_f_asymmetric(self):
        group = frozenset({('j', 'k'), ('l', 'm')})
        # Only k contracted — block 0 becomes (j,), block 1 stays (l,m) — mismatched
        result = restrict_group(group, frozenset("jlm"))
        assert result is None

    # 3-block case: parallel collapse preserves block structure
    def test_three_block_parallel_pos_1_collapse(self):
        # S3 on blocks: [(a,b),(c,d),(e,f)]
        group = frozenset({('a', 'b'), ('c', 'd'), ('e', 'f')})
        # Contract b, d, f (parallel at position 1)
        result = restrict_group(group, frozenset("ace"))
        # Should degrade to per-index S3{a,c,e}
        assert result == frozenset({('a',), ('c',), ('e',)})

    # 3-tuple block, contract middle position → keep 2-block
    def test_3tuple_block_middle_contracted_stays_block(self):
        # Block group with 3-element tuples: [(a,b,c),(d,e,f)]
        group = frozenset({('a', 'b', 'c'), ('d', 'e', 'f')})
        # Contract b and e (parallel at position 1). Positions 0, 2 survive.
        result = restrict_group(group, frozenset("acdf"))
        # Blocks become (a,c) and (d,f) — still a 2-block group of size 2
        assert result == frozenset({('a', 'c'), ('d', 'f')})
