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
