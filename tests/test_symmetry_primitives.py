"""Unit tests for low-level symmetry primitives in _symmetry.py.

Tests restrict_group, merge_two, pick_stronger, and merge_overlapping_groups
in isolation. These are the building blocks of propagate_symmetry used by
the path optimizer.

All test inputs use the new tuple-based IndexSymmetry format:
    frozenset({('i',), ('j',)}) is per-index S2{i,j}
    frozenset({('i', 'j'), ('k', 'l')}) is block S2 on blocks (i,j) and (k,l)
"""

import pytest
from mechestim._opt_einsum._symmetry import (
    restrict_group,
    merge_two,
    pick_stronger,
    merge_overlapping_groups,
)
