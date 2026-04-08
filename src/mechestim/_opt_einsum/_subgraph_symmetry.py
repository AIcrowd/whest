"""Subset-keyed subgraph symmetry detection for einsum intermediates.

One oracle per contract_path call. Given the original operand list,
subscript parts, per-operand declared symmetries, and output subscript,
builds a bipartite graph once and exposes `.sym(subset)` which returns
the IndexSymmetry of the intermediate tensor for any subset of the
original operands, computed lazily on first access and cached.

See docs/explanation/subgraph-symmetry.md for the algorithm walkthrough.

TODO(sigma-to-pi): the hybrid block-candidate path in Step 2b is a
carry-over of the old _enumerate_block_candidates logic. The natural
unification is to extend Step 2a to derive the induced permutation pi
on V per sigma (instead of iterating pairs), which subsumes both paths.
Deferred to a follow-up iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Any

from ._symmetry import IndexSymmetry


_MISSING = object()
