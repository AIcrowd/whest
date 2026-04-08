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


@dataclass(frozen=True)
class EinsumBipartite:
    """Bipartite graph representation of an einsum expression.

    Left vertices U: one per (operand_idx, equivalence-class-id).
    An equivalence class within an operand is a maximal set of axes
    identified by that operand's declared symmetry partition. For a
    dense operand with subscript "ai" we get two U vertices — one for
    class {a}, one for class {i}. For a fully symmetric operand T with
    subscript "ij" we get one U vertex for class {i, j}.

    Right vertices are labels, partitioned at the top level into
    free_labels (V, the final output) and summed_labels (W, contracted
    at the top level). Subset induction may reclassify labels from W
    to V when they cross the cut.
    """

    # Parallel tuples over U vertices:
    u_vertices: tuple[tuple[int, int], ...]       # (operand_idx, class_id)
    u_labels: tuple[frozenset[str], ...]          # which labels this class contains
    u_operand: tuple[int, ...]                    # operand index
    incidence: tuple[dict[str, int], ...]         # {label -> multiplicity}

    # Right vertices, top-level partition:
    free_labels: frozenset[str]                   # V at the top level
    summed_labels: frozenset[str]                 # W at the top level

    # Python-identity groups: partition of [0..num_operands),
    # non-singleton blocks enumerate identical operands.
    identical_operand_groups: tuple[tuple[int, ...], ...]

    # Per-operand label set, needed for subset induction to compute
    # crossing labels efficiently without re-scanning incidence.
    operand_labels: tuple[frozenset[str], ...]


# Placeholder stubs — implemented in Tasks 1.3, 1.4, 1.5.
def _build_bipartite(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("implemented in Task 1.3")


def _compute_subset_symmetry(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("implemented in Task 1.5")


class SubgraphSymmetryOracle:  # placeholder
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("implemented in Task 1.5")
