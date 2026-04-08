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


def _build_bipartite(
    operands: list[Any],
    subscript_parts: list[str],
    per_op_syms: list[IndexSymmetry | None],
    output_chars: str,
) -> EinsumBipartite:
    """Construct the bipartite graph for an einsum expression.

    Parameters
    ----------
    operands : list
        Original operand objects; Python identity is used to detect
        repeated operands.
    subscript_parts : list[str]
        Per-operand subscript strings (e.g., ["ij", "jk"]).
    per_op_syms : list[IndexSymmetry | None]
        Declared symmetry for each operand, in the tuple-based
        IndexSymmetry format.
    output_chars : str
        Output subscript string.
    """
    u_vertices: list[tuple[int, int]] = []
    u_labels: list[frozenset[str]] = []
    u_operand: list[int] = []
    incidence: list[dict[str, int]] = []
    operand_labels: list[frozenset[str]] = []

    for op_idx, sub in enumerate(subscript_parts):
        operand_labels.append(frozenset(sub))
        sym = per_op_syms[op_idx]

        # Build equivalence classes on the axes of this operand.
        # Each axis (position in sub) starts in its own singleton class.
        # Declared symmetry groups merge axes into classes by their
        # positional index within the operand.
        #
        # Declared sym is a list of frozenset({1-tuple, ...}) groups.
        # A group {("i",), ("j",)} on subscript "ij" means positions 0
        # and 1 are in the same class.
        class_of_position: dict[int, int] = {k: k for k in range(len(sub))}

        if sym is not None:
            for group in sym:
                # Only per-index (1-tuple) groups participate in U-vertex
                # merging. Higher-block groups are not currently produced
                # by SymmetricTensor (see _einsum.py's converter) and we
                # leave them as no-ops here.
                if any(len(t) != 1 for t in group):
                    continue
                chars_in_group = {t[0] for t in group}
                positions_in_group = [
                    k for k, c in enumerate(sub) if c in chars_in_group
                ]
                if len(positions_in_group) >= 2:
                    canonical = positions_in_group[0]
                    for k in positions_in_group[1:]:
                        class_of_position[k] = class_of_position[canonical]

        # Normalize class ids to 0..num_classes-1 in order of first occurrence
        class_order: dict[int, int] = {}
        for k in range(len(sub)):
            c = class_of_position[k]
            if c not in class_order:
                class_order[c] = len(class_order)
            class_of_position[k] = class_order[c]

        # Build one U vertex per class, with incidence = label multiplicity
        # across axes in the class.
        num_classes = len(class_order)
        class_incidence: list[dict[str, int]] = [dict() for _ in range(num_classes)]
        class_labels: list[set[str]] = [set() for _ in range(num_classes)]
        for k, c in enumerate(sub):
            cls = class_of_position[k]
            class_incidence[cls][c] = class_incidence[cls].get(c, 0) + 1
            class_labels[cls].add(c)

        for cls in range(num_classes):
            u_vertices.append((op_idx, cls))
            u_labels.append(frozenset(class_labels[cls]))
            u_operand.append(op_idx)
            incidence.append(class_incidence[cls])

    # Partition labels into free (V) vs summed (W) at the top level
    output_set = frozenset(output_chars)
    all_labels = frozenset().union(*operand_labels) if operand_labels else frozenset()
    free_labels = all_labels & output_set
    summed_labels = all_labels - output_set

    # Identical-operand groups via Python id
    id_to_positions: dict[int, list[int]] = {}
    for op_idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(op_idx)
    identical_operand_groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )

    return EinsumBipartite(
        u_vertices=tuple(u_vertices),
        u_labels=tuple(u_labels),
        u_operand=tuple(u_operand),
        incidence=tuple(incidence),
        free_labels=free_labels,
        summed_labels=summed_labels,
        identical_operand_groups=identical_operand_groups,
        operand_labels=tuple(operand_labels),
    )


@dataclass(frozen=True)
class _Subgraph:
    """Induced subgraph of an EinsumBipartite on a subset of operands.

    u_local: list of U-vertex indices (indices into graph.u_vertices) that
             belong to operands in the subset.
    v_labels: labels that are free at this step (output or crossing the cut).
    w_labels: labels that are summed entirely within the subset.
    id_groups: identical-operand groups restricted to the subset.
    """

    u_local: tuple[int, ...]
    v_labels: frozenset[str]
    w_labels: frozenset[str]
    id_groups: tuple[tuple[int, ...], ...]


def _induce_subgraph(
    graph: EinsumBipartite, subset: frozenset[int]
) -> _Subgraph:
    u_local = tuple(
        idx for idx, op_idx in enumerate(graph.u_operand) if op_idx in subset
    )

    labels_in_subset: set[str] = set()
    for idx in u_local:
        labels_in_subset.update(graph.incidence[idx].keys())

    # Labels appearing in operands outside the subset (crossing the cut).
    outside_labels: set[str] = set()
    for op_idx, op_lbls in enumerate(graph.operand_labels):
        if op_idx not in subset:
            outside_labels.update(op_lbls)

    # V at this step = labels_in_subset ∩ (free_labels ∪ outside_labels)
    v_labels = frozenset(labels_in_subset & (graph.free_labels | outside_labels))
    w_labels = frozenset(labels_in_subset - v_labels)

    id_groups = tuple(
        tuple(sorted(set(g) & subset))
        for g in graph.identical_operand_groups
        if len(set(g) & subset) >= 2
    )

    return _Subgraph(
        u_local=u_local,
        v_labels=v_labels,
        w_labels=w_labels,
        id_groups=id_groups,
    )


# Placeholder stubs — implemented in Tasks 1.4, 1.5.



def _compute_subset_symmetry(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("implemented in Task 1.5")


class SubgraphSymmetryOracle:  # placeholder
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("implemented in Task 1.5")
