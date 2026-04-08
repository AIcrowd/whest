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


class SubgraphSymmetryOracle:
    """Subset-keyed symmetry oracle for einsum intermediates.

    One oracle per contract_path call. Symmetries are computed lazily
    on first access to a subset and cached in memory.
    """

    def __init__(
        self,
        operands: list[Any],
        subscript_parts: list[str],
        per_op_syms: list[IndexSymmetry | None],
        output_chars: str,
    ) -> None:
        self._graph = _build_bipartite(
            operands=operands,
            subscript_parts=subscript_parts,
            per_op_syms=per_op_syms,
            output_chars=output_chars,
        )
        self._cache: dict[frozenset[int], IndexSymmetry | None] = {}

    def sym(self, subset: frozenset[int]) -> IndexSymmetry | None:
        cached = self._cache.get(subset, _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        result = _compute_subset_symmetry(self._graph, subset)
        self._cache[subset] = result
        return result


def _compute_subset_symmetry(
    graph: EinsumBipartite,
    subset: frozenset[int],
) -> IndexSymmetry | None:
    # Step 0: induce the subgraph
    sub = _induce_subgraph(graph, subset)
    if not sub.v_labels:
        return None

    # Step 1: canonical column encoding of M_S
    # Row order: sub.u_local in the order given.
    # Column: for each label L, the tuple of incidence multiplicities
    # across rows in sub.u_local.
    row_order = sub.u_local
    all_labels = sub.v_labels | sub.w_labels
    col_of: dict[str, tuple[int, ...]] = {}
    for label in all_labels:
        col_of[label] = tuple(
            graph.incidence[u].get(label, 0) for u in row_order
        )

    # Step 2a: per-index pair detection
    pair_set: set[tuple[str, str]] = set()
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))
    w_col_multiset = tuple(sorted(col_of[lbl] for lbl in w_sorted))

    for tilde_sigma in _enumerate_id_group_permutations(sub.id_groups):
        sigma_row_perm = _lift_operand_perm_to_u(
            tilde_sigma, row_order, graph
        )
        if sigma_row_perm is None:
            continue
        # sigma_row_perm[k] = the NEW row that replaces row_order[k]
        # when sigma is applied. To build sigma(M_S)[:, label], we look
        # up graph.incidence[sigma_row_perm[k]].get(label, 0) for k in
        # range(len(row_order)).
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in all_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[sigma_row_perm[k]].get(label, 0)
                for k in range(len(row_order))
            )

        # W-columns must be a permutation
        sigma_w_multiset = tuple(sorted(sigma_col_of[lbl] for lbl in w_sorted))
        if sigma_w_multiset != w_col_multiset:
            continue

        # Pair check
        for i_idx in range(len(v_sorted)):
            for j_idx in range(i_idx + 1, len(v_sorted)):
                i_lbl = v_sorted[i_idx]
                j_lbl = v_sorted[j_idx]
                if col_of[i_lbl] != sigma_col_of[j_lbl]:
                    continue
                if col_of[j_lbl] != sigma_col_of[i_lbl]:
                    continue
                # All other V columns unchanged
                ok = True
                for k_idx in range(len(v_sorted)):
                    if k_idx == i_idx or k_idx == j_idx:
                        continue
                    k_lbl = v_sorted[k_idx]
                    if col_of[k_lbl] != sigma_col_of[k_lbl]:
                        ok = False
                        break
                if ok:
                    pair_set.add((i_lbl, j_lbl))

    per_index_groups = _pairs_to_groups(sub.v_labels, pair_set)

    # Step 2b and Step 3 land in Tasks 1.6 and 1.7.
    return per_index_groups or None


def _enumerate_id_group_permutations(
    id_groups: tuple[tuple[int, ...], ...],
) -> list[dict[int, int]]:
    """Enumerate all permutations that permute within each identical-operand group.

    Returns a list of mappings {original_op_idx -> permuted_op_idx}.
    The identity is included. For groups g1, g2, ... of sizes k1, k2, ...,
    there are k1! * k2! * ... total permutations.
    """
    per_group_perms: list[list[dict[int, int]]] = []
    for group in id_groups:
        group_list = list(group)
        perms_for_group: list[dict[int, int]] = []
        for p in permutations(group_list):
            perms_for_group.append(dict(zip(group_list, p)))
        per_group_perms.append(perms_for_group)

    if not per_group_perms:
        return [dict()]  # identity only

    result: list[dict[int, int]] = []
    for combo in product(*per_group_perms):
        merged: dict[int, int] = {}
        for d in combo:
            merged.update(d)
        result.append(merged)
    return result


def _lift_operand_perm_to_u(
    tilde_sigma: dict[int, int],
    row_order: tuple[int, ...],
    graph: EinsumBipartite,
) -> tuple[int, ...] | None:
    """Lift a permutation of operands to a permutation of U-vertices.

    For each U vertex in row_order belonging to operand i (where
    tilde_sigma[i] = j), find the U vertex of operand j in the same
    positional class. If the lift is not well-defined, return None.

    Returns a tuple of length len(row_order) where element k is the
    U-vertex index that row_order[k] maps TO under sigma.
    """
    # Build an operand-local class ordering: for each operand, list its
    # U-vertex indices in the order they appear in graph.u_vertices.
    operand_to_u_vertices: dict[int, list[int]] = {}
    for u_idx, op_idx in enumerate(graph.u_operand):
        operand_to_u_vertices.setdefault(op_idx, []).append(u_idx)

    # Build row_order's per-operand position map: for each row index k,
    # determine which operand it belongs to and its class-position within
    # that operand (i.e., its index in operand_to_u_vertices[op]).
    result = list(row_order)
    for k, u_idx in enumerate(row_order):
        op_idx = graph.u_operand[u_idx]
        if op_idx not in tilde_sigma or tilde_sigma[op_idx] == op_idx:
            continue  # identity on this operand
        j_op = tilde_sigma[op_idx]
        # Which class-position within operand op_idx is this U vertex?
        op_classes = operand_to_u_vertices[op_idx]
        pos = op_classes.index(u_idx)
        # Map to the class-position in operand j_op
        j_classes = operand_to_u_vertices.get(j_op, [])
        if pos >= len(j_classes):
            return None  # class count mismatch; lift undefined
        result[k] = j_classes[pos]
    return tuple(result)


def _pairs_to_groups(
    v_labels: frozenset[str], pair_set: set[tuple[str, str]]
) -> IndexSymmetry:
    """Union-find over V, joining endpoints in pair_set. Returns one
    IndexSymmetry group per connected component of size >= 2."""
    parent: dict[str, str] = {lbl: lbl for lbl in v_labels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in pair_set:
        if a in parent and b in parent:
            union(a, b)

    components: dict[str, list[str]] = {}
    for lbl in v_labels:
        root = find(lbl)
        components.setdefault(root, []).append(lbl)

    return [
        frozenset((lbl,) for lbl in sorted(comp))
        for comp in components.values()
        if len(comp) >= 2
    ]
