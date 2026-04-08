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

    # Per-operand subscript string, needed by the block path helper
    # to reconstruct free-to-one-operand label sets in positional order.
    operand_subscripts: tuple[str, ...]  # parallel to operand_labels


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
        operand_subscripts=tuple(subscript_parts),
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
    sub = _induce_subgraph(graph, subset)
    if not sub.v_labels:
        return None

    # Step 1: canonical column encoding
    row_order = sub.u_local
    all_labels = sub.v_labels | sub.w_labels
    col_of: dict[str, tuple[int, ...]] = {}
    for label in all_labels:
        col_of[label] = tuple(
            graph.incidence[u].get(label, 0) for u in row_order
        )

    # Step 2a: per-index pair detection
    per_index_groups = _detect_per_index_pairs(graph, sub, row_order, col_of)

    # Step 2b: block candidate detection (hybrid carry-over)
    block_groups = _detect_block_candidates(graph, sub, subset, col_of, row_order)

    # Step 3: merge per-index + block candidates
    all_candidates = list(per_index_groups) + list(block_groups)
    if not all_candidates:
        return None
    merged = _merge_overlapping_groups(all_candidates)
    return merged or None


def _detect_per_index_pairs(
    graph: EinsumBipartite,
    sub: _Subgraph,
    row_order: tuple[int, ...],
    col_of: dict[str, tuple[int, ...]],
) -> IndexSymmetry:
    pair_set: set[tuple[str, str]] = set()
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))
    w_col_multiset = tuple(sorted(col_of[lbl] for lbl in w_sorted))

    for tilde_sigma in _enumerate_id_group_permutations(sub.id_groups):
        sigma_row_perm = _lift_operand_perm_to_u(tilde_sigma, row_order, graph)
        if sigma_row_perm is None:
            continue
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in sub.v_labels | sub.w_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[sigma_row_perm[k]].get(label, 0)
                for k in range(len(row_order))
            )

        sigma_w_multiset = tuple(
            sorted(sigma_col_of[lbl] for lbl in w_sorted)
        )
        if sigma_w_multiset != w_col_multiset:
            continue

        for i_idx in range(len(v_sorted)):
            for j_idx in range(i_idx + 1, len(v_sorted)):
                i_lbl = v_sorted[i_idx]
                j_lbl = v_sorted[j_idx]
                if col_of[i_lbl] != sigma_col_of[j_lbl]:
                    continue
                if col_of[j_lbl] != sigma_col_of[i_lbl]:
                    continue
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

    return _pairs_to_groups(sub.v_labels, pair_set)


def _detect_block_candidates(
    graph: EinsumBipartite,
    sub: _Subgraph,
    subset: frozenset[int],
    col_of: dict[str, tuple[int, ...]],
    row_order: tuple[int, ...],
) -> IndexSymmetry:
    """Hybrid block candidate path. Ported from _enumerate_block_candidates.

    For each pair (i, j) of operand positions within `subset` where the
    operands are in the same identical-operand group, build the positional
    block-swap sigma that maps "free-only-to-i" labels to "free-only-to-j"
    labels in subscript order, then validate via the same column-equality
    machinery used in Step 2a.
    """
    id_group_of: dict[int, tuple[int, ...]] = {}
    for group in graph.identical_operand_groups:
        for op in group:
            id_group_of[op] = tuple(sorted(set(group) & subset))

    groups: list[frozenset[tuple[str, ...]]] = []
    seen_group: set[frozenset[tuple[str, ...]]] = set()

    subscripts = graph.operand_subscripts

    for i in sorted(subset):
        for j in sorted(subset):
            if j <= i:
                continue
            if id_group_of.get(i) is None or id_group_of.get(j) is None:
                continue
            if j not in id_group_of[i]:
                continue

            sub_i = subscripts[i]
            sub_j = subscripts[j]
            other_ops_labels = frozenset().union(
                *[
                    graph.operand_labels[k]
                    for k in subset
                    if k != i and k != j
                ]
            )

            sub_i_set = frozenset(sub_i)
            sub_j_set = frozenset(sub_j)
            free_i = sub_i_set - sub_j_set - other_ops_labels
            free_j = sub_j_set - sub_i_set - other_ops_labels
            free_i = free_i & sub.v_labels
            free_j = free_j & sub.v_labels
            if len(free_i) != len(free_j) or len(free_i) == 0:
                continue

            free_i_ordered = tuple(c for c in sub_i if c in free_i)
            free_j_ordered = tuple(c for c in sub_j if c in free_j)

            sigma: dict[str, str] = {}
            for a, b in zip(free_i_ordered, free_j_ordered):
                sigma[a] = b
                sigma[b] = a

            if _block_sigma_is_valid(
                graph, sub, subset, i, j, sigma, col_of, row_order
            ):
                if len(free_i) == 1:
                    group = frozenset({(free_i_ordered[0],), (free_j_ordered[0],)})
                else:
                    group = frozenset({free_i_ordered, free_j_ordered})
                if group not in seen_group:
                    seen_group.add(group)
                    groups.append(group)

    return groups


def _block_sigma_is_valid(
    graph: EinsumBipartite,
    sub: _Subgraph,
    subset: frozenset[int],
    i: int,
    j: int,
    sigma: dict[str, str],
    col_of: dict[str, tuple[int, ...]],
    row_order: tuple[int, ...],
) -> bool:
    """Validate a block sigma in two ways (OR):

    1. Column-equality path: apply the operand swap (i ↔ j) as a row
       permutation on M_S and check that V columns match under the label
       sigma and W columns are a multiset-permutation.

    2. Subscript-multiset path: apply sigma to the label names of each
       operand in `subset` and check that the resulting (subscript_set,
       operand_id) multiset matches the original.  This catches cases
       like "ij,jk->ik" with same operand X where i↔k is valid because
       swapping the operands and relabeling reproduces the same multiset,
       even though the column patterns differ due to the summed index j.
    """
    # Path 1: column-equality (geometric)
    tilde_sigma = {i: j, j: i}
    sigma_row_perm = _lift_operand_perm_to_u(tilde_sigma, row_order, graph)
    if sigma_row_perm is not None:
        all_labels = sub.v_labels | sub.w_labels
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in all_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[sigma_row_perm[k]].get(label, 0)
                for k in range(len(row_order))
            )

        w_sorted = tuple(sorted(sub.w_labels))
        w_ok = tuple(sorted(sigma_col_of[lbl] for lbl in w_sorted)) == tuple(
            sorted(col_of[lbl] for lbl in w_sorted)
        )
        if w_ok and all(
            col_of[v_lbl] == sigma_col_of[sigma.get(v_lbl, v_lbl)]
            for v_lbl in sub.v_labels
        ):
            return True

    # Path 2: subscript-multiset (algebraic)
    # Apply sigma to each operand's subscript and check if the resulting
    # (frozenset_of_labels, operand_id) multiset equals the original.
    from collections import Counter

    original_multiset = Counter(
        (frozenset(graph.operand_subscripts[op]), id(None))  # id placeholder
        for op in subset
    )
    # We don't have the actual operand objects here, but we can use the
    # operand index as a proxy since all operands in an id_group share the
    # same Python id. The key is: does applying sigma to subscripts produce
    # the same multiset of subscript-frozensets for identical operands?
    # Build the original multiset: {(subscript_frozenset, operand_class): count}
    # where operand_class groups operands by identity.
    # Use the identical_operand_groups to determine which ops share identity.
    id_class_of: dict[int, int] = {}
    for group_idx, group in enumerate(graph.identical_operand_groups):
        for op in group:
            if op in subset:
                id_class_of[op] = group_idx
    # ops not in any group get unique classes
    next_class = len(graph.identical_operand_groups)
    for op in sorted(subset):
        if op not in id_class_of:
            id_class_of[op] = next_class
            next_class += 1

    from collections import Counter as _Counter
    original_ctr = _Counter(
        (frozenset(graph.operand_subscripts[op]), id_class_of[op])
        for op in subset
    )
    relabeled_ctr = _Counter(
        (frozenset(sigma.get(c, c) for c in graph.operand_subscripts[op]), id_class_of[op])
        for op in subset
    )
    return original_ctr == relabeled_ctr


def _merge_overlapping_groups(
    candidates: list[frozenset[tuple[str, ...]]],
) -> list[frozenset[tuple[str, ...]]]:
    """Merge groups that share any individual label character.

    For groups of the same block size, take the union of their blocks.
    For groups of different block sizes, keep the one with the larger
    block size (block symmetries dominate per-index).
    """
    merged: list[frozenset[tuple[str, ...]]] = []
    for g in candidates:
        g_chars = frozenset(c for block in g for c in block)
        overlapping = [
            k
            for k, m in enumerate(merged)
            if any(c in g_chars for block in m for c in block)
        ]
        if not overlapping:
            merged.append(g)
            continue

        combined = g
        for k in sorted(overlapping, reverse=True):
            other = merged.pop(k)
            s1 = len(next(iter(combined)))
            s2 = len(next(iter(other)))
            if s1 == s2:
                combined = frozenset(combined | other)
            else:
                combined = combined if s1 > s2 else other
        merged.append(combined)
    return merged


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
