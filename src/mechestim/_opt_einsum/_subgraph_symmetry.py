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
from itertools import permutations, product
from typing import Any

from ._symmetry import IndexSymmetry, SubsetSymmetry

_MISSING = object()


# ---------------------------------------------------------------------------
# π-based detection helpers (added alongside old Step 2a/2b path)
# ---------------------------------------------------------------------------


def _detect_fingerprint_equivalences(
    col_of: dict[str, tuple[int, ...]],
    labels: frozenset[str],
) -> IndexSymmetry:
    """Group labels by column fingerprint; return per-index groups for groups >= 2."""
    fp_groups: dict[tuple[int, ...], list[str]] = {}
    for lbl in sorted(labels):
        fp = col_of[lbl]
        fp_groups.setdefault(fp, []).append(lbl)
    return [
        frozenset((lbl,) for lbl in group)
        for group in fp_groups.values()
        if len(group) >= 2
    ]


def _derive_pi_canonical(
    sigma_col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], set[str]],
    v_labels: frozenset[str],
    w_labels: frozenset[str],
) -> dict[str, str] | None:
    """Build π by canonical hash lookup. Returns None on failure.

    For each label ℓ, looks up σ(M)'s column fingerprint in fp_to_labels
    and picks the lex-first unused candidate. Validates that π is a
    bijection and preserves the V/W partition.
    """
    pi: dict[str, str] = {}
    used: set[str] = set()
    all_labels = v_labels | w_labels

    for label in sorted(all_labels):
        fp = sigma_col_of[label]
        candidates = fp_to_labels.get(fp)
        if not candidates:
            return None
        pick = None
        for c in sorted(candidates):
            if c not in used:
                pick = c
                break
        if pick is None:
            return None
        pi[label] = pick
        used.add(pick)

    # Validate V→V and W→W.
    for lbl, target in pi.items():
        if lbl in v_labels and target not in v_labels:
            return None
        if lbl in w_labels and target not in w_labels:
            return None

    return pi


def _classify_pi_cycles(
    pi: dict[str, str],
    target_labels: frozenset[str],
    graph: "EinsumBipartite",
    sub: "_Subgraph",
) -> list[frozenset[tuple[str, ...]]]:
    """Decompose π restricted to target_labels into cycles, classify into groups."""
    # Decompose into disjoint cycles.
    visited: set[str] = set()
    cycles: list[tuple[str, ...]] = []
    for start in sorted(target_labels):
        if start in visited:
            continue
        if pi.get(start, start) == start:
            visited.add(start)
            continue
        cycle: list[str] = []
        cur = start
        while cur not in visited:
            cycle.append(cur)
            visited.add(cur)
            cur = pi[cur]
        cycles.append(tuple(cycle))

    if not cycles:
        return []

    # Single cycle: per-index group.
    if len(cycles) == 1:
        return [frozenset((lbl,) for lbl in cycles[0])]

    # Multiple cycles: check if all same length for block classification.
    cycle_lengths = {len(c) for c in cycles}
    if len(cycle_lengths) != 1:
        # Mixed lengths: each cycle is an independent per-index group.
        return [frozenset((lbl,) for lbl in c) for c in cycles]

    # All same length — attempt block construction.
    blocks = _build_blocks_from_cycles(cycles, graph, sub)
    if blocks is not None:
        return [frozenset(blocks)]

    # Block construction failed — fall back to per-index per cycle.
    return [frozenset((lbl,) for lbl in c) for c in cycles]


def _build_blocks_from_cycles(
    cycles: list[tuple[str, ...]],
    graph: "EinsumBipartite",
    sub: "_Subgraph",
) -> list[tuple[str, ...]] | None:
    """Given same-length disjoint cycles, group labels by operand into blocks.

    Returns list of k tuples (one per block) or None if the cycle
    structure doesn't correspond to a clean block swap.
    """
    k = len(cycles[0])  # cycle length = number of blocks

    def _single_operand_of(label: str) -> int | None:
        ops = [i for i, lbls in enumerate(graph.operand_labels) if label in lbls]
        return ops[0] if len(ops) == 1 else None

    # Compute operand sequence per cycle.
    cycle_op_seqs: list[list[int]] = []
    for cycle in cycles:
        op_seq: list[int] = []
        for lbl in cycle:
            op = _single_operand_of(lbl)
            if op is None:
                return None
            op_seq.append(op)
        if len(set(op_seq)) != k:
            return None  # cycle revisits same operand
        cycle_op_seqs.append(op_seq)

    reference = cycle_op_seqs[0]

    # Align each cycle to reference operand order via cyclic rotation.
    blocks: list[list[str]] = [[] for _ in range(k)]
    for cycle, op_seq in zip(cycles, cycle_op_seqs):
        rotation = _find_rotation(op_seq, reference)
        if rotation is None:
            return None
        aligned = cycle[rotation:] + cycle[:rotation]
        for block_idx, lbl in enumerate(aligned):
            blocks[block_idx].append(lbl)

    # Order within each block by subscript position.
    for block_idx in range(k):
        op = reference[block_idx]
        subscript = graph.operand_subscripts[op]
        blocks[block_idx].sort(key=lambda lbl: subscript.index(lbl))

    # Validate all blocks same size.
    if not all(len(b) == len(cycles) for b in blocks):
        return None

    return [tuple(b) for b in blocks]


def _find_rotation(
    seq: list[int], reference: list[int]
) -> int | None:
    """Find r such that seq[r:] + seq[:r] == reference."""
    k = len(seq)
    for r in range(k):
        if seq[r:] + seq[:r] == reference:
            return r
    return None


def _detect_symmetries_via_pi(
    graph: "EinsumBipartite",
    sub: "_Subgraph",
    row_order: tuple[int, ...],
    col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], set[str]],
) -> tuple[list[frozenset[tuple[str, ...]]], list[frozenset[tuple[str, ...]]]]:
    """Unified σ loop: derive π per σ, classify cycles on V and W.

    Returns (v_groups, w_groups) — candidate groups for each side.
    """
    v_groups: list[frozenset[tuple[str, ...]]] = []
    w_groups: list[frozenset[tuple[str, ...]]] = []
    all_labels = sub.v_labels | sub.w_labels

    for tilde_sigma in _enumerate_id_group_permutations(sub.id_groups):
        # Skip identity σ — it always gives π = identity.
        if not tilde_sigma or all(
            tilde_sigma.get(k, k) == k for k in tilde_sigma
        ):
            continue

        sigma_row_perm = _lift_operand_perm_to_u(tilde_sigma, row_order, graph)
        if sigma_row_perm is None:
            continue

        # Compute σ(M)'s column fingerprints.
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in all_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[sigma_row_perm[k]].get(label, 0)
                for k in range(len(row_order))
            )

        # Derive π.
        pi = _derive_pi_canonical(
            sigma_col_of, fp_to_labels, sub.v_labels, sub.w_labels
        )
        if pi is None:
            continue

        # Classify cycles on V.
        if sub.v_labels:
            v_groups.extend(_classify_pi_cycles(pi, sub.v_labels, graph, sub))

        # Classify cycles on W.
        if sub.w_labels:
            w_groups.extend(_classify_pi_cycles(pi, sub.w_labels, graph, sub))

    return v_groups, w_groups


# ---------------------------------------------------------------------------
# End of π-based detection helpers
# ---------------------------------------------------------------------------


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
    u_vertices: tuple[tuple[int, int], ...]  # (operand_idx, class_id)
    u_labels: tuple[frozenset[str], ...]  # which labels this class contains
    u_operand: tuple[int, ...]  # operand index
    incidence: tuple[dict[str, int], ...]  # {label -> multiplicity}

    # Right vertices, top-level partition:
    free_labels: frozenset[str]  # V at the top level
    summed_labels: frozenset[str]  # W at the top level

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


def _induce_subgraph(graph: EinsumBipartite, subset: frozenset[int]) -> _Subgraph:
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
    on first access to a subset and cached in memory. Returns a
    ``SubsetSymmetry`` with ``.output`` (V-side) and ``.inner`` (W-side).
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
        self._cache: dict[frozenset[int], SubsetSymmetry] = {}

    def sym(self, subset: frozenset[int]) -> SubsetSymmetry:
        cached = self._cache.get(subset, _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        result = _compute_subset_symmetry(self._graph, subset)
        self._cache[subset] = result
        return result


def _compute_subset_symmetry(
    graph: EinsumBipartite,
    subset: frozenset[int],
) -> SubsetSymmetry:
    sub = _induce_subgraph(graph, subset)
    if not sub.v_labels and not sub.w_labels:
        return SubsetSymmetry(None, None)

    # Step 1: column fingerprints.
    row_order = sub.u_local
    all_labels = sub.v_labels | sub.w_labels
    col_of: dict[str, tuple[int, ...]] = {}
    for label in all_labels:
        col_of[label] = tuple(graph.incidence[u].get(label, 0) for u in row_order)

    # Step 2: reverse index.
    fp_to_labels: dict[tuple[int, ...], set[str]] = {}
    for lbl, fp in col_of.items():
        fp_to_labels.setdefault(fp, set()).add(lbl)

    # Step 3: fast path (fingerprint equivalences, no σ needed).
    v_fast = _detect_fingerprint_equivalences(col_of, sub.v_labels) if sub.v_labels else []
    w_fast = _detect_fingerprint_equivalences(col_of, sub.w_labels) if sub.w_labels else []

    # Step 4: σ loop — derive π per σ, classify cycles on V and W.
    v_sigma, w_sigma = _detect_symmetries_via_pi(
        graph, sub, row_order, col_of, fp_to_labels
    )

    # Step 5: merge.
    v_merged = _merge_overlapping_groups(v_fast + v_sigma)
    w_merged = _merge_overlapping_groups(w_fast + w_sigma)

    return SubsetSymmetry(
        output=v_merged or None,
        inner=w_merged or None,
    )


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

        sigma_w_multiset = tuple(sorted(sigma_col_of[lbl] for lbl in w_sorted))
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
                *[graph.operand_labels[k] for k in subset if k != i and k != j]
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
    """Validate a block sigma by applying the operand swap (i ↔ j) as a
    row permutation on M_S and the label sigma as a column relabel, then
    checking that V columns match and W columns are a permutation."""
    tilde_sigma = {i: j, j: i}
    sigma_row_perm = _lift_operand_perm_to_u(tilde_sigma, row_order, graph)
    if sigma_row_perm is None:
        return False

    all_labels = sub.v_labels | sub.w_labels
    sigma_col_of: dict[str, tuple[int, ...]] = {}
    for label in all_labels:
        sigma_col_of[label] = tuple(
            graph.incidence[sigma_row_perm[k]].get(label, 0)
            for k in range(len(row_order))
        )

    w_sorted = tuple(sorted(sub.w_labels))
    if tuple(sorted(sigma_col_of[lbl] for lbl in w_sorted)) != tuple(
        sorted(col_of[lbl] for lbl in w_sorted)
    ):
        return False

    for v_lbl in sub.v_labels:
        mapped = sigma.get(v_lbl, v_lbl)
        if col_of[v_lbl] != sigma_col_of[mapped]:
            return False
    return True


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
