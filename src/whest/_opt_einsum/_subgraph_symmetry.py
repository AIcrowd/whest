"""Subset-keyed subgraph symmetry detection for einsum intermediates.

One oracle per contract_path call. Given the original operand list,
subscript parts, per-operand declared symmetries, and output subscript,
builds a bipartite graph once and exposes ``.sym(subset)`` which returns
a ``SubsetSymmetry`` with ``.output`` (V-side) and ``.inner`` (W-side)
symmetries, computed lazily on first access and cached.

The detection algorithm derives the induced column permutation π for
each operand permutation σ via column-fingerprint hash lookup, then
collects the resulting Permutation objects to build a PermutationGroup
directly. A fingerprint fast-path detects S_k symmetry without running
the σ-loop when all labels on a side share the same column fingerprint.

See docs/explanation/subgraph-symmetry.md for the algorithm walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any

from whest._perm_group import Permutation as Perm
from whest._perm_group import PermutationGroup

from ._symmetry import SubsetSymmetry

_MISSING = object()


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


def _collect_pi_permutations(
    graph: "EinsumBipartite",
    sub: "_Subgraph",
    row_order: tuple[int, ...],
    col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], set[str]],
) -> tuple[list[Perm], list[Perm]]:
    """Collect all valid π's as Permutation objects for V and W labels.

    Returns
    -------
    v_perms : list[Perm]
        Non-identity permutations on V labels (output / free side).
    w_perms : list[Perm]
        Non-identity permutations on W labels (inner / summed side).
    """
    v_perms: list[Perm] = []
    w_perms: list[Perm] = []
    all_labels = sub.v_labels | sub.w_labels
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))
    v_idx = {lbl: i for i, lbl in enumerate(v_sorted)}
    w_idx = {lbl: i for i, lbl in enumerate(w_sorted)}

    for tilde_sigma in _enumerate_id_group_permutations(sub.id_groups):
        # Skip identity σ — it always gives π = identity.
        if not tilde_sigma or all(tilde_sigma.get(k, k) == k for k in tilde_sigma):
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

        # Restrict π to V labels — emit Perm if non-identity.
        if sub.v_labels and any(pi.get(lbl, lbl) != lbl for lbl in sub.v_labels):
            arr = [v_idx[pi.get(lbl, lbl)] for lbl in v_sorted]
            v_perms.append(Perm(arr))

        # Restrict π to W labels — emit Perm if non-identity.
        if sub.w_labels and any(pi.get(lbl, lbl) != lbl for lbl in sub.w_labels):
            arr = [w_idx[pi.get(lbl, lbl)] for lbl in w_sorted]
            w_perms.append(Perm(arr))

    return v_perms, w_perms


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

    # Declared symmetry groups per operand, preserved from construction
    # so the fingerprint fast path can use exact declared groups instead
    # of always promoting to S_k.
    per_op_groups: tuple[tuple[PermutationGroup, ...] | None, ...]


def _build_bipartite(
    operands: list[Any],
    subscript_parts: list[str],
    per_op_groups: list[list[PermutationGroup] | None],
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
    per_op_groups : list
        Declared symmetry for each operand, as a list of
        PermutationGroup objects (with ``_labels`` set), or None.
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
        groups = per_op_groups[op_idx]

        # Build equivalence classes on the axes of this operand.
        # Each axis (position in sub) starts in its own singleton class.
        # Declared symmetry groups merge axes into classes via orbit
        # analysis on the PermutationGroup.
        class_of_position: dict[int, int] = {k: k for k in range(len(sub))}

        if groups is not None:
            for group in groups:
                if group._labels is None:
                    continue
                for orbit in group.orbits():
                    if len(orbit) < 2:
                        continue
                    chars_in_orbit = {group._labels[i] for i in orbit}
                    positions_in_orbit = [
                        k for k, c in enumerate(sub) if c in chars_in_orbit
                    ]
                    if len(positions_in_orbit) >= 2:
                        canonical = positions_in_orbit[0]
                        for k in positions_in_orbit[1:]:
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
        per_op_groups=tuple(
            tuple(gs) if gs is not None else None for gs in per_op_groups
        ),
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
        per_op_groups: list[list[PermutationGroup] | None],
        output_chars: str,
    ) -> None:
        self._graph = _build_bipartite(
            operands=operands,
            subscript_parts=subscript_parts,
            per_op_groups=per_op_groups,
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


def _find_declared_group_for_labels(
    graph: EinsumBipartite,
    subset: frozenset[int],
    target_labels: tuple[str, ...],
) -> PermutationGroup | None:
    """Find a declared group covering *target_labels*, relabeled to match order.

    Returns a new PermutationGroup with generators conjugated so that
    position *i* corresponds to ``target_labels[i]``, or None if no
    declared group covers these labels.
    """
    target_set = frozenset(target_labels)
    for op_idx in subset:
        groups = graph.per_op_groups[op_idx]
        if groups is None:
            continue
        for group in groups:
            if group._labels is None or frozenset(group._labels) != target_set:
                continue
            # Labels match — relabel generators if ordering differs.
            if tuple(group._labels) == target_labels:
                return group
            old_labels = group._labels
            # p[old_pos] = new_pos  (maps old label order → target order)
            new_pos_of = {lbl: i for i, lbl in enumerate(target_labels)}
            p = [new_pos_of[old_labels[i]] for i in range(len(old_labels))]
            p_inv = [0] * len(p)
            for i, pi in enumerate(p):
                p_inv[pi] = i
            new_gens = []
            for gen in group.generators:
                arr = gen.array_form
                new_arr = [p[arr[p_inv[i]]] for i in range(len(arr))]
                new_gens.append(Perm(new_arr))
            return PermutationGroup(*new_gens)
    return None


def _compute_subset_symmetry(
    graph: EinsumBipartite,
    subset: frozenset[int],
) -> SubsetSymmetry:
    sub = _induce_subgraph(graph, subset)
    if not sub.v_labels and not sub.w_labels:
        return SubsetSymmetry(None, None)

    # Column fingerprints for π derivation.
    row_order = sub.u_local
    all_labels = sub.v_labels | sub.w_labels
    col_of: dict[str, tuple[int, ...]] = {}
    for label in all_labels:
        col_of[label] = tuple(graph.incidence[u].get(label, 0) for u in row_order)

    fp_to_labels: dict[tuple[int, ...], set[str]] = {}
    for lbl, fp in col_of.items():
        fp_to_labels.setdefault(fp, set()).add(lbl)

    # Collect exact π generators via σ-loop.
    v_perms, w_perms = _collect_pi_permutations(
        graph, sub, row_order, col_of, fp_to_labels
    )
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))

    # Build V-side group.
    v_group: PermutationGroup | None = None
    if v_perms:
        v_group = PermutationGroup(*v_perms, axes=tuple(range(len(v_sorted))))
        v_group._labels = v_sorted
    elif sub.v_labels and len(sub.v_labels) >= 2:
        # Fingerprint fast path: labels that share a fingerprint with at least
        # one other V label are symmetry-related.  Check for a declared group
        # covering those labels before defaulting to S_k.
        fp_groups: dict[tuple[int, ...], list[str]] = {}
        for lbl in sub.v_labels:
            fp_groups.setdefault(col_of[lbl], []).append(lbl)
        v_equiv = sorted(
            lbl for grp in fp_groups.values() if len(grp) >= 2 for lbl in grp
        )
        if len(v_equiv) >= 2:
            declared = _find_declared_group_for_labels(graph, subset, tuple(v_equiv))
            if declared is not None:
                v_group = declared
                v_group._labels = tuple(v_equiv)
            else:
                v_group = PermutationGroup.symmetric(
                    len(v_equiv), axes=tuple(range(len(v_equiv)))
                )
                v_group._labels = tuple(v_equiv)

    # Build W-side group.
    w_group: PermutationGroup | None = None
    if w_perms:
        w_group = PermutationGroup(*w_perms, axes=tuple(range(len(w_sorted))))
        w_group._labels = w_sorted
    elif sub.w_labels and len(sub.w_labels) >= 2:
        fp_groups_w: dict[tuple[int, ...], list[str]] = {}
        for lbl in sub.w_labels:
            fp_groups_w.setdefault(col_of[lbl], []).append(lbl)
        w_equiv = sorted(
            lbl for grp in fp_groups_w.values() if len(grp) >= 2 for lbl in grp
        )
        if len(w_equiv) >= 2:
            declared = _find_declared_group_for_labels(graph, subset, tuple(w_equiv))
            if declared is not None:
                w_group = declared
                w_group._labels = tuple(w_equiv)
            else:
                w_group = PermutationGroup.symmetric(
                    len(w_equiv), axes=tuple(range(len(w_equiv)))
                )
                w_group._labels = tuple(w_equiv)

    return SubsetSymmetry(output=v_group, inner=w_group)


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
