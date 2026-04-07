"""Symmetry-aware cost functions for opt_einsum contraction paths.

This module provides:
- IndexSymmetry type for describing permutation symmetries of tensor indices
- restrict_group: restrict a symmetry group to surviving indices
- merge_two / pick_stronger: helpers for combining overlapping groups
- merge_overlapping_groups: connected-component merge of symmetry groups
- propagate_symmetry: track which symmetries survive a pairwise contraction
- unique_elements / compute_unique_size: count distinct elements under symmetry
- symmetric_flop_count: FLOP estimate that accounts for input/output symmetry

No imports from mechestim — only stdlib (math) and relative imports from _opt_einsum.
"""

from __future__ import annotations

from math import comb
from typing import Collection

from ._helpers import flop_count

# Type alias: a list of frozensets, each frozenset names symmetry-equivalent
# "blocks" of indices. Each block is a tuple of index characters. Per-index
# groups use 1-tuples (e.g. frozenset({('i',), ('j',)}) is S2{i,j}). Block
# groups use k-tuples for k >= 2 (e.g. frozenset({('j','k'),('l','m')}) is
# block S2 on blocks (j,k) and (l,m)).
#
# Invariants:
#   1. All tuples in a given frozenset have the same length (uniform block size).
#   2. Indices used across all blocks in a single group are disjoint.
#   3. At least 2 blocks per group (singletons are dropped at construction).
IndexSymmetry = list[frozenset[tuple[str, ...]]]


def restrict_group(
    group: frozenset[tuple[str, ...]],
    surviving_indices: frozenset[str],
) -> frozenset[tuple[str, ...]] | None:
    """Restrict a symmetry group to indices that survive in the output.

    For per-index groups (block size 1), drops blocks whose single character
    isn't in ``surviving_indices``. Returns None if fewer than 2 blocks remain.

    For block groups (block size >= 2), uses a parallel-survival check:
    for each block-position, either all blocks have that position surviving
    (keep), or none do (drop), or it's mixed (block sym breaks -> return None).
    Surviving positions are projected to form new blocks; if the projection
    yields fewer than 2 distinct blocks, returns None.

    Parameters
    ----------
    group : frozenset of tuple of str
        A single symmetry group (tuples of indices). All tuples in the group
        must have the same length (uniform block size invariant).
    surviving_indices : frozenset of str
        Indices that survive in the output of the current step.

    Returns
    -------
    frozenset of tuple of str or None
        The restricted group, or None if the symmetry doesn't survive.
    """
    blocks = list(group)
    if not blocks:
        return None
    s = len(blocks[0])  # block size (invariant: uniform within a group)

    if s == 1:
        new_blocks = [b for b in blocks if b[0] in surviving_indices]
        if len(new_blocks) < 2:
            return None
        return frozenset(new_blocks)

    # Block case (s >= 2): parallel survival check
    surviving_positions = []
    for pos in range(s):
        present = [b[pos] in surviving_indices for b in blocks]
        if all(present):
            surviving_positions.append(pos)
        elif any(present):
            # Mixed: some blocks surviving at this pos, others not -> breaks
            return None
        # else: all contracted at this pos -- fine, position is gone

    if not surviving_positions:
        return None

    new_blocks = [tuple(b[pos] for pos in surviving_positions) for b in blocks]
    # Check for trivial collapse (all blocks became the same)
    if len(set(new_blocks)) < 2:
        return None
    return frozenset(new_blocks)


def merge_two(
    g1: frozenset[tuple[str, ...]],
    g2: frozenset[tuple[str, ...]],
) -> frozenset[tuple[str, ...]] | None:
    """Merge two overlapping symmetry groups via set union.

    Returns None when the groups have different block sizes (cannot be
    combined into a single group while preserving the uniform-block-size
    invariant). Callers should fall back to ``pick_stronger`` in that case.
    """
    s1 = len(next(iter(g1)))
    s2 = len(next(iter(g2)))
    if s1 != s2:
        return None
    return frozenset(g1 | g2)


def pick_stronger(
    g1: frozenset[tuple[str, ...]],
    g2: frozenset[tuple[str, ...]],
) -> frozenset[tuple[str, ...]]:
    """When two groups cannot merge cleanly, keep the one with larger
    block size (conservative: block symmetries typically give more savings
    per unit of index space than per-index symmetries)."""
    s1 = len(next(iter(g1)))
    s2 = len(next(iter(g2)))
    return g1 if s1 >= s2 else g2


def merge_overlapping_groups(
    candidates: list[frozenset[tuple[str, ...]]],
) -> list[frozenset[tuple[str, ...]]]:
    """Merge symmetry groups that share any individual index character.

    Uses a connected-components approach: each new group is combined with
    any already-merged group it overlaps with (sharing at least one index
    character). When the overlap is between groups of the same block size,
    we take the union of their block sets. When block sizes differ, we fall
    back to ``pick_stronger`` and keep only one -- documented limitation.

    Parameters
    ----------
    candidates : list of frozenset of tuple of str
        Candidate symmetry groups, possibly overlapping.

    Returns
    -------
    list of frozenset of tuple of str
        Merged groups. No two groups in the result share any index character.
    """
    merged: list[frozenset[tuple[str, ...]]] = []
    for g in candidates:
        g_chars = frozenset(c for block in g for c in block)
        overlapping_idx = [
            i
            for i, m in enumerate(merged)
            if any(c in g_chars for block in m for c in block)
        ]
        if not overlapping_idx:
            merged.append(g)
            continue

        combined = g
        for i in sorted(overlapping_idx, reverse=True):
            other = merged.pop(i)
            two = merge_two(combined, other)
            if two is None:
                combined = pick_stronger(combined, other)
            else:
                combined = two
        merged.append(combined)

    return merged


def propagate_symmetry(
    sym1: IndexSymmetry | None,
    k1: frozenset[str],
    sym2: IndexSymmetry | None,
    k2: frozenset[str],
    k12: frozenset[str],
    induced_output_symmetry: IndexSymmetry | None = None,
) -> IndexSymmetry | None:
    """Propagate symmetry through a pairwise contraction.

    Combines two sources of symmetry:
    1. Per-input symmetry (sym1, sym2) restricted to the indices that survive
       the contraction. Block groups degrade via restrict_group's parallel-check
       rule: parallel collapse -> per-index, non-parallel -> breaks.
    2. Induced output symmetry (global groups from equal-args detection)
       restricted to the same surviving set.

    Both sources are fed into merge_overlapping_groups to unify overlapping
    groups (pairwise S2 -> S3, block + per-index -> pick_stronger, etc.).

    Parameters
    ----------
    sym1, sym2 : IndexSymmetry or None
        Symmetry of the two input tensors, in the tuple-based format.
    k1, k2 : frozenset of str
        Index sets of the two input tensors.  Currently unused by the
        body (restriction is done via k12 and restrict_group), but kept
        in the signature for API stability and for potential future
        contracted-index symmetry optimisations.
    k12 : frozenset of str
        Index set of the output tensor (indices surviving this step).
    induced_output_symmetry : IndexSymmetry or None, optional
        Global output-level symmetry constraints from equal-args detection.
        Each group is restricted to k12 and merged with the per-input groups.

    Returns
    -------
    IndexSymmetry or None
        Symmetry of the output, or None if no symmetry survives.
    """
    candidates: list[frozenset[tuple[str, ...]]] = []
    seen: set[frozenset[tuple[str, ...]]] = set()

    # Source 1: restrict each input symmetry group
    for sym in (sym1, sym2):
        if sym is None:
            continue
        for group in sym:
            restricted = restrict_group(group, k12)
            if restricted is None or restricted in seen:
                continue
            seen.add(restricted)
            candidates.append(restricted)

    # Source 2: restrict each induced output constraint
    if induced_output_symmetry:
        for group in induced_output_symmetry:
            restricted = restrict_group(group, k12)
            if restricted is None or restricted in seen:
                continue
            seen.add(restricted)
            candidates.append(restricted)

    if not candidates:
        return None

    merged = merge_overlapping_groups(candidates)
    return merged if merged else None


def unique_elements(
    indices: frozenset[str],
    size_dict: dict[str, int],
    symmetry: IndexSymmetry | None,
) -> int:
    """Count distinct elements of a tensor with the given symmetry.

    Handles both per-index (tuples of length 1) and block (tuples of length >= 2)
    groups via the general stars-and-bars formula C(n^s + k - 1, k) where s is
    the block size and k is the number of blocks.

    For per-index groups (s=1), this reduces to the familiar C(n + k - 1, k).
    Free (non-symmetric) indices contribute their full size.

    Parameters
    ----------
    indices : frozenset of str
        All index labels of the tensor.
    size_dict : dict
        Mapping from index label to dimension size.
    symmetry : IndexSymmetry or None
        Symmetric groups, or None for a dense tensor.

    Returns
    -------
    int
        Number of unique elements.
    """
    if not indices:
        return 1

    accounted: set[str] = set()
    count = 1

    if symmetry:
        for group in symmetry:
            # Collect all characters across all blocks in this group
            all_chars = frozenset(c for block in group for c in block)
            if all_chars & accounted:
                # Already accounted for by an earlier group -- skip
                continue
            if not all_chars <= indices:
                # Some character in the group isn't in the tensor's index set
                continue

            blocks = list(group)
            if len(blocks) < 2:
                continue
            s = len(blocks[0])
            n = size_dict[blocks[0][0]]
            k = len(blocks)

            block_card = n**s
            count *= comb(block_card + k - 1, k)
            accounted |= all_chars

    for idx in indices:
        if idx not in accounted:
            count *= size_dict[idx]

    return count


# Alias
compute_unique_size = unique_elements


def symmetric_flop_count(
    idx_contraction: Collection[str],
    inner: bool,
    num_terms: int,
    size_dictionary: dict[str, int],
    *,
    input_symmetries: list[IndexSymmetry | None] | None = None,
    output_symmetry: IndexSymmetry | None = None,
    output_indices: frozenset[str] | None = None,
) -> int:
    """FLOP count that accounts for symmetric structure.

    Without any symmetry information this returns the same value as
    :func:`._helpers.flop_count`.

    Strategy:
    - Start with the dense FLOP count.
    - Reduce by the ratio ``unique_output / total_output``, where
      ``unique_output`` is the number of distinct output elements under
      ``output_symmetry`` (computed exactly via the stars-and-bars formula
      ``C(n+k-1, k)`` for each symmetric group of *k* indices of size *n*).
    - Return at least 1.

    The output symmetry fully determines the reduction.  Input symmetry on
    surviving indices creates the output symmetry (tracked by
    :func:`propagate_symmetry`), but it is not an independent source of
    savings -- applying both would double-count.  The ``input_symmetries``
    parameter is retained for API compatibility and for potential future
    optimisations on contracted-index symmetry (a genuinely separate concern).

    Parameters
    ----------
    idx_contraction : collection of str
        All indices involved in this contraction step.
    inner : bool
        Whether there is an inner (trace / summation) product.
    num_terms : int
        Number of input tensors being contracted.
    size_dictionary : dict
        Index label -> dimension size.
    input_symmetries : list of (IndexSymmetry | None), optional
        Symmetry info for each input tensor.  Currently unused for cost
        reduction (output_symmetry captures the savings), but kept for
        API stability and future contracted-index optimisations.
    output_symmetry : IndexSymmetry | None, optional
        Symmetry of the output tensor.
    output_indices : frozenset of str or None, optional
        Indices that survive into the output of this contraction step.

    Returns
    -------
    int
        Estimated FLOP count.
    """
    from ._helpers import compute_size_by_dict

    # Dense baseline
    cost = flop_count(idx_contraction, inner, num_terms, size_dictionary)

    # Reduce for output symmetry: only compute unique output elements.
    if output_symmetry and output_indices is not None:
        total = compute_size_by_dict(output_indices, size_dictionary)
        unique = unique_elements(output_indices, size_dictionary, output_symmetry)
        cost = cost * unique // total

    return max(cost, 1)
