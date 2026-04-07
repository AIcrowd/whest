"""Symmetry-aware cost functions for opt_einsum contraction paths.

This module provides:
- IndexSymmetry type for describing permutation symmetries of tensor indices
- restrict_group: restrict a symmetry group to surviving indices
- propagate_symmetry: track which symmetries survive a pairwise contraction
- unique_elements / compute_unique_size: count distinct elements under symmetry
- symmetry_factor: product of factorial(group_size) for all symmetric groups
- symmetric_flop_count: FLOP estimate that accounts for input/output symmetry

No imports from mechestim — only stdlib (math) and relative imports from _opt_einsum.
"""

from __future__ import annotations

from math import comb, factorial
from typing import Collection

from ._helpers import flop_count

# Type alias: a list of frozensets, each frozenset names indices that are
# mutually symmetric (e.g. [frozenset("ij"), frozenset("kl")] means S2 x S2).
IndexSymmetry = list[frozenset[str]]


def restrict_group(
    group: frozenset[tuple[str, ...]],
    surviving_indices: frozenset[str],
) -> frozenset[tuple[str, ...]] | None:
    """Restrict a symmetry group to indices that survive in the output.

    For per-index groups (block size 1), drops blocks whose single character
    isn't in ``surviving_indices``. Returns None if fewer than 2 blocks remain.

    For block groups (block size >= 2), uses a parallel-survival check:
    for each block-position, either all blocks have that position surviving
    (keep), or none do (drop), or it's mixed (block sym breaks → return None).
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
            # Mixed: some blocks surviving at this pos, others not → breaks
            return None
        # else: all contracted at this pos — fine, position is gone

    if not surviving_positions:
        return None

    new_blocks = [
        tuple(b[pos] for pos in surviving_positions)
        for b in blocks
    ]
    # Check for trivial collapse (all blocks became the same)
    if len(set(new_blocks)) < 2:
        return None
    return frozenset(new_blocks)


def propagate_symmetry(
    sym1: IndexSymmetry | None,
    k1: frozenset[str],
    sym2: IndexSymmetry | None,
    k2: frozenset[str],
    k12: frozenset[str],
) -> IndexSymmetry | None:
    """Propagate symmetry through a pairwise contraction.

    For each symmetric group in the inputs, restrict it to the indices that
    survive in the output (k12). Groups that shrink below size 2 are dropped.

    Parameters
    ----------
    sym1, sym2 : IndexSymmetry or None
        Symmetry of the two input tensors.
    k1, k2 : frozenset of str
        Index sets of the two input tensors.
    k12 : frozenset of str
        Index set of the output tensor.

    Returns
    -------
    IndexSymmetry or None
        Symmetry of the output, or None if no symmetry survives.
    """
    candidates: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for sym in (sym1, sym2):
        if sym is None:
            continue
        for group in sym:
            surviving = group & k12
            if len(surviving) >= 2 and surviving not in seen:
                seen.add(surviving)
                candidates.append(surviving)

    # Merge overlapping groups.  Two symmetry groups that share an index
    # generate a larger permutation group on their union — e.g. S2{a,d}
    # and S2{a,e} together generate S3{a,d,e} (transpositions sharing an
    # element generate the full symmetric group).  Keeping them separate
    # causes unique_elements to double-count shared indices.  We merge
    # by computing connected components of the "shares an index" graph.
    if len(candidates) > 1:
        merged: list[frozenset[str]] = []
        for g in candidates:
            # Find all existing merged groups that overlap with g
            overlapping = [i for i, m in enumerate(merged) if m & g]
            if not overlapping:
                merged.append(g)
            else:
                # Union g with all overlapping groups
                combined = g
                for i in sorted(overlapping, reverse=True):
                    combined = combined | merged.pop(i)
                merged.append(combined)
        candidates = merged

    return candidates if candidates else None


def unique_elements(
    indices: frozenset[str],
    size_dict: dict[str, int],
    symmetry: IndexSymmetry | None,
) -> int:
    """Count distinct elements of a tensor with the given symmetry.

    For each symmetric group of k indices each of size n, the number of
    unique index tuples is C(n + k - 1, k)  (stars and bars).
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
            active = group & indices
            # Skip if already accounted (handles duplicate groups from
            # e.g. Hadamard dedup merging two tensors with the same symmetry)
            if active <= accounted:
                continue
            if len(active) < 2:
                continue
            # All indices in a symmetric group must have the same size.
            n = next(size_dict[idx] for idx in active)
            k = len(active)
            count *= comb(n + k - 1, k)
            accounted.update(active)

    # Free indices: full size
    for idx in indices:
        if idx not in accounted:
            count *= size_dict[idx]

    return count


# Alias
compute_unique_size = unique_elements


def symmetry_factor(symmetry: IndexSymmetry | None) -> int:
    """Product of factorial(len(group)) over all symmetric groups.

    This equals the order of the symmetry group (number of permutations
    that leave the tensor invariant).

    Parameters
    ----------
    symmetry : IndexSymmetry or None

    Returns
    -------
    int
    """
    if not symmetry:
        return 1
    result = 1
    for group in symmetry:
        result *= factorial(len(group))
    return result


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
    savings — applying both would double-count.  The ``input_symmetries``
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
