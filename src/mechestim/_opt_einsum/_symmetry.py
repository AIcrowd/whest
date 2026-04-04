"""Symmetry-aware cost functions for opt_einsum contraction paths.

This module provides:
- IndexSymmetry type for describing permutation symmetries of tensor indices
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
    result: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for sym in (sym1, sym2):
        if sym is None:
            continue
        for group in sym:
            surviving = group & k12
            if len(surviving) >= 2 and surviving not in seen:
                seen.add(surviving)
                result.append(surviving)
    return result if result else None


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
    - For each input symmetry group, find the *surviving* subgroup — the
      indices that appear in ``output_indices`` (i.e. are not summed away).
      Only the surviving subgroup benefits from the symmetry reduction
      ``C(n+k-1, k) / n^k``, because summed indices interact with the other
      factor and every value must still be visited.
    - Divide by the output symmetry factor (we only need to compute each
      unique output element once).
    - Return at least 1.

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
        Symmetry info for each input tensor.
    output_symmetry : IndexSymmetry | None, optional
        Symmetry of the output tensor.
    output_indices : frozenset of str or None, optional
        Indices that survive into the output of this contraction step.
        When provided, only the intersection of each symmetry group with
        these indices benefits from symmetry reduction.  When *None*,
        falls back to the legacy behaviour (intersect with
        ``idx_contraction``).

    Returns
    -------
    int
        Estimated FLOP count.
    """
    # Dense baseline
    cost = flop_count(idx_contraction, inner, num_terms, size_dictionary)

    idx_set = frozenset(idx_contraction)

    # Determine which indices to use for the surviving-subgroup intersection.
    survive_set = output_indices if output_indices is not None else idx_set

    # Scale down for input symmetries
    if input_symmetries:
        for sym in input_symmetries:
            if sym is None:
                continue
            for group in sym:
                active = group & survive_set
                if len(active) < 2:
                    continue
                n = next(size_dictionary[idx] for idx in active)
                k = len(active)
                # Ratio of unique tuples to total tuples
                cost = cost * comb(n + k - 1, k) // (n**k)

    # Scale down for output symmetry
    if output_symmetry:
        sf = symmetry_factor(output_symmetry)
        cost = cost // sf

    return max(cost, 1)
