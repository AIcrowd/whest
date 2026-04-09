"""Symmetry-aware cost helpers for opt_einsum contraction paths.

This module provides:
- IndexSymmetry type alias for describing permutation symmetries.
- unique_elements / compute_unique_size: count distinct elements under symmetry.
- symmetric_flop_count: FLOP estimate reduced by output symmetry.

Detection of symmetries is handled by ``_subgraph_symmetry.SubgraphSymmetryOracle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, prod
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


@dataclass(frozen=True)
class SubsetSymmetry:
    """Symmetry info for one contraction subset, split by V/W.

    Attributes
    ----------
    output : IndexSymmetry or None
        V-side symmetry: symmetries of the output tensor's free labels.
    inner : IndexSymmetry or None
        W-side symmetry: symmetries among the contracted (summed) labels.
        Used for inner-sum FLOP reduction when ``use_inner_symmetry`` is
        enabled in ``symmetric_flop_count``.
    """

    output: IndexSymmetry | None
    inner: IndexSymmetry | None


def unique_elements(
    indices: frozenset[str],
    size_dict: dict[str, int],
    symmetry: IndexSymmetry | None,
) -> int:
    """Count distinct elements of a tensor with the given symmetry.

    Handles both per-index (tuples of length 1) and block (tuples of length >= 2)
    groups via the stars-and-bars formula ``C(block_card + k - 1, k)`` where
    ``block_card`` is the cardinality of one block (the product of axis sizes
    over the labels in the block) and ``k`` is the number of blocks in the group.

    For per-index groups (block size 1) this reduces to the familiar
    ``C(n + k - 1, k)``. For block groups with possibly heterogeneous axis
    sizes (e.g. block ``(a, b)`` with ``size[a] != size[b]``), the block
    cardinality is computed as the product over the labels in any single
    block. A well-formed block group requires every block to have the same
    per-position axis sizes (otherwise the swap is not a valid symmetry),
    so reading from ``blocks[0]`` is sufficient.

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
            k = len(blocks)
            # Block cardinality = product of axis sizes over the labels in
            # one block. For per-index groups (block size 1), this reduces
            # to the single label's dimension. For block groups with
            # heterogeneous axis sizes, this correctly handles each
            # position's actual dimension rather than assuming uniformity.
            block_card = prod(size_dict[c] for c in blocks[0])

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
    output_symmetry: IndexSymmetry | None = None,
    output_indices: frozenset[str] | None = None,
    inner_symmetry: IndexSymmetry | None = None,
    inner_indices: frozenset[str] | None = None,
    use_inner_symmetry: bool = False,
) -> int:
    """FLOP count reduced by output-tensor and inner-sum symmetry.

    Without any symmetry information this returns the same value as
    :func:`._helpers.flop_count`. With ``output_symmetry`` and
    ``output_indices``, the dense FLOP count is scaled by
    ``unique_output / total_output``.

    When ``use_inner_symmetry`` is True and ``inner_symmetry`` /
    ``inner_indices`` are provided, an additional multiplicative
    reduction of ``unique_inner / total_inner`` is applied.
    """
    from ._helpers import compute_size_by_dict

    cost = flop_count(idx_contraction, inner, num_terms, size_dictionary)

    if output_symmetry and output_indices is not None:
        total = compute_size_by_dict(output_indices, size_dictionary)
        unique = unique_elements(output_indices, size_dictionary, output_symmetry)
        cost = cost * unique // total

    if use_inner_symmetry and inner_symmetry and inner_indices is not None:
        total_inner = compute_size_by_dict(inner_indices, size_dictionary)
        if total_inner > 0:
            unique_inner = unique_elements(
                inner_indices, size_dictionary, inner_symmetry
            )
            cost = cost * unique_inner // total_inner

    return max(cost, 1)
