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

from mechestim._perm_group import PermutationGroup

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
    output_group : PermutationGroup or None
        Exact permutation group for the output (V) labels.  When provided,
        Burnside's lemma is used instead of the stars-and-bars approximation.
    inner_group : PermutationGroup or None
        Exact permutation group for the inner (W) labels.
    """

    output: IndexSymmetry | None
    inner: IndexSymmetry | None
    output_group: PermutationGroup | None = None
    inner_group: PermutationGroup | None = None


def unique_elements(
    indices: frozenset[str],
    size_dict: dict[str, int],
    symmetry: IndexSymmetry | None = None,
    *,
    perm_group: PermutationGroup | None = None,
) -> int:
    """Count distinct elements of a tensor with the given symmetry.

    When ``perm_group`` is provided, Burnside's lemma is used for exact
    orbit counting.  This is more precise than the stars-and-bars
    approximation used for ``IndexSymmetry`` groups.

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
    perm_group : PermutationGroup or None
        When provided, use Burnside counting instead of stars-and-bars.
        The group's ``_labels`` attribute maps integer positions to string
        labels; if absent, ``sorted(indices)[:degree]`` is used.

    Returns
    -------
    int
        Number of unique elements.
    """
    if not indices:
        return 1

    # --- Burnside path ---
    if perm_group is not None:
        if perm_group._labels is not None:
            label_list = list(perm_group._labels)
        else:
            sorted_labels = sorted(indices)
            label_list = sorted_labels[: perm_group.degree]
        pg_size_dict: dict[int, int] = {}
        accounted_pg: set[str] = set()
        for i, lbl in enumerate(label_list):
            pg_size_dict[i] = size_dict[lbl]
            accounted_pg.add(lbl)
        count = perm_group.burnside_unique_count(pg_size_dict)
        # Free indices not covered by the group contribute their full size.
        for idx in indices:
            if idx not in accounted_pg:
                count *= size_dict[idx]
        return count

    # --- IndexSymmetry (stars-and-bars) path ---
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
    per_operand_free_counts: tuple[int, ...] | None = None,
    output_group: PermutationGroup | None = None,
    inner_group: PermutationGroup | None = None,
) -> int:
    r"""FLOP count for a symmetric tensor contraction via the Φ algorithm.

    When ``per_operand_free_counts`` and ``inner_indices`` are provided,
    computes the cost of the symmetry-preserving algorithm
    (Solomonik & Demmel 2015, Theorem 5.4) with :math:`\mu = \nu = 1`:

    .. math::

        F^\Phi = \binom{n+\omega-1}{\omega}
                 \bigl[1 + \tbinom{\omega}{s} + \tbinom{\omega}{t}
                 + \tbinom{\omega}{v}\bigr]
                 + \binom{n+s+v-1}{s+v}
                 + \binom{n+t+v-1}{t+v}
                 + \binom{n+s+t-1}{s+t}

    Falls back to the previous direct-evaluation estimate when the
    per-operand decomposition is unavailable or when there are no
    contracted indices (``v = 0``).

    Parameters
    ----------
    per_operand_free_counts : tuple of int, optional
        Number of free (non-contracted) indices contributed by each
        operand.  For a pairwise contraction this is ``(s, t)``.
    """
    from ._helpers import compute_size_by_dict

    # --- Direct-evaluation estimate ---
    cost = flop_count(idx_contraction, inner, num_terms, size_dictionary)

    # Use PermutationGroup (Burnside) when available, else fall back to IndexSymmetry.
    if output_group is not None and output_indices is not None:
        total = compute_size_by_dict(output_indices, size_dictionary)
        unique = unique_elements(
            output_indices, size_dictionary, perm_group=output_group
        )
        cost = cost * unique // total
    elif output_symmetry and output_indices is not None:
        total = compute_size_by_dict(output_indices, size_dictionary)
        unique = unique_elements(output_indices, size_dictionary, output_symmetry)
        cost = cost * unique // total

    if use_inner_symmetry:
        if inner_group is not None and inner_indices is not None:
            total_inner = compute_size_by_dict(inner_indices, size_dictionary)
            if total_inner > 0:
                unique_inner = unique_elements(
                    inner_indices, size_dictionary, perm_group=inner_group
                )
                cost = cost * unique_inner // total_inner
        elif inner_symmetry and inner_indices is not None:
            total_inner = compute_size_by_dict(inner_indices, size_dictionary)
            if total_inner > 0:
                unique_inner = unique_elements(
                    inner_indices, size_dictionary, inner_symmetry
                )
                cost = cost * unique_inner // total_inner

    cost = max(cost, 1)

    # --- Φ (symmetry-preserving) estimate ---
    # Only for pairwise contractions with contracted indices and uniform
    # index dimensions (the paper assumes n-dimensional symmetric tensors).
    _all_inds = (output_indices or frozenset()) | (inner_indices or frozenset())
    _dims = {size_dictionary[c] for c in _all_inds} if _all_inds else set()
    _uniform_dims = len(_dims) == 1

    if (
        per_operand_free_counts is not None
        and len(per_operand_free_counts) == 2
        and inner_indices
        and len(inner_indices) > 0
        and _uniform_dims
    ):
        v = len(inner_indices)
        omega = sum(per_operand_free_counts) + v
        all_indices = (output_indices or frozenset()) | inner_indices

        # Φ's intermediate Ẑ is fully symmetric across ALL ω indices by
        # construction (Solomonik & Demmel 2015, Algorithm 5.1).
        full_sym: IndexSymmetry = (
            [frozenset((lbl,) for lbl in all_indices)] if len(all_indices) >= 2 else []
        )
        unique_all = unique_elements(all_indices, size_dictionary, full_sym or None)

        # Per-element cost: 1 mult + C(ω, free_i) adds per operand
        # + C(ω, v) adds for Ẑ→Z accumulation.
        add_factor = comb(omega, v)
        for free_i in per_operand_free_counts:
            add_factor += comb(omega, free_i)
        phi_cost = unique_all * (1 + add_factor)

        # Lower-order terms: operand intermediates A^(p), B^(q).
        sorted_labels = sorted(all_indices)
        for free_i in per_operand_free_counts:
            order_i = free_i + v
            if order_i > 0 and order_i < omega:
                sub = frozenset(sorted_labels[:order_i])
                sub_sym: IndexSymmetry = (
                    [frozenset((lbl,) for lbl in sub)] if len(sub) >= 2 else []
                )
                phi_cost += unique_elements(sub, size_dictionary, sub_sym or None)

        # Output symmetrization term: ((n, s+t)).
        st = sum(per_operand_free_counts)
        if st >= 2 and output_indices:
            out_sub = frozenset(sorted_labels[:st])
            out_sym_phi: IndexSymmetry = [frozenset((lbl,) for lbl in out_sub)]
            phi_cost += unique_elements(out_sub, size_dictionary, out_sym_phi or None)

        cost = min(cost, max(phi_cost, 1))

    return cost
