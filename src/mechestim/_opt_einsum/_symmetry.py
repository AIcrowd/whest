"""Symmetry-aware cost helpers for opt_einsum contraction paths.

This module provides:
- unique_elements / compute_unique_size: count distinct elements under symmetry.
- symmetric_flop_count: FLOP estimate reduced by output symmetry.

Detection of symmetries is handled by ``_subgraph_symmetry.SubgraphSymmetryOracle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Collection

from mechestim._perm_group import PermutationGroup

from ._helpers import flop_count


@dataclass(frozen=True)
class SubsetSymmetry:
    """Symmetry info for one contraction subset, split by V/W.

    Attributes
    ----------
    output : PermutationGroup or None
        Exact permutation group for the output (V) labels.
    inner : PermutationGroup or None
        Exact permutation group for the inner (W) labels.
    """

    output: PermutationGroup | None
    inner: PermutationGroup | None


def unique_elements(
    indices: frozenset[str],
    size_dict: dict[str, int],
    perm_group: PermutationGroup | None = None,
) -> int:
    """Count distinct elements of a tensor with the given symmetry.

    When perm_group is provided, uses Burnside's lemma for exact counting.
    When None, returns the dense count (product of all sizes).
    """
    if not indices:
        return 1

    if perm_group is not None:
        if perm_group._labels is not None:
            label_list = list(perm_group._labels)
        else:
            label_list = sorted(indices)[: perm_group.degree]
        pg_size_dict: dict[int, int] = {}
        accounted: set[str] = set()
        for i, lbl in enumerate(label_list):
            pg_size_dict[i] = size_dict[lbl]
            accounted.add(lbl)
        count = perm_group.burnside_unique_count(pg_size_dict)
        for idx in indices:
            if idx not in accounted:
                count *= size_dict[idx]
        return count

    # No symmetry -- dense count.
    count = 1
    for idx in indices:
        count *= size_dict[idx]
    return count


compute_unique_size = unique_elements


def symmetric_flop_count(
    idx_contraction: Collection[str],
    inner: bool,
    num_terms: int,
    size_dictionary: dict[str, int],
    *,
    output_group: PermutationGroup | None = None,
    output_indices: frozenset[str] | None = None,
    inner_group: PermutationGroup | None = None,
    inner_indices: frozenset[str] | None = None,
    use_inner_symmetry: bool = False,
    per_operand_free_counts: tuple[int, ...] | None = None,
) -> int:
    r"""FLOP count for a symmetric tensor contraction via the Phi algorithm.

    When ``per_operand_free_counts`` and ``inner_indices`` are provided,
    computes the cost of the symmetry-preserving algorithm
    (Solomonik & Demmel 2015, Theorem 5.4) with :math:`\mu = \nu = 1`.

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

    if output_group is not None and output_indices is not None:
        total = compute_size_by_dict(output_indices, size_dictionary)
        unique = unique_elements(
            output_indices, size_dictionary, perm_group=output_group
        )
        cost = cost * unique // total

    if use_inner_symmetry:
        if inner_group is not None and inner_indices is not None:
            total_inner = compute_size_by_dict(inner_indices, size_dictionary)
            if total_inner > 0:
                unique_inner = unique_elements(
                    inner_indices, size_dictionary, perm_group=inner_group
                )
                cost = cost * unique_inner // total_inner

    cost = max(cost, 1)

    # --- Phi (symmetry-preserving) estimate ---
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

        # Phi's intermediate Z-hat is fully symmetric across ALL omega indices by
        # construction (Solomonik & Demmel 2015, Algorithm 5.1).
        if len(all_indices) >= 2:
            full_group = PermutationGroup.symmetric(len(all_indices))
            full_group._labels = tuple(sorted(all_indices))
            unique_all = unique_elements(
                all_indices, size_dictionary, perm_group=full_group
            )
        else:
            unique_all = unique_elements(all_indices, size_dictionary)

        # Per-element cost: 1 mult + C(omega, free_i) adds per operand
        # + C(omega, v) adds for Z-hat->Z accumulation.
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
                if len(sub) >= 2:
                    sub_group = PermutationGroup.symmetric(len(sub))
                    sub_group._labels = tuple(sorted(sub))
                    phi_cost += unique_elements(
                        sub, size_dictionary, perm_group=sub_group
                    )
                else:
                    phi_cost += unique_elements(sub, size_dictionary)

        # Output symmetrization term: ((n, s+t)).
        st = sum(per_operand_free_counts)
        if st >= 2 and output_indices:
            out_sub = frozenset(sorted_labels[:st])
            out_group = PermutationGroup.symmetric(len(out_sub))
            out_group._labels = tuple(sorted(out_sub))
            phi_cost += unique_elements(out_sub, size_dictionary, perm_group=out_group)

        cost = min(cost, max(phi_cost, 1))

    return cost
