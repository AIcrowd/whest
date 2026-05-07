"""Public einsum_accumulation_cost — pure inspection function."""

from __future__ import annotations

from typing import Any

import numpy as _np

from flopscope._opt_einsum._parser import parse_einsum_input
from flopscope._symmetric import SymmetricTensor

from ._cost import AccumulationCost, compute_accumulation_cost


def _per_op_symmetries(operands):
    """Extract per-operand SymmetryGroup from SymmetricTensor inputs."""
    out = []
    for op in operands:
        if isinstance(op, SymmetricTensor) and op.symmetry is not None:
            out.append(op.symmetry)
        else:
            out.append(None)
    return tuple(out)


def _identity_pattern(operands):
    """Group operand positions sharing object id (for repeated-operand wreath)."""
    id_to_positions: dict[int, list[int]] = {}
    for idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(idx)
    groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )
    return groups if groups else None


def einsum_accumulation_cost(
    subscripts: str,
    *operands,
    partition_budget: int | None = None,
) -> AccumulationCost:
    """Compute the symmetry-aware direct-event cost of an einsum expression.

    Returns the path-independent total = (k-1)·∏ M_a + ∏ α_a, decomposed
    per component, with the regime that fired and the full ladder trace.

    This function is for inspection and debugging — it does not execute the
    einsum. To execute, use ``flopscope.einsum()``, which charges this same
    cost against the active BudgetContext.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g. ``'ij,jk->ik'``).
    *operands : numpy.ndarray or SymmetricTensor
        Input arrays. SymmetricTensor inputs contribute their declared symmetry
        to the detected pointwise group G_pt.
    partition_budget : int, optional
        Maximum number of typed partitions per component before the
        partition-count regime refuses. Defaults to
        ``flopscope.get_setting('partition_budget')`` (typically 100 000).

    Returns
    -------
    AccumulationCost
        Whole-einsum cost decomposition.
    """
    input_subscripts, output_subscript, _ = parse_einsum_input((subscripts, *operands))
    canonical_subscripts = f'{input_subscripts}->{output_subscript}'
    input_parts = tuple(input_subscripts.split(','))
    shapes = tuple(tuple(_np.asarray(op).shape) for op in operands)

    return compute_accumulation_cost(
        canonical_subscripts=canonical_subscripts,
        input_parts=input_parts,
        output_subscript=output_subscript,
        shapes=shapes,
        per_op_symmetries=_per_op_symmetries(operands),
        identity_pattern=_identity_pattern(operands),
        partition_budget=partition_budget,
    )
