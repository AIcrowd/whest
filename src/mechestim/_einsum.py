"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

import numpy as _np

from mechestim._perm_group import PermutationGroup
from mechestim._symmetric import SymmetricTensor, validate_symmetry
from mechestim._validation import check_nan_inf, require_budget


def _symmetry_info_to_perm_groups(sym_info, subscript_chars: str):
    """Convert SymmetryInfo (positional axes) to label-indexed PermutationGroups.

    Returns a list of PermutationGroup objects with _labels set to the
    corresponding einsum characters, or None if no symmetry.
    """
    if sym_info is None:
        return None
    groups = []
    for group in sym_info.groups:
        if group.axes is None or group.degree < 2:
            continue
        labels = tuple(subscript_chars[ax] for ax in group.axes)
        new_group = PermutationGroup(*group.generators)
        new_group._labels = labels
        groups.append(new_group)
    return groups if groups else None


def _execute_pairwise(path_info, operands: list):
    """Execute pairwise contractions according to the optimized path."""
    ops = list(operands)
    for contract_inds, step in zip(path_info.path, path_info.steps):
        # Pop operands in reverse sorted order (same as opt_einsum convention)
        inds = sorted(contract_inds, reverse=True)
        tensors = [ops.pop(i) for i in inds]
        result = _np.einsum(step.subscript, *tensors)
        ops.append(result)
    return ops[0]


def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    optimize: str | bool | list = "auto",
    symmetric_axes: list[tuple[int, ...]] | None = None,
    symmetry: PermutationGroup | list[PermutationGroup] | None = None,  # NEW
    **kwargs,
) -> _np.ndarray:
    """Evaluate Einstein summation with FLOP counting and optional path optimization.

    Wraps ``numpy.einsum`` with analytical FLOP cost computation and
    optional symmetry savings. If any input is a ``SymmetricTensor``,
    the cost is automatically reduced. If ``symmetric_axes`` is provided
    and the output passes validation, a ``SymmetricTensor`` is returned.

    All contractions go through opt_einsum's ``contract_path`` to find an
    optimal pairwise decomposition. The FLOP cost uses opt_einsum's cost
    model which includes ``op_factor`` (multiply-add = 2 FLOPs for inner
    products).

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g., ``'ij,jk->ik'``).
    *operands : numpy.ndarray
        Input arrays. ``SymmetricTensor`` inputs are detected automatically
        for cost savings.
    optimize : str, bool, or list of tuple, optional
        Contraction path strategy. Default ``'auto'``.

        - ``'auto'``, ``'greedy'``, ``'optimal'``, ``'dp'``, etc.:
          Use the named algorithm to find the best path.
        - A list of int-tuples (e.g. ``[(1, 2), (0, 1)]``): use this
          explicit contraction path. Obtain one from ``me.einsum_path()``
          or construct manually. Each tuple names the operand positions
          to contract at that step; the result is appended to the end.
        - ``False``: treated as ``'auto'``.
    symmetric_axes : list of tuple of int, optional
        **Output** dimension symmetry groups. Declares that the result
        is symmetric in the given axes and wraps it as a
        ``SymmetricTensor``. For example, ``[(0, 1)]`` means the output
        satisfies ``result[i,j,...] == result[j,i,...]``. This does NOT
        declare input symmetry — use ``me.as_symmetric()`` for that.

    Returns
    -------
    numpy.ndarray or SymmetricTensor
        The result of the einsum.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetric_axes`` is provided but the result is not symmetric.
    """
    if symmetric_axes is not None and symmetry is not None:
        raise ValueError("symmetric_axes and symmetry are mutually exclusive")

    budget = require_budget()
    shapes = [op.shape for op in operands]

    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]

    input_parts = subscripts.split("->")[0].split(",")
    output_str = subscripts.split("->")[1] if "->" in subscripts else ""

    perm_groups = [
        _symmetry_info_to_perm_groups(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]

    from mechestim._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

    oracle = SubgraphSymmetryOracle(
        operands=list(operands),
        subscript_parts=input_parts,
        per_op_groups=perm_groups,
        output_chars=output_str,
    )

    from mechestim._opt_einsum import contract_path as _contract_path

    path, path_info = _contract_path(
        subscripts,
        *shapes,
        shapes=True,
        optimize=optimize if optimize is not False else "auto",
        symmetry_oracle=oracle,
    )

    budget.deduct(
        "einsum",
        flop_cost=path_info.optimized_cost,
        subscripts=subscripts,
        shapes=tuple(shapes),
    )

    # Execute pairwise steps
    result = _execute_pairwise(path_info, list(operands))

    # Handle output symmetry wrapping
    if symmetry is not None and isinstance(result, _np.ndarray) and result.ndim >= 2:
        from mechestim._symmetric import validate_symmetry_groups

        perm_groups = (
            [symmetry] if isinstance(symmetry, PermutationGroup) else list(symmetry)
        )
        validate_symmetry_groups(result, perm_groups)
        sym_axes = [g.axes for g in perm_groups if g.axes is not None]
        result = SymmetricTensor(result, sym_axes, perm_groups=perm_groups)
    elif symmetric_axes and isinstance(result, _np.ndarray) and result.ndim >= 2:
        validate_symmetry(result, symmetric_axes)
        result = SymmetricTensor(result, symmetric_axes=symmetric_axes)

    check_nan_inf(result, "einsum")
    return result


def einsum_path(subscripts: str, *operands, optimize: str | bool | list = "auto"):
    """Compute the optimal contraction path without executing.

    Returns ``(path, PathInfo)`` with zero budget cost. The returned
    ``path`` can be passed back to ``me.einsum(..., optimize=path)``
    to execute with that exact contraction order.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    *operands : numpy.ndarray
        Input arrays.
    optimize : str, bool, or list of tuple, optional
        Path optimization strategy. Default ``'auto'``.

    Returns
    -------
    path : list of tuple of int
        The contraction path. Pass to ``me.einsum(..., optimize=path)``.
    info : PathInfo
        Diagnostics including per-step costs and symmetry savings.
    """
    budget = require_budget()
    budget.deduct("einsum_path", flop_cost=1, subscripts=None, shapes=())

    shapes = [op.shape for op in operands]
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]
    input_parts = subscripts.split("->")[0].split(",")
    output_str = subscripts.split("->")[1] if "->" in subscripts else ""

    perm_groups = [
        _symmetry_info_to_perm_groups(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]

    from mechestim._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

    oracle = SubgraphSymmetryOracle(
        operands=list(operands),
        subscript_parts=input_parts,
        per_op_groups=perm_groups,
        output_chars=output_str,
    )

    from mechestim._opt_einsum import contract_path as _contract_path

    path, path_info = _contract_path(
        subscripts,
        *shapes,
        shapes=True,
        optimize=optimize if optimize is not False else "auto",
        symmetry_oracle=oracle,
    )
    return list(path), path_info


import sys as _sys  # noqa: E402

from mechestim._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__], skip_names={"einsum_path"})
