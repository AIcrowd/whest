"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

import functools

import numpy as _np

from whest._config import get_setting
from whest._perm_group import PermutationGroup
from whest._symmetric import SymmetricTensor, validate_symmetry
from whest._validation import check_nan_inf, require_budget


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


def _symmetry_fingerprint(operands, input_parts):
    """Build a hashable symmetry fingerprint for cache keying.

    For each operand, captures None (no symmetry) or a tuple of
    (axes, generator_array_forms) per PermutationGroup. This fully
    determines the symmetry structure without referencing tensor values.
    """
    parts = []
    for op, chars in zip(operands, input_parts):
        if not isinstance(op, SymmetricTensor) or op.symmetry_info is None:
            parts.append(None)
            continue
        groups = op.symmetry_info.groups or []
        group_fps = []
        for g in groups:
            axes = g.axes
            gens = tuple(tuple(gen.array_form) for gen in g.generators)
            group_fps.append((axes, gens))
        parts.append(tuple(group_fps) if group_fps else None)
    return tuple(parts)


def _identity_pattern(operands):
    """Build a hashable pattern of which operands are the same Python object.

    Returns None if all operands are distinct objects (common case).
    Otherwise returns a tuple of tuples, where each inner tuple lists
    positions sharing the same object identity (only groups of size >= 2).

    This mirrors the identical_operand_groups logic in _build_bipartite.
    """
    id_to_positions: dict[int, list[int]] = {}
    for idx, op in enumerate(operands):
        id_to_positions.setdefault(id(op), []).append(idx)
    groups = tuple(
        tuple(positions)
        for positions in id_to_positions.values()
        if len(positions) >= 2
    )
    return groups if groups else None


def _make_path_cache(maxsize):
    """Create a new lru_cache-wrapped path computation function."""

    @functools.lru_cache(maxsize=maxsize)
    def _compute(subscripts, shapes, optimize, symmetry_fingerprint, identity_pattern):
        from whest._perm_group import Permutation

        input_parts = subscripts.split("->")[0].split(",")
        output_str = subscripts.split("->")[1] if "->" in subscripts else ""

        perm_groups = []
        for fp_entry, chars in zip(symmetry_fingerprint, input_parts):
            if fp_entry is None:
                perm_groups.append(None)
                continue
            groups = []
            for axes, gen_arrays in fp_entry:
                gens = [Permutation(list(g)) for g in gen_arrays]
                group = PermutationGroup(*gens, axes=axes)
                labels = tuple(chars[ax] for ax in axes)
                group._labels = labels
                groups.append(group)
            perm_groups.append(groups if groups else None)

        # Build dummy operands for oracle (only shapes + identity matter)
        dummy_ops = [_np.empty(s) for s in shapes]
        # Re-alias dummies to match the identity_pattern
        if identity_pattern is not None:
            for group in identity_pattern:
                canonical = dummy_ops[group[0]]
                for pos in group[1:]:
                    dummy_ops[pos] = canonical

        from whest._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

        oracle = SubgraphSymmetryOracle(
            operands=dummy_ops,
            subscript_parts=input_parts,
            per_op_groups=perm_groups,
            output_chars=output_str,
        )

        from whest._opt_einsum import contract_path as _contract_path

        _path, path_info = _contract_path(
            subscripts,
            *shapes,
            shapes=True,
            optimize=optimize if not isinstance(optimize, tuple) else list(optimize),
            symmetry_oracle=oracle,
        )
        return path_info

    return _compute


_path_cache = _make_path_cache(4096)


def _rebuild_einsum_cache():
    """Rebuild the path cache with the current configured maxsize."""
    global _path_cache
    _path_cache = _make_path_cache(int(get_setting("einsum_path_cache_size")))


def clear_einsum_cache():
    """Clear the einsum path cache."""
    _path_cache.cache_clear()


def einsum_cache_info():
    """Return cache statistics (hits, misses, maxsize, currsize)."""
    return _path_cache.cache_info()


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
    model where FMA = 1 operation (see ``_cost_model.FMA_COST``).

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
        **Output** dimension symmetry groups (S_k only). Declares that the
        result is symmetric in the given axes and wraps it as a
        ``SymmetricTensor``. For example, ``[(0, 1)]`` means the output
        satisfies ``result[i,j,...] == result[j,i,...]``. This does NOT
        declare input symmetry — use ``me.as_symmetric()`` for that.
        Mutually exclusive with *symmetry*.
    symmetry : PermutationGroup or list of PermutationGroup, optional
        **Output** permutation group symmetry. Declares that the result
        is symmetric under the given ``PermutationGroup``(s) and wraps it
        as a ``SymmetricTensor``. Unlike *symmetric_axes* (which always
        means S_k), this supports any permutation group — for example,
        ``PermutationGroup.cyclic(3, axes=(0, 1, 2))`` declares cyclic
        symmetry where ``result[i,j,k] == result[j,k,i] == result[k,i,j]``
        but ``result[i,j,k]`` need not equal ``result[j,i,k]``.
        Each group must have ``axes`` set. Mutually exclusive with
        *symmetric_axes*. This does NOT declare input symmetry — use
        ``me.as_symmetric()`` for that.

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
        If ``symmetric_axes`` or ``symmetry`` is provided but the result
        does not satisfy the declared symmetry. Validation checks the
        data against each generator of the group.
    """
    if symmetric_axes is not None and symmetry is not None:
        raise ValueError("symmetric_axes and symmetry are mutually exclusive")

    budget = require_budget()
    shapes = tuple(tuple(op.shape) for op in operands)

    input_parts = subscripts.split("->")[0].split(",")

    # Build cache key components
    sym_fp = _symmetry_fingerprint(operands, input_parts)
    id_pat = _identity_pattern(operands)

    # Normalize optimize for hashability
    if optimize is False:
        opt_key = "auto"
    elif isinstance(optimize, list):
        opt_key = tuple(tuple(t) for t in optimize)
    else:
        opt_key = optimize

    # Get cached PathInfo (or compute on miss)
    path_info = _path_cache(subscripts, shapes, opt_key, sym_fp, id_pat)

    budget.deduct(
        "einsum",
        flop_cost=path_info.optimized_cost,
        subscripts=subscripts,
        shapes=shapes,
    )

    # Execute pairwise steps
    result = _execute_pairwise(path_info, list(operands))

    # Handle output symmetry wrapping
    if symmetry is not None and isinstance(result, _np.ndarray) and result.ndim >= 2:
        from whest._symmetric import validate_symmetry_groups

        perm_groups_out = (
            [symmetry] if isinstance(symmetry, PermutationGroup) else list(symmetry)
        )
        validate_symmetry_groups(result, perm_groups_out)
        sym_axes = [g.axes for g in perm_groups_out if g.axes is not None]
        result = SymmetricTensor(result, sym_axes, perm_groups=perm_groups_out)
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

    shapes = tuple(tuple(op.shape) for op in operands)
    input_parts = subscripts.split("->")[0].split(",")

    sym_fp = _symmetry_fingerprint(operands, input_parts)
    id_pat = _identity_pattern(operands)

    if optimize is False:
        opt_key = "auto"
    elif isinstance(optimize, list):
        opt_key = tuple(tuple(t) for t in optimize)
    else:
        opt_key = optimize

    path_info = _path_cache(subscripts, shapes, opt_key, sym_fp, id_pat)
    return list(path_info.path), path_info


import sys as _sys  # noqa: E402

from whest._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__], skip_names={"einsum_path"})
