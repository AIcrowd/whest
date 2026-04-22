"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

import functools

import numpy as _np

from whest._config import get_setting
from whest._pointwise import _prepare_symmetric_out, _validate_result_symmetry
from whest._perm_group import SymmetryGroup
from whest._symmetric import SymmetricTensor
from whest._symmetry_utils import normalize_symmetry_input, validate_symmetry_group
from whest._validation import check_nan_inf, require_budget


def _symmetry_fingerprint(operands, input_parts):
    """Build a hashable symmetry fingerprint for cache keying.

    For each operand, captures None (no symmetry) or a tuple of
    (axes, generator_array_forms) per SymmetryGroup. This fully
    determines the symmetry structure without referencing tensor values.
    """
    parts = []
    for op, _chars in zip(operands, input_parts, strict=False):
        if not isinstance(op, SymmetricTensor) or op.symmetry is None:
            parts.append(None)
            continue
        group = op.symmetry
        axes = group.axes
        gens = tuple(tuple(gen.array_form) for gen in group.generators)
        parts.append(((axes, gens),))
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
    def _compute(
        subscripts,
        shapes,
        optimize,
        symmetry_fingerprint,
        identity_pattern,
        use_inner_symmetry=True,
    ):
        from whest._perm_group import _PermutationCompat as Permutation

        input_parts = subscripts.split("->")[0].split(",")
        output_str = subscripts.split("->")[1] if "->" in subscripts else ""

        perm_groups = []
        for fp_entry, chars in zip(symmetry_fingerprint, input_parts, strict=False):
            if fp_entry is None:
                perm_groups.append(None)
                continue
            groups = []
            for axes, gen_arrays in fp_entry:
                gens = [Permutation(list(g)) for g in gen_arrays]
                group = SymmetryGroup(*gens, axes=axes)
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
    """Clear the einsum path cache.

    Discards all cached contraction paths. Subsequent ``einsum()`` and
    ``einsum_path()`` calls will recompute paths from scratch.
    """
    _path_cache.cache_clear()


def einsum_cache_info():
    """Return einsum path cache statistics.

    Returns a named tuple with ``hits``, ``misses``, ``maxsize``, and
    ``currsize`` fields (the standard ``functools.lru_cache`` statistics).

    Example::

        info = we.einsum_cache_info()
        print(f"Cache hit rate: {info.hits / max(info.hits + info.misses, 1):.0%}")
    """
    return _path_cache.cache_info()


def _execute_pairwise(path_info, operands: list):
    """Execute pairwise contractions according to the optimized path."""
    ops = list(operands)
    for contract_inds, step in zip(path_info.path, path_info.steps, strict=False):
        # Pop operands in reverse sorted order (same as opt_einsum convention)
        inds = sorted(contract_inds, reverse=True)
        tensors = [ops.pop(i) for i in inds]
        result = _np.einsum(step.subscript, *tensors)
        ops.append(result)
    return ops[0]


def _normalize_optimize(optimize):
    if optimize is False:
        return "auto"
    if isinstance(optimize, list):
        return tuple(tuple(t) for t in optimize)
    return optimize


def _parse_einsum_parts(subscripts: str, operands):
    from whest._opt_einsum._parser import parse_einsum_input

    input_subscripts, output_subscript, _ = parse_einsum_input((subscripts, *operands))
    canonical_subscripts = f"{input_subscripts}->{output_subscript}"
    return canonical_subscripts, input_subscripts.split(","), output_subscript


def _get_path_info(subscripts: str, operands, optimize):
    canonical_subscripts, input_parts, output_subscript = _parse_einsum_parts(
        subscripts,
        operands,
    )
    shapes = tuple(tuple(op.shape) for op in operands)
    sym_fp = _symmetry_fingerprint(operands, input_parts)
    id_pat = _identity_pattern(operands)
    inner_sym = bool(get_setting("use_inner_symmetry"))
    path_info = _path_cache(
        canonical_subscripts,
        shapes,
        _normalize_optimize(optimize),
        sym_fp,
        id_pat,
        inner_sym,
    )
    return canonical_subscripts, input_parts, output_subscript, shapes, path_info


def _relabel_group_to_output(group, source_labels: tuple[str, ...], output_subscript: str):
    if group is None or not source_labels or not output_subscript:
        return None
    output_positions = {label: idx for idx, label in enumerate(output_subscript)}
    try:
        source_positions = tuple(output_positions[label] for label in source_labels)
    except KeyError:
        return None
    if len(set(source_positions)) != len(source_positions):
        return None

    order = tuple(sorted(range(len(source_positions)), key=source_positions.__getitem__))
    axes = tuple(source_positions[idx] for idx in order)
    source_to_sorted = {source_idx: sorted_idx for sorted_idx, source_idx in enumerate(order)}

    from whest._perm_group import _PermutationCompat as Permutation

    generators = []
    for gen in group.generators:
        generators.append(
            Permutation(
                [source_to_sorted[gen.array_form[source_idx]] for source_idx in order]
            )
        )

    remapped = SymmetryGroup(*generators, axes=axes)
    remapped._labels = tuple(output_subscript[axis] for axis in axes)
    return validate_symmetry_group(remapped, ndim=len(output_subscript))


def _remap_inferred_group(group, output_subscript: str):
    labels = getattr(group, "_labels", None)
    if group is None or labels is None:
        return None
    return _relabel_group_to_output(group, tuple(labels), output_subscript)


def _infer_pathless_output_symmetry(operands, input_parts, output_subscript: str):
    if len(operands) != 1:
        return None
    operand = operands[0]
    if not isinstance(operand, SymmetricTensor) or operand.symmetry is None:
        return None
    group = operand.symmetry
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    source_labels = tuple(input_parts[0][axis] for axis in axes)
    return _relabel_group_to_output(group, source_labels, output_subscript)


def _resolve_output_symmetry(
    *,
    symmetry,
    operands,
    input_parts,
    output_subscript: str,
    path_info,
):
    if symmetry is not None:
        return normalize_symmetry_input(symmetry, ndim=len(output_subscript))
    inferred = _infer_pathless_output_symmetry(operands, input_parts, output_subscript)
    if inferred is not None:
        return inferred
    if path_info.steps:
        inferred = _remap_inferred_group(path_info.steps[-1].output_group, output_subscript)
        if inferred is not None:
            return inferred
    return _infer_pathless_output_symmetry(operands, input_parts, output_subscript)


def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    out=None,
    optimize: str | bool | list = "auto",
    symmetry=None,
    **kwargs,
) -> _np.ndarray:
    """Evaluate Einstein summation with FLOP counting and optional path optimization.

    Wraps ``numpy.einsum`` with analytical FLOP cost computation and
    optional symmetry savings. If any input is a ``SymmetricTensor``,
    the cost is automatically reduced. If ``symmetry`` is provided and the output passes validation, a ``SymmetricTensor`` is returned.

    All contractions go through opt_einsum's ``contract_path`` to find an
    optimal pairwise decomposition. The FLOP cost uses opt_einsum's cost
    model where FMA = 1 operation (see ``_cost_model.FMA_COST``).

    Contraction paths are cached in a module-level LRU cache keyed on
    (subscripts, shapes, optimizer, symmetry structure, operand identity).
    Repeated calls with the same inputs skip path recomputation entirely.
    See ``clear_einsum_cache()`` and ``einsum_cache_info()``.

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
    symmetry : SymmetryGroup or symmetry shorthand, optional
        Declares output symmetry and wraps the validated result as a
        ``SymmetricTensor``. This does NOT declare input symmetry; use
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
        If ``symmetry`` is provided but the result
        does not satisfy the declared symmetry. Validation checks the
        data against each generator of the group.
    """
    budget = require_budget()
    canonical_subscripts, input_parts, output_subscript, shapes, path_info = _get_path_info(
        subscripts,
        operands,
        optimize,
    )
    target_symmetry = _resolve_output_symmetry(
        symmetry=symmetry,
        operands=operands,
        input_parts=input_parts,
        output_subscript=output_subscript,
        path_info=path_info,
    )
    effective_out_symmetry = target_symmetry
    if effective_out_symmetry is None and isinstance(out, SymmetricTensor):
        effective_out_symmetry = out.symmetry
    target_symmetry = _prepare_symmetric_out(out, effective_out_symmetry)

    with budget.deduct(
        "einsum",
        flop_cost=path_info.optimized_cost,
        subscripts=canonical_subscripts,
        shapes=tuple(shapes),
    ):
        if path_info.steps:
            result = _execute_pairwise(path_info, list(operands))
        else:
            result = _np.einsum(canonical_subscripts, *operands)

    if out is not None:
        _validate_result_symmetry(result, target_symmetry)
        _np.copyto(_np.asarray(out), _np.asarray(result), casting="unsafe")
        check_nan_inf(out, "einsum")
        return out

    if target_symmetry is not None:
        _validate_result_symmetry(result, target_symmetry)
        result = SymmetricTensor(_np.asarray(result), symmetry=target_symmetry)
    else:
        result = _aswhest(_np.asarray(result))

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
    with budget.deduct("einsum_path", flop_cost=1, subscripts=None, shapes=()):
        pass
    _canonical_subscripts, _input_parts, _output_subscript, _shapes, path_info = _get_path_info(
        subscripts,
        operands,
        optimize,
    )
    return list(path_info.path), path_info


import sys as _sys  # noqa: E402

from whest._ndarray import _aswhest, wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__], skip_names={"einsum", "einsum_path"})
