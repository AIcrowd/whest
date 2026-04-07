"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

from itertools import combinations as _combinations

import numpy as _np

from mechestim._symmetric import SymmetricTensor, validate_symmetry
from mechestim._validation import check_nan_inf, require_budget


def _symmetry_info_to_index_symmetry(sym_info, subscript_chars: str):
    """Convert SymmetryInfo (positional axes) to IndexSymmetry (char labels).

    sym_info.symmetric_axes is like [(0, 1, 2)] -- positional indices.
    subscript_chars is like "ijk" -- the einsum subscript for this operand.
    Returns IndexSymmetry like [frozenset({('i',), ('j',), ('k',)})], or None.

    Per-index symmetries (as declared via SymmetricTensor) always map to
    1-tuple blocks in the new uniform tuple representation.
    """
    if sym_info is None:
        return None
    groups = []
    for group in sym_info.symmetric_axes:
        char_group = frozenset((subscript_chars[d],) for d in group)
        if len(char_group) >= 2:
            groups.append(char_group)
    return groups if groups else None


def _operand_has_permutation_as_sym(per_op_sym, orig_sub: str, new_sub: str) -> bool:
    """Check whether an operand's declared per-op symmetry already includes the
    permutation that relabels orig_sub → new_sub on its own indices.

    The permutation is determined positionally: position i of new_sub should equal
    the image of position i of orig_sub. We then check whether the induced
    permutation on index chars is a composition of the operand's declared groups.

    This is used as the self-mapping guard in _is_valid_symmetry: when an operand
    maps to itself under a non-trivial sigma, we require that the operand is
    already invariant under that relabeling.
    """
    if per_op_sym is None:
        return False
    # Build the mapping from original char to new char at matching positions
    perm: dict[str, str] = {}
    for a, b in zip(orig_sub, new_sub):
        if a in perm and perm[a] != b:
            return False  # inconsistent permutation
        perm[a] = b
    # Check that perm is the identity on any char not in a declared group,
    # and is consistent with some declared group otherwise.
    for a, b in perm.items():
        if a == b:
            continue
        # a and b must be in the same symmetric group
        in_same_group = any(
            any(a in block for block in group) and any(b in block for block in group)
            for group in per_op_sym
        )
        if not in_same_group:
            return False
    return True


def _is_valid_symmetry(
    sigma: dict[str, str],
    subscript_parts: list[str],
    operands: list,
    per_op_syms: list,
) -> bool:
    """Check whether a permutation sigma of output indices is a valid symmetry
    of the einsum contraction.

    sigma is a dict {a: b, b: a, ...} representing the index relabeling.
    Valid iff each relabeled operand can be matched to an original operand with
    the same index set AND the same Python identity. When an operand self-maps
    with a non-trivial relabel, the operand must have a declared symmetry
    covering that relabel.

    Parameters
    ----------
    sigma : dict[str, str]
        The proposed permutation of output indices.
    subscript_parts : list[str]
        The einsum subscript for each operand (e.g., ['ij', 'jk']).
    operands : list
        The actual operand objects (used for identity comparison via `is`).
    per_op_syms : list
        Per-operand declared symmetries (from SymmetricTensor), for the
        self-mapping guard.

    Returns
    -------
    bool
        True iff sigma is a valid symmetry of the contraction.
    """
    relabeled = [
        "".join(sigma.get(c, c) for c in subscript) for subscript in subscript_parts
    ]

    used: set[int] = set()
    for i, new_sub in enumerate(relabeled):
        new_idx_set = frozenset(new_sub)
        matched = False
        for j, orig_sub in enumerate(subscript_parts):
            if j in used:
                continue
            if frozenset(orig_sub) != new_idx_set:
                continue
            if operands[j] is not operands[i]:
                continue
            if i == j and new_sub != orig_sub:
                # Self-map with non-trivial relabel: require declared sym
                if not _operand_has_permutation_as_sym(
                    per_op_syms[i], orig_sub, new_sub
                ):
                    continue
            used.add(j)
            matched = True
            break
        if not matched:
            return False
    return True


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
    budget = require_budget()
    shapes = [op.shape for op in operands]

    # Extract symmetry info from SymmetricTensor inputs
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]

    # Convert SymmetryInfo -> IndexSymmetry for the path optimizer
    input_parts = subscripts.split("->")[0].split(",")
    index_symmetries = [
        _symmetry_info_to_index_symmetry(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]
    has_symmetry = any(s is not None for s in index_symmetries)

    # Detect induced output symmetry from operand identity (e.g. `einsum(..., X, X)`)
    output_str = subscripts.split("->")[1] if "->" in subscripts else ""
    induced_output_symmetry = _detect_induced_output_symmetry(
        operands=list(operands),
        subscript_parts=input_parts,
        output_chars=output_str,
        per_op_syms=index_symmetries,
    )
    has_induced = induced_output_symmetry is not None

    from mechestim._opt_einsum import contract_path as _contract_path

    path, path_info = _contract_path(
        subscripts,
        *shapes,
        shapes=True,
        optimize=optimize if optimize is not False else "auto",
        input_symmetries=index_symmetries if (has_symmetry or has_induced) else None,
        induced_output_symmetry=induced_output_symmetry,
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
    if symmetric_axes and isinstance(result, _np.ndarray) and result.ndim >= 2:
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
    shapes = [op.shape for op in operands]
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]
    input_parts = subscripts.split("->")[0].split(",")
    index_symmetries = [
        _symmetry_info_to_index_symmetry(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]
    has_symmetry = any(s is not None for s in index_symmetries)

    # Detect induced output symmetry from operand identity (e.g. `einsum(..., X, X)`)
    output_str = subscripts.split("->")[1] if "->" in subscripts else ""
    induced_output_symmetry = _detect_induced_output_symmetry(
        operands=list(operands),
        subscript_parts=input_parts,
        output_chars=output_str,
        per_op_syms=index_symmetries,
    )
    has_induced = induced_output_symmetry is not None

    from mechestim._opt_einsum import contract_path as _contract_path

    path, path_info = _contract_path(
        subscripts,
        *shapes,
        shapes=True,
        optimize=optimize if optimize is not False else "auto",
        input_symmetries=index_symmetries if (has_symmetry or has_induced) else None,
        induced_output_symmetry=induced_output_symmetry,
    )
    return list(path), path_info


def _enumerate_per_index_candidates(output_chars: str) -> list[dict[str, str]]:
    """Enumerate all per-index transposition sigmas for candidate S2 groups.

    For each pair of output indices (a, b), produce the swap sigma {a: b, b: a}.

    Parameters
    ----------
    output_chars : str
        The output subscript characters (e.g., 'ijk').

    Returns
    -------
    list of dict[str, str]
        Each dict is a proposed sigma. Pass each to _is_valid_symmetry.
    """
    chars = list(dict.fromkeys(output_chars))  # preserve order, dedupe
    return [{a: b, b: a} for a, b in _combinations(chars, 2)]


def _enumerate_block_candidates(
    operands: list,
    subscript_parts: list[str],
    output_chars: frozenset[str],
) -> list[dict[str, str]]:
    """Enumerate block-swap sigmas for pairs of equal operands.

    For each pair (i, j) of operand positions where operands[i] is operands[j]
    and each operand contributes the same-size block of free indices to the
    output, produce the block-swap sigma that aligns the free indices positionally.

    "Free to operand i only" means: in operand i's subscript AND in the output
    AND NOT in any other operand's subscript (including the other equal operand).

    Parameters
    ----------
    operands : list
        Original operand objects.
    subscript_parts : list[str]
        Einsum subscript per operand.
    output_chars : frozenset[str]
        The final output's index characters.

    Returns
    -------
    list of dict[str, str]
        Candidate sigmas for block-swap induction.
    """
    candidates: list[dict[str, str]] = []
    n = len(operands)

    for i, j in _combinations(range(n), 2):
        if operands[i] is not operands[j]:
            continue

        sub_i = subscript_parts[i]
        sub_j = subscript_parts[j]
        other_subs = [subscript_parts[k] for k in range(n) if k != i and k != j]
        other_chars = (
            frozenset().union(*[frozenset(s) for s in other_subs])
            if other_subs
            else frozenset()
        )

        # Free to op i only: in sub_i, in output, not in sub_j or any other op
        sub_i_set = frozenset(sub_i)
        sub_j_set = frozenset(sub_j)
        free_i_set = sub_i_set - sub_j_set - other_chars
        free_j_set = sub_j_set - sub_i_set - other_chars
        free_i_set = free_i_set & output_chars
        free_j_set = free_j_set & output_chars

        if len(free_i_set) != len(free_j_set) or len(free_i_set) == 0:
            continue

        # Positional pairing: enumerate free indices in the order they appear
        # in each operand's subscript.
        free_i_ordered = [c for c in sub_i if c in free_i_set]
        free_j_ordered = [c for c in sub_j if c in free_j_set]

        sigma: dict[str, str] = {}
        for a, b in zip(free_i_ordered, free_j_ordered):
            sigma[a] = b
            sigma[b] = a
        candidates.append(sigma)

    return candidates


def _sigma_to_group(sigma: dict[str, str]) -> frozenset[tuple[str, ...]] | None:
    """Convert a sigma (char-level permutation) to a symmetry group.

    The sigma represents a permutation; we convert to the orbits of that
    permutation, then group the orbits by their size into blocks.

    For a simple transposition {a:b, b:a}, the orbit is {a, b} — two separate
    1-element blocks → per-index S2 = frozenset({(a,), (b,)}).

    For a composition like {j:l, l:j, k:m, m:k} (two disjoint transpositions
    forming a block swap), the orbits are {j, l} and {k, m}, grouped by
    positional correspondence into a single block group
    frozenset({(j, k), (l, m)}).

    IMPORTANT: the positional correspondence must be recoverable from how the
    sigma was built. We rely on the ordered keys in the dict to preserve
    block structure: sigma[a_0] = b_0, sigma[a_1] = b_1, ... → blocks (a_0,a_1)
    and (b_0,b_1).
    """
    # Split the sigma into pairs: (a, b) where a < b in insertion order
    seen: set[str] = set()
    a_list: list[str] = []
    b_list: list[str] = []
    for a, b in sigma.items():
        if a in seen or b in seen:
            continue
        seen.add(a)
        seen.add(b)
        a_list.append(a)
        b_list.append(b)
    if not a_list:
        return None
    return frozenset({tuple(a_list), tuple(b_list)})


def _detect_induced_output_symmetry(
    operands: list,
    subscript_parts: list[str],
    output_chars: str,
    per_op_syms: list,
) -> list[frozenset[tuple[str, ...]]] | None:
    """Detect output symmetry induced by Python-identity equal operands.

    Enumerates per-index and block candidate sigmas, validates each, converts
    valid sigmas to symmetry groups, and merges overlapping groups. Returns
    None if no induction applies.

    Parameters
    ----------
    operands : list
        Original operand objects (for `is` comparison).
    subscript_parts : list[str]
        Einsum subscript per operand.
    output_chars : str
        Final output subscript.
    per_op_syms : list
        Per-operand declared symmetries for the self-mapping guard.

    Returns
    -------
    IndexSymmetry or None
        The induced output symmetry, merged into a list of groups.
    """
    if len(operands) < 2 or not output_chars:
        return None

    output_set = frozenset(output_chars)

    candidates: list[frozenset[tuple[str, ...]]] = []

    # Per-index candidates
    for sigma in _enumerate_per_index_candidates(output_chars):
        if _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms):
            group = _sigma_to_group(sigma)
            if group is not None:
                candidates.append(group)

    # Block candidates
    for sigma in _enumerate_block_candidates(operands, subscript_parts, output_set):
        if _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms):
            group = _sigma_to_group(sigma)
            if group is not None and group not in candidates:
                candidates.append(group)

    if not candidates:
        return None

    # Merge overlapping groups via the existing primitive
    from mechestim._opt_einsum._symmetry import merge_overlapping_groups

    merged = merge_overlapping_groups(candidates)
    return merged if merged else None


import sys as _sys  # noqa: E402

from mechestim._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__], skip_names={"einsum_path"})
