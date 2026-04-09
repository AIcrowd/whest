"""Determines if a contraction can use BLAS or not."""

from collections.abc import Sequence

from ._typing import ArrayIndexType

__all__ = ["can_blas"]


def _has_symmetric_input(
    inputs: list[str],
    input_symmetries: list | None,
) -> tuple[bool, bool]:
    """Check if any input has a symmetric group covering 2+ indices used in the contraction."""
    if input_symmetries is None:
        return False, False
    left_sym = False
    right_sym = False
    if input_symmetries[0]:
        input0_chars = set(inputs[0])
        for group in input_symmetries[0]:
            # group is frozenset[tuple[str, ...]] -- extract all chars from all blocks
            group_chars = frozenset(c for block in group for c in block)
            if len(group_chars & input0_chars) >= 2:
                left_sym = True
                break
    if input_symmetries[1]:
        input1_chars = set(inputs[1])
        for group in input_symmetries[1]:
            group_chars = frozenset(c for block in group for c in block)
            if len(group_chars & input1_chars) >= 2:
                right_sym = True
                break
    return left_sym, right_sym


def can_blas(
    inputs: list[str],
    result: str,
    idx_removed: ArrayIndexType,
    shapes: Sequence[tuple[int]] | None = None,
    input_symmetries: list | None = None,
) -> str | bool:
    """Checks if we can use a BLAS call.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation
    shapes : sequence of tuple[int], optional
        If given, check also that none of the indices are broadcast dimensions.
    input_symmetries : list of (IndexSymmetry or None), optional
        Symmetry groups for each input. When an input has a symmetric group
        covering 2+ of its indices, the BLAS classification is refined:
        GEMM → SYMM, GEMV/EINSUM → SYMV, DOT → SYDT.
        When None (default), behavior is identical to upstream.

    Returns:
    -------
    type : str or bool
        The type of BLAS call to be used or False if none.

    Notes:
    -----
    We assume several operations are not efficient such as a transposed
    DDOT, therefore 'ijk,jki->' should prefer einsum. These return the blas
    type appended with "/EINSUM" to differentiate when they can still be done
    with tensordot if required, e.g. when a backend has no einsum.

    Examples:
    --------
    >>> can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    >>> can_blas(['ab', 'cd'], 'abcd', set())
    'OUTER/EINSUM'

    >>> # looks like GEMM but actually 'j' is broadcast:
    >>> can_blas(['ij', 'jk'], 'ik', set('j'), shapes=[(4, 1), (5, 6)])
    False
    """
    # Can only do two
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # check for broadcast indices e.g:
    #     "ij,jk->ik" (but one of the 'j' dimensions is broadcast up)
    if shapes is not None:
        for c in idx_removed:
            if shapes[0][input_left.find(c)] != shapes[1][input_right.find(c)]:
                return False

    # Prefer einsum if not removing indices
    #     (N.B. tensordot outer faster for large arrays?)
    if len(idx_removed) == 0:
        base_result: str | bool = "OUTER/EINSUM"
    else:
        # Build a few temporaries
        sets = [set(x) for x in inputs]
        keep_left = sets[0] - idx_removed
        keep_right = sets[1] - idx_removed
        rs = len(idx_removed)

        # DDOT
        if inputs[0] == inputs[1]:
            base_result = "DOT"

        # DDOT does not make sense if you have to transpose - prefer einsum
        elif sets[0] == sets[1]:
            base_result = "DOT/EINSUM"

        # GEMM no transpose
        elif input_left[-rs:] == input_right[:rs]:
            base_result = "GEMM"

        # GEMM transpose both
        elif input_left[:rs] == input_right[-rs:]:
            base_result = "GEMM"

        # GEMM transpose right
        elif input_left[-rs:] == input_right[-rs:]:
            base_result = "GEMM"

        # GEMM transpose left
        elif input_left[:rs] == input_right[:rs]:
            base_result = "GEMM"

        # Einsum is faster than vectordot if we have to copy
        elif (len(keep_left) == 0) or (len(keep_right) == 0):
            base_result = "GEMV/EINSUM"

        # Conventional tensordot
        else:
            base_result = "TDOT"

    # Refine for symmetric inputs
    if input_symmetries is not None:
        left_sym, right_sym = _has_symmetric_input(inputs, input_symmetries)
        if left_sym or right_sym:
            if base_result == "GEMM":
                return "SYMM"
            elif base_result == "GEMV/EINSUM":
                return "SYMV"
            elif base_result == "DOT":
                return "SYDT"

    return base_result
