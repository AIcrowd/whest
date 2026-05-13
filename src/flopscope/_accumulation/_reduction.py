"""Reduction-cost orchestrator + helpers.

Implements the two-tier reduction cost model:
- Tier 1 (ufunc.reduce): orbit-mapping α via compute_accumulation_cost
- Tier 2 (non-ufunc):   dense × (num_output_orbits / num_output_elems)

See .aicrowd/superpowers/specs/2026-05-13-symmetry-aware-reduction-cost-design.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flopscope._perm_group import SymmetryGroup


def _normalize_axis(
    axis: int | tuple[int, ...] | list[int] | None,
    ndim: int,
) -> tuple[int, ...]:
    """Normalize an axis specifier to a sorted tuple of non-negative ints.

    - None → all axes (full reduction)
    - int → singleton, with negative-index wrapping
    - tuple/list → sorted unique, with negative-index wrapping
    """
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(sorted({a % ndim for a in axis}))


def _num_output_orbits(
    input_shape: tuple[int, ...],
    axes_summed: tuple[int, ...],
    symmetry: SymmetryGroup | None,
) -> int:
    """Count orbits of the output multi-index under the output stabilizer.

    Output shape = input_shape with axes_summed removed.
    Output stabilizer = reduce_group(symmetry, ndim=ndim, axis=axes_summed).
    Returns the size-aware Burnside orbit count on the output shape.
    """
    ndim = len(input_shape)
    axes_set = frozenset(axes_summed)
    output_shape = tuple(input_shape[i] for i in range(ndim) if i not in axes_set)

    if not output_shape:
        return 1

    if symmetry is None:
        return math.prod(output_shape)

    from flopscope._symmetry_utils import reduce_group

    from ._burnside import size_aware_burnside

    output_group = reduce_group(
        symmetry,
        ndim=ndim,
        axis=axes_summed,
        keepdims=False,
    )

    if output_group is None:
        return math.prod(output_shape)

    elements = output_group.elements()
    return size_aware_burnside(elements, output_shape)
