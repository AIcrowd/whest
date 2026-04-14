"""Symmetric tensor support: SymmetryInfo, SymmetricTensor, as_symmetric."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from whest._config import get_setting
from whest._perm_group import PermutationGroup
from whest.errors import SymmetryError, SymmetryLossWarning


@dataclass(frozen=True)
class SymmetryInfo:
    """Metadata about tensor symmetry groups.

    Parameters
    ----------
    symmetric_axes : list of tuple of int
        Groups of dimension indices that are symmetric under permutation.
    shape : tuple of int
        Full tensor shape.
    groups : list of PermutationGroup, optional
        PermutationGroup objects carrying axis info. Auto-built from
        symmetric_axes when not supplied.
    """

    symmetric_axes: list[tuple[int, ...]] = field(default_factory=list)
    shape: tuple[int, ...] = ()
    groups: list | None = None

    def __post_init__(self) -> None:
        # Normalize each group to a sorted tuple.
        normalized = [tuple(sorted(g)) for g in self.symmetric_axes]
        # frozen=True means we must use object.__setattr__
        object.__setattr__(self, "symmetric_axes", normalized)

        if self.groups is None:
            auto_groups = [
                PermutationGroup.symmetric(len(g), axes=g)
                for g in normalized
                if len(g) >= 2
            ]
            object.__setattr__(self, "groups", auto_groups)

    @property
    def unique_elements(self) -> int:
        """Number of unique elements accounting for symmetry.

        Uses Burnside's lemma via PermutationGroup when groups are present.
        Free (non-symmetric) dims contribute their full size.
        The total is the product.
        """
        accounted: set[int] = set()
        result = 1
        for group in self.groups:
            axes = group.axes
            if axes is None:
                continue
            size_dict = {i: self.shape[axes[i]] for i in range(group.degree)}
            result *= group.burnside_unique_count(size_dict)
            accounted.update(axes)
        # Multiply by free dims.
        for i, s in enumerate(self.shape):
            if i not in accounted:
                result *= s
        return result

    @property
    def symmetry_factor(self) -> int:
        """Product of group orders for each symmetry group."""
        result = 1
        for group in self.groups:
            result *= group.order()
        return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_symmetry(
    data: np.ndarray,
    symmetric_axes: list[tuple[int, ...]],
) -> None:
    """Validate that *data* has the claimed symmetry.

    For each group, checks that all dims have equal sizes and that all
    pairwise transpositions are satisfied within tolerance.

    Raises
    ------
    SymmetryError
        If the data is not symmetric along the claimed axes.
    """
    for group in symmetric_axes:
        if len(group) < 2:
            continue
        # Check equal sizes.
        sizes = [data.shape[d] for d in group]
        if len(set(sizes)) != 1:
            raise SymmetryError(axes=group, max_deviation=float("inf"))
        # Check pairwise transpositions.
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                axes = list(range(data.ndim))
                axes[group[i]], axes[group[j]] = axes[group[j]], axes[group[i]]
                transposed = data.transpose(axes)
                if not np.allclose(data, transposed, atol=1e-6, rtol=1e-5):
                    max_dev = float(np.max(np.abs(data - transposed)))
                    raise SymmetryError(axes=group, max_deviation=max_dev)


def validate_symmetry_groups(data: np.ndarray, groups: list) -> None:
    """Validate that *data* is symmetric under the given PermutationGroups.

    Raises
    ------
    ValueError
        If a group has no axes set.
    SymmetryError
        If the data is not symmetric under the claimed group.
    """
    for group in groups:
        axes = group.axes
        if axes is None:
            axes = tuple(range(group.degree))
            group._axes = axes
        for orbit in group.orbits():
            sizes = {data.shape[axes[i]] for i in orbit}
            if len(sizes) != 1:
                raise SymmetryError(
                    axes=tuple(axes[i] for i in orbit), max_deviation=float("inf")
                )
        for gen in group.generators:
            if gen.is_identity:
                continue
            perm = list(range(data.ndim))
            for i in range(group.degree):
                perm[axes[i]] = axes[gen._array_form[i]]
            transposed = data.transpose(perm)
            if not np.allclose(data, transposed, atol=1e-6, rtol=1e-5):
                max_dev = float(np.max(np.abs(data - transposed)))
                raise SymmetryError(axes=tuple(axes), max_deviation=max_dev)


def is_symmetric(
    data: np.ndarray,
    symmetric_axes: tuple[int, ...] | list[tuple[int, ...]],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Check whether *data* is symmetric along the given axes.

    Parameters
    ----------
    data : numpy.ndarray
        The tensor data.
    symmetric_axes : tuple of int or list of tuple of int
        A single symmetry group ``(0, 1)`` or a list ``[(0, 1), (2, 3)]``.
    atol, rtol : float
        Tolerances passed to :func:`numpy.allclose`.

    Returns
    -------
    bool
    """
    if (
        isinstance(symmetric_axes, tuple)
        and symmetric_axes
        and not isinstance(symmetric_axes[0], tuple)
    ):
        groups: list[tuple[int, ...]] = [symmetric_axes]
    else:
        groups = list(symmetric_axes)

    for group in groups:
        if len(group) < 2:
            continue
        sizes = [data.shape[d] for d in group]
        if len(set(sizes)) != 1:
            return False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                axes = list(range(data.ndim))
                axes[group[i]], axes[group[j]] = axes[group[j]], axes[group[i]]
                transposed = data.transpose(axes)
                if not np.allclose(data, transposed, atol=atol, rtol=rtol):
                    return False
    return True


# ---------------------------------------------------------------------------
# Symmetry-loss warning helper
# ---------------------------------------------------------------------------


def _warn_symmetry_loss(
    lost_dims: list[tuple[int, ...]],
    reason: str,
    stacklevel: int = 3,
) -> None:
    """Emit a :class:`SymmetryLossWarning` if warnings are enabled."""
    if not get_setting("symmetry_warnings"):
        return
    dim_str = ", ".join(str(g) for g in lost_dims)
    warnings.warn(
        f"Symmetry lost along dims {dim_str}: {reason}. "
        "Use as_symmetric() to re-tag if you know the result is symmetric. "
        "Suppress with we.configure(symmetry_warnings=False).",
        SymmetryLossWarning,
        stacklevel=stacklevel,
    )


# ---------------------------------------------------------------------------
# Symmetry propagation helpers
# ---------------------------------------------------------------------------


def propagate_symmetry_slice(
    groups: list[PermutationGroup],
    shape: tuple[int, ...],
    key,
) -> list[PermutationGroup] | None:
    """Compute new symmetry groups after ``__getitem__(key)``.

    Parameters
    ----------
    groups : list of PermutationGroup
        Each group has ``axes`` indicating which tensor dimensions it acts on.
    shape : tuple of int
        Original tensor shape.
    key : indexing key

    Returns *None* if no symmetry survives (caller should return plain ndarray).
    """
    ndim = len(shape)

    if not isinstance(key, tuple):
        key = (key,)

    for k in key:
        if isinstance(k, (np.ndarray, list)):
            return None

    # Expand Ellipsis.
    expanded: list = []
    ellipsis_seen = False
    for k in key:
        if k is Ellipsis:
            if ellipsis_seen:
                raise IndexError("only one Ellipsis allowed")
            ellipsis_seen = True
            n_newaxis_in_key = sum(1 for kk in key if kk is None)
            n_explicit = len(key) - 1 - n_newaxis_in_key
            n_fill = ndim - n_explicit
            expanded.extend([slice(None)] * n_fill)
        else:
            expanded.append(k)
    if not ellipsis_seen:
        n_newaxis = sum(1 for k in expanded if k is None)
        while len(expanded) - n_newaxis < ndim:
            expanded.append(slice(None))
    key_expanded = expanded

    # Classify each original dim.
    old_dim_idx = 0
    dim_actions: dict[int, str | tuple] = {}

    for k in key_expanded:
        if k is None:
            continue
        if old_dim_idx >= ndim:
            break
        if isinstance(k, (int, np.integer)):
            dim_actions[old_dim_idx] = "removed"
            old_dim_idx += 1
        elif isinstance(k, slice):
            start, stop, step = k.indices(shape[old_dim_idx])
            new_size = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
            if new_size == shape[old_dim_idx]:
                dim_actions[old_dim_idx] = "untouched"
            else:
                dim_actions[old_dim_idx] = ("resized", new_size)
            old_dim_idx += 1
        else:
            return None

    while old_dim_idx < ndim:
        dim_actions[old_dim_idx] = "untouched"
        old_dim_idx += 1

    # Build old→new dim mapping.
    removed_dims = {d for d, a in dim_actions.items() if a == "removed"}
    old_to_new: dict[int, int | None] = {}
    newaxis_positions: list[int] = []
    orig_idx = 0
    for k in key_expanded:
        if k is None:
            newaxis_positions.append(orig_idx)
        else:
            orig_idx += 1

    new_idx = 0
    for d in range(ndim):
        while newaxis_positions and newaxis_positions[0] <= d:
            newaxis_positions.pop(0)
            new_idx += 1
        if d in removed_dims:
            old_to_new[d] = None
        else:
            old_to_new[d] = new_idx
            new_idx += 1

    # Process each group.
    new_groups: list[PermutationGroup] = []
    for group in groups:
        axes = group.axes
        if axes is None:
            continue

        # Map tensor axes to group-local indices.
        local_removed: set[int] = set()
        local_kept: list[int] = []
        for local_idx, tensor_dim in enumerate(axes):
            action = dim_actions.get(tensor_dim, "untouched")
            if action == "removed":
                local_removed.add(local_idx)
            else:
                local_kept.append(local_idx)

        if not local_kept:
            continue

        # Pointwise stabilizer: each removed axis must map to itself.
        # (Setwise would only be valid when all removed axes share the
        # same slice value, which we can't determine in general.)
        stab = group.pointwise_stabilizer(local_removed)

        # Among surviving axes, check for size mismatches.
        size_map: dict[int, int] = {}
        for local_idx in local_kept:
            tensor_dim = axes[local_idx]
            action = dim_actions.get(tensor_dim, "untouched")
            if isinstance(action, tuple) and action[0] == "resized":
                size_map[local_idx] = action[1]
            else:
                size_map[local_idx] = shape[tensor_dim]

        sizes = set(size_map.values())
        if len(sizes) > 1:
            for sz in sizes:
                same_size = {li for li, s in size_map.items() if s == sz}
                complement = set(local_kept) - same_size
                if complement:
                    stab = stab.setwise_stabilizer(same_size)

        # Restrict to kept local indices.
        kept_tuple = tuple(local_kept)
        if len(kept_tuple) < 2:
            continue

        restricted = stab.restrict(kept_tuple)

        if restricted.order() <= 1:
            continue

        # Remap axes to new tensor dim numbering.
        new_axes = tuple(old_to_new[axes[k]] for k in kept_tuple)
        if any(a is None for a in new_axes):
            continue

        final = PermutationGroup(*restricted.generators, axes=new_axes)
        new_groups.append(final)

    return new_groups if new_groups else None


def propagate_symmetry_reduce(
    groups: list[PermutationGroup],
    ndim: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool = False,
) -> list[PermutationGroup] | None:
    """Compute new symmetry groups after a reduction.

    Parameters
    ----------
    groups : list of PermutationGroup
        Each group has ``axes`` indicating which tensor dimensions it acts on.
    ndim : int
        Original tensor rank.
    axis : int, tuple of int, or None
        Axes being reduced.
    keepdims : bool
        Whether reduced dims are kept at size 1.

    Returns *None* if no symmetry survives.
    """
    if axis is None:
        return None

    # Normalize axis.
    if isinstance(axis, int):
        axes_set = {axis % ndim}
    else:
        axes_set = {a % ndim for a in axis}

    # Build old→new mapping (only needed when not keepdims).
    old_to_new: dict[int, int] = {}
    if not keepdims:
        new_idx = 0
        for d in range(ndim):
            if d not in axes_set:
                old_to_new[d] = new_idx
                new_idx += 1
    else:
        old_to_new = {d: d for d in range(ndim)}

    new_groups: list[PermutationGroup] = []
    for group in groups:
        grp_axes = group.axes
        if grp_axes is None:
            continue

        # Map tensor axes to group-local indices.
        local_reduced: set[int] = set()
        local_kept: list[int] = []
        for local_idx, tensor_dim in enumerate(grp_axes):
            if tensor_dim in axes_set:
                local_reduced.add(local_idx)
            else:
                local_kept.append(local_idx)

        if not local_reduced:
            # Group is entirely outside the reduced axes — just remap.
            new_axes = tuple(old_to_new[grp_axes[i]] for i in range(group.degree))
            new_groups.append(PermutationGroup(*group.generators, axes=new_axes))
            continue

        if not local_kept:
            # All axes in this group are reduced — group vanishes from output.
            continue

        # Setwise stabilizer: elements mapping reduced axes among themselves.
        stab = group.setwise_stabilizer(local_reduced)

        # Restrict to kept local indices.
        kept_tuple = tuple(local_kept)
        if len(kept_tuple) < 2:
            continue

        restricted = stab.restrict(kept_tuple)
        if restricted.order() <= 1:
            continue

        # Remap to new tensor axis numbering.
        new_axes = tuple(old_to_new[grp_axes[k]] for k in kept_tuple)
        final = PermutationGroup(*restricted.generators, axes=new_axes)
        new_groups.append(final)

    return new_groups if new_groups else None


def intersect_symmetry(
    groups_a: list[PermutationGroup] | None,
    groups_b: list[PermutationGroup] | None,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> list[PermutationGroup] | None:
    """Intersect symmetry groups for binary ops, accounting for broadcasting.

    For groups acting on the same output axes, computes the element-set
    intersection.  Broadcast-stretched dimensions (size 1 → larger) are
    removed from groups before intersecting.

    Parameters
    ----------
    groups_a, groups_b : list of PermutationGroup or None
        Symmetry groups for each operand.
    shape_a, shape_b : tuple of int
        Input shapes (before broadcasting).
    output_shape : tuple of int
        Broadcast output shape.

    Returns
    -------
    list of PermutationGroup or None
        Groups present in both operands, or *None* if no shared symmetry.
    """
    if groups_a is None or groups_b is None:
        return None

    ndim_out = len(output_shape)
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)

    offset_a = ndim_out - ndim_a
    offset_b = ndim_out - ndim_b

    def _remap_axes(groups: list[PermutationGroup], offset: int, input_shape: tuple[int, ...]):
        """Remap group axes to output dims and remove broadcast-stretched dims."""
        result = []
        for group in groups:
            if group.axes is None:
                continue
            new_axes = []
            local_kept = []
            for local_idx, tensor_dim in enumerate(group.axes):
                out_dim = tensor_dim + offset
                # Check if broadcast-stretched (size 1 → larger).
                if 0 <= tensor_dim < len(input_shape):
                    if input_shape[tensor_dim] == 1 and output_shape[out_dim] > 1:
                        continue  # stretched — remove from group
                new_axes.append(out_dim)
                local_kept.append(local_idx)
            if len(local_kept) >= 2:
                restricted = group.restrict(tuple(local_kept))
                if restricted.order() > 1:
                    result.append(PermutationGroup(
                        *restricted.generators, axes=tuple(new_axes)
                    ))
        return result

    aligned_a = _remap_axes(groups_a, offset_a, shape_a)
    aligned_b = _remap_axes(groups_b, offset_b, shape_b)

    # Intersect: for groups acting on the same output axes, compute element intersection.
    b_by_axes: dict[tuple[int, ...], PermutationGroup] = {}
    for g in aligned_b:
        if g.axes is not None:
            b_by_axes[g.axes] = g

    intersection: list[PermutationGroup] = []
    for ga in aligned_a:
        if ga.axes is None:
            continue
        gb = b_by_axes.get(ga.axes)
        if gb is None:
            continue
        # Element-set intersection.
        common_elements = set(ga.elements()) & set(gb.elements())
        if len(common_elements) <= 1:
            continue
        intersection.append(PermutationGroup(*common_elements, axes=ga.axes))

    return intersection if intersection else None


# ---------------------------------------------------------------------------
# SymmetricTensor  (np.ndarray subclass)
# ---------------------------------------------------------------------------


class SymmetricTensor(np.ndarray):
    """An ndarray that carries symmetry metadata.

    Do not instantiate directly; use :func:`as_symmetric`.
    """

    def __new__(
        cls,
        input_array: np.ndarray,
        symmetric_axes: list[tuple[int, ...]],
        *,
        perm_groups: list | None = None,
    ) -> SymmetricTensor:
        obj = np.asarray(input_array).view(cls)
        obj._symmetric_axes = [tuple(sorted(g)) for g in symmetric_axes]
        if perm_groups is not None:
            obj._symmetry_groups = list(perm_groups)
        else:
            # Auto-build S_k groups from symmetric_axes
            obj._symmetry_groups = [
                PermutationGroup.symmetric(len(g), axes=g)
                for g in obj._symmetric_axes
                if len(g) >= 2
            ]
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        axes = getattr(obj, "_symmetric_axes", None)
        groups = getattr(obj, "_symmetry_groups", None)
        # Validate that symmetric axes are still valid for the new shape.
        # Views, reshapes, and other numpy internals can produce arrays with
        # different shapes that inherit metadata via __array_finalize__.
        if axes and self.shape != getattr(obj, "shape", None):
            valid = []
            valid_groups = []
            for i, group in enumerate(axes):
                if all(
                    d < self.ndim and self.shape[d] == self.shape[group[0]]
                    for d in group
                ):
                    valid.append(group)
                    if groups and i < len(groups):
                        valid_groups.append(groups[i])
            axes = valid or None
            groups = valid_groups or None
        self._symmetric_axes = axes
        self._symmetry_groups = groups

    # -- public API --

    @property
    def symmetric_axes(self) -> list[tuple[int, ...]]:
        """Symmetry groups carried by this tensor."""
        return list(self._symmetric_axes) if self._symmetric_axes else []

    @property
    def symmetry_info(self) -> SymmetryInfo:
        """Return a :class:`SymmetryInfo` for this tensor."""
        return SymmetryInfo(
            symmetric_axes=self.symmetric_axes,
            shape=self.shape,
            groups=list(self._symmetry_groups) if self._symmetry_groups else None,
        )

    def is_symmetric(
        self,
        symmetric_axes: tuple[int, ...] | list[tuple[int, ...]] | None = None,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> bool:
        """Check whether the data satisfies the given (or carried) symmetry.

        Parameters
        ----------
        symmetric_axes : tuple or list of tuples, optional
            Axes to check.  If *None*, checks the axes already carried
            by this ``SymmetricTensor``.
        atol, rtol : float
            Tolerances passed to :func:`numpy.allclose`.
        """
        axes = symmetric_axes if symmetric_axes is not None else self._symmetric_axes
        return is_symmetric(np.asarray(self), axes, atol=atol, rtol=rtol)

    # -- slicing with symmetry propagation --

    def __getitem__(self, key):  # type: ignore[override]
        """Index with symmetry propagation.

        Computes the pointwise-stabilizer subgroup for axes removed by
        integer indexing, then restricts surviving groups to the output
        axes.  Returns a plain ``ndarray`` when no symmetry survives.
        Emits :class:`~whest.errors.SymmetryLossWarning` on partial or
        total symmetry loss.
        """
        result = super().__getitem__(key)
        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return result if not isinstance(result, np.ndarray) else np.asarray(result)

        if not self._symmetry_groups:
            return np.asarray(result)

        new_groups = propagate_symmetry_slice(self._symmetry_groups, self.shape, key)
        if new_groups is not None:
            out = np.asarray(result).view(SymmetricTensor)
            out._symmetry_groups = new_groups
            out._symmetric_axes = [g.axes for g in new_groups if g.axes is not None]
            # Warn if symmetry was partially lost.
            if len(new_groups) < len(self._symmetry_groups):
                lost_axes = [
                    g.axes for g in self._symmetry_groups
                    if g.axes is not None
                ]
                if lost_axes:
                    _warn_symmetry_loss(
                        lost_axes, "slicing changed dim sizes or removed dims"
                    )
            return out
        else:
            if self._symmetry_groups:
                _warn_symmetry_loss(
                    [g.axes for g in self._symmetry_groups if g.axes is not None],
                    "slicing removed all symmetric dim groups",
                )
            return np.asarray(result)

    # -- copy preserves metadata --

    def copy(self, order: str = "C") -> SymmetricTensor:  # type: ignore[override]
        result = super().copy(order=order)
        # super().copy() goes through __array_finalize__ which copies _symmetric_axes
        # but result may have been cast to plain ndarray, so re-view:
        out = result.view(SymmetricTensor)
        out._symmetric_axes = list(self._symmetric_axes)
        out._symmetry_groups = (
            list(self._symmetry_groups) if self._symmetry_groups else []
        )
        return out

    # -- pickling --

    def __reduce__(self):
        # Use np.ndarray's pickle + our metadata
        pickled_state = super().__reduce__()
        # pickled_state is (reconstruct, args, state)
        new_state = pickled_state[2] + (self._symmetric_axes, self._symmetry_groups)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Detect format by checking if the second-to-last element looks like
        # symmetric_axes (a list of tuples of ints) vs a list of PermutationGroup.
        # New format: state[-1] is _symmetry_groups (list of PermutationGroup or [])
        # Old format: state[-1] is _symmetric_axes (list of tuples)
        last = state[-1]
        # Check if last element looks like a list of PermutationGroup objects
        # (new format) vs list of tuples (old format)
        if isinstance(last, list) and (
            len(last) == 0 or isinstance(last[0], PermutationGroup)
        ):
            # Could be new format (empty list or list of PermutationGroup)
            # Check second-to-last to confirm it's symmetric_axes
            second_last = state[-2]
            if isinstance(second_last, list) and (
                len(second_last) == 0 or isinstance(second_last[0], tuple)
            ):
                # New format
                self._symmetry_groups = last
                self._symmetric_axes = second_last
                super().__setstate__(state[:-2])
                return
        # Old format: last element is _symmetric_axes
        self._symmetric_axes = last
        self._symmetry_groups = [
            PermutationGroup.symmetric(len(g), axes=g)
            for g in self._symmetric_axes
            if len(g) >= 2
        ]
        super().__setstate__(state[:-1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def as_symmetric(
    data: np.ndarray,
    symmetric_axes: tuple[int, ...] | list[tuple[int, ...]] | None = None,
    *,
    symmetry: PermutationGroup | list | None = None,
) -> SymmetricTensor:
    """Wrap *data* as a :class:`SymmetricTensor` after validating symmetry.

    Parameters
    ----------
    data : numpy.ndarray
        The tensor data.
    symmetric_axes : tuple of int or list of tuple of int, optional
        A single symmetry group ``(0, 1)`` or a list ``[(0, 1), (2, 3)]``.
        Mutually exclusive with *symmetry*.
    symmetry : PermutationGroup or list of PermutationGroup, optional
        One or more :class:`PermutationGroup` objects (must have axes set).
        Mutually exclusive with *symmetric_axes*.

    Returns
    -------
    SymmetricTensor

    Raises
    ------
    ValueError
        If both *symmetric_axes* and *symmetry* are provided.
    SymmetryError
        If the data does not satisfy the claimed symmetry.
    """
    if symmetric_axes is not None and symmetry is not None:
        raise ValueError(
            "symmetric_axes and symmetry are mutually exclusive; provide only one"
        )

    if symmetry is not None:
        # PermutationGroup path
        if isinstance(symmetry, PermutationGroup):
            perm_groups = [symmetry]
        else:
            perm_groups = list(symmetry)

        validate_symmetry_groups(data, perm_groups)

        # Build symmetric_axes from groups for backward compat
        axes_list = []
        for g in perm_groups:
            if g.axes is not None:
                axes_list.append(g.axes)

        return SymmetricTensor(data, axes_list, perm_groups=perm_groups)

    # Legacy symmetric_axes path
    if symmetric_axes is None:
        raise ValueError("Either symmetric_axes or symmetry must be provided")

    if isinstance(symmetric_axes, tuple):
        groups: list[tuple[int, ...]] = [symmetric_axes]
    else:
        groups = list(symmetric_axes)

    validate_symmetry(data, groups)
    return SymmetricTensor(data, groups)
