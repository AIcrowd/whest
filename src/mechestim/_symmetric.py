"""Symmetric tensor support: SymmetryInfo, SymmetricTensor, as_symmetric."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import comb, factorial

import numpy as np

from mechestim._config import get_setting
from mechestim.errors import SymmetryError, SymmetryLossWarning


@dataclass(frozen=True)
class SymmetryInfo:
    """Metadata about tensor symmetry groups.

    Parameters
    ----------
    symmetric_axes : list of tuple of int
        Groups of dimension indices that are symmetric under permutation.
    shape : tuple of int
        Full tensor shape.
    """

    symmetric_axes: list[tuple[int, ...]]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        # Normalize each group to a sorted tuple.
        normalized = [tuple(sorted(g)) for g in self.symmetric_axes]
        # frozen=True means we must use object.__setattr__
        object.__setattr__(self, "symmetric_axes", normalized)

    @property
    def unique_elements(self) -> int:
        """Number of unique elements accounting for symmetry.

        For each symmetric group of *k* dims each of size *n*,
        the count is C(n + k - 1, k).  Free (non-symmetric) dims
        contribute their full size.  The total is the product.
        """
        # Collect all dims that belong to a symmetric group.
        symmetric_indices: set[int] = set()
        result = 1
        for group in self.symmetric_axes:
            symmetric_indices.update(group)
            k = len(group)
            n = self.shape[group[0]]  # all dims in a group must be same size
            result *= comb(n + k - 1, k)
        # Multiply by free dims.
        for i, s in enumerate(self.shape):
            if i not in symmetric_indices:
                result *= s
        return result

    @property
    def symmetry_factor(self) -> int:
        """Product of factorial(len(group)) for each group."""
        result = 1
        for group in self.symmetric_axes:
            result *= factorial(len(group))
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
        "Suppress with me.configure(symmetry_warnings=False).",
        SymmetryLossWarning,
        stacklevel=stacklevel,
    )


# ---------------------------------------------------------------------------
# Symmetry propagation helpers
# ---------------------------------------------------------------------------


def propagate_symmetry_slice(
    symmetric_axes: list[tuple[int, ...]],
    shape: tuple[int, ...],
    key,
) -> list[tuple[int, ...]] | None:
    """Compute new symmetry groups after ``__getitem__(key)``.

    Returns *None* if no symmetry survives (caller should return plain ndarray).
    """
    ndim = len(shape)

    # Normalize key to a tuple.
    if not isinstance(key, tuple):
        key = (key,)

    # Advanced indexing (ndarray / list) → bail out.
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
            n_explicit = len(key) - 1 - n_newaxis_in_key  # -1 for Ellipsis
            n_fill = ndim - n_explicit
            expanded.extend([slice(None)] * n_fill)
        else:
            expanded.append(k)
    if not ellipsis_seen:
        # Pad with slice(None) for unspecified trailing dims.
        n_newaxis = sum(1 for k in expanded if k is None)
        while len(expanded) - n_newaxis < ndim:
            expanded.append(slice(None))
    key_expanded = expanded

    # Walk through the key, tracking each original dim.
    # old_dim_idx tracks which original dim we're consuming.
    old_dim_idx = 0
    # For each original dim: "removed", ("kept", new_size), or "untouched"
    dim_actions: dict[int, str | tuple] = {}

    for k in key_expanded:
        if k is None:
            # np.newaxis — adds a dim, doesn't consume an original dim.
            continue
        if old_dim_idx >= ndim:
            break
        if isinstance(k, (int, np.integer)):
            dim_actions[old_dim_idx] = "removed"
            old_dim_idx += 1
        elif isinstance(k, slice):
            # Compute resulting size for this dim.
            start, stop, step = k.indices(shape[old_dim_idx])
            new_size = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
            if new_size == shape[old_dim_idx]:
                dim_actions[old_dim_idx] = "untouched"
            else:
                dim_actions[old_dim_idx] = ("resized", new_size)
            old_dim_idx += 1
        else:
            # Unknown indexer — bail.
            return None

    # Fill remaining dims as untouched.
    while old_dim_idx < ndim:
        dim_actions[old_dim_idx] = "untouched"
        old_dim_idx += 1

    # Build old→new dim mapping (removed dims get None).
    removed_dims = {d for d, a in dim_actions.items() if a == "removed"}
    old_to_new: dict[int, int | None] = {}
    new_idx = 0
    # Account for newaxis insertions: they shift new indices.
    # Simple approach: count newaxis positions before each original dim.
    newaxis_positions: list[int] = []
    orig_idx = 0
    for i, k in enumerate(key_expanded):
        if k is None:
            newaxis_positions.append(orig_idx)
        else:
            orig_idx += 1

    new_idx = 0
    for d in range(ndim):
        # Insert newaxis dims that come before this original dim.
        while newaxis_positions and newaxis_positions[0] <= d:
            newaxis_positions.pop(0)
            new_idx += 1
        if d in removed_dims:
            old_to_new[d] = None
        else:
            old_to_new[d] = new_idx
            new_idx += 1

    # Remap symmetry groups.
    new_groups: list[tuple[int, ...]] = []
    for group in symmetric_axes:
        # Collect surviving dims with their effective sizes.
        surviving_with_size: list[tuple[int, int]] = []  # (new_dim, size)
        for d in group:
            action = dim_actions.get(d, "untouched")
            if action == "removed":
                continue
            new_d = old_to_new.get(d)
            if new_d is None:
                continue
            if isinstance(action, tuple) and action[0] == "resized":
                surviving_with_size.append((new_d, action[1]))
            else:
                surviving_with_size.append((new_d, shape[d]))

        if not surviving_with_size:
            continue

        # Group by size — only dims with the same size can stay in a group.
        # Use the most common size (the untouched original size).
        from collections import Counter

        size_counts = Counter(s for _, s in surviving_with_size)
        # Pick the original (untouched) size as the canonical one.
        # That's the size that appears for untouched dims.
        original_size = shape[group[0]]
        if original_size in size_counts:
            canonical_size = original_size
        else:
            # All dims were resized; pick the most common size.
            canonical_size = size_counts.most_common(1)[0][0]

        same_size_dims = [d for d, s in surviving_with_size if s == canonical_size]
        if len(same_size_dims) >= 2:
            new_groups.append(tuple(sorted(same_size_dims)))

    return new_groups if new_groups else None


def propagate_symmetry_reduce(
    symmetric_axes: list[tuple[int, ...]],
    ndim: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool = False,
) -> list[tuple[int, ...]] | None:
    """Compute new symmetry groups after a reduction.

    Returns *None* if no symmetry survives.
    """
    if axis is None:
        return None

    # Normalize axis.
    if isinstance(axis, int):
        axes = (axis % ndim,)
    else:
        axes = tuple(a % ndim for a in axis)
    axes_set = set(axes)

    if keepdims:
        # Reduced dims stay but have size 1 → pull from groups.
        new_groups: list[tuple[int, ...]] = []
        for group in symmetric_axes:
            surviving = tuple(d for d in group if d not in axes_set)
            if len(surviving) >= 2:
                new_groups.append(surviving)
        return new_groups if new_groups else None
    else:
        # Removed dims — renumber.
        old_to_new: dict[int, int] = {}
        new_idx = 0
        for d in range(ndim):
            if d not in axes_set:
                old_to_new[d] = new_idx
                new_idx += 1

        new_groups = []
        for group in symmetric_axes:
            surviving = []
            for d in group:
                if d in old_to_new:
                    surviving.append(old_to_new[d])
            if len(surviving) >= 2:
                new_groups.append(tuple(sorted(surviving)))
        return new_groups if new_groups else None


def intersect_symmetry(
    dims_a: list[tuple[int, ...]] | None,
    dims_b: list[tuple[int, ...]] | None,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> list[tuple[int, ...]] | None:
    """Intersect symmetry groups for binary ops, accounting for broadcasting."""
    if dims_a is None or dims_b is None:
        return None

    ndim_out = len(output_shape)
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)

    # Align dims to the right (broadcasting alignment).
    offset_a = ndim_out - ndim_a
    offset_b = ndim_out - ndim_b

    # Remap to output dim indices.
    def _remap(dims: list[tuple[int, ...]], offset: int) -> list[tuple[int, ...]]:
        return [tuple(d + offset for d in g) for g in dims]

    aligned_a = _remap(dims_a, offset_a)
    aligned_b = _remap(dims_b, offset_b)

    # Identify broadcast-stretched dims for each input.
    def _remove_stretched(
        groups: list[tuple[int, ...]],
        input_shape: tuple[int, ...],
        offset: int,
    ) -> list[tuple[int, ...]]:
        result = []
        for group in groups:
            surviving = []
            for d in group:
                orig_d = d - offset
                if 0 <= orig_d < len(input_shape):
                    if input_shape[orig_d] == 1 and output_shape[d] > 1:
                        continue  # stretched
                surviving.append(d)
            if len(surviving) >= 2:
                result.append(tuple(sorted(surviving)))
        return result

    cleaned_a = _remove_stretched(aligned_a, shape_a, offset_a)
    cleaned_b = _remove_stretched(aligned_b, shape_b, offset_b)

    # Intersect: keep groups present in both.
    set_b = set(tuple(g) for g in cleaned_b)
    intersection = [g for g in cleaned_a if tuple(g) in set_b]

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
    ) -> SymmetricTensor:
        obj = np.asarray(input_array).view(cls)
        obj._symmetric_axes = [tuple(sorted(g)) for g in symmetric_axes]
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        self._symmetric_axes = getattr(obj, "_symmetric_axes", None)

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
        result = super().__getitem__(key)
        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return result if not isinstance(result, np.ndarray) else np.asarray(result)

        new_groups = propagate_symmetry_slice(self._symmetric_axes, self.shape, key)
        if new_groups is not None:
            out = np.asarray(result).view(SymmetricTensor)
            out._symmetric_axes = new_groups
            # Warn if symmetry was partially lost.
            old_set = set(self._symmetric_axes)
            new_set = set(new_groups)
            if new_set != old_set:
                lost = [g for g in self._symmetric_axes if g not in new_set]
                if lost:
                    _warn_symmetry_loss(
                        lost, "slicing changed dim sizes or removed dims"
                    )
            return out
        else:
            # All symmetry lost.
            if self._symmetric_axes:
                _warn_symmetry_loss(
                    self._symmetric_axes, "slicing removed all symmetric dim groups"
                )
            return np.asarray(result)

    # -- copy preserves metadata --

    def copy(self, order: str = "C") -> SymmetricTensor:  # type: ignore[override]
        result = super().copy(order=order)
        # super().copy() goes through __array_finalize__ which copies _symmetric_axes
        # but result may have been cast to plain ndarray, so re-view:
        out = result.view(SymmetricTensor)
        out._symmetric_axes = list(self._symmetric_axes)
        return out

    # -- pickling --

    def __reduce__(self):
        # Use np.ndarray's pickle + our metadata
        pickled_state = super().__reduce__()
        # pickled_state is (reconstruct, args, state)
        new_state = pickled_state[2] + (self._symmetric_axes,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Last element is our _symmetric_axes
        self._symmetric_axes = state[-1]
        super().__setstate__(state[:-1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def as_symmetric(
    data: np.ndarray,
    symmetric_axes: tuple[int, ...] | list[tuple[int, ...]],
) -> SymmetricTensor:
    """Wrap *data* as a :class:`SymmetricTensor` after validating symmetry.

    Parameters
    ----------
    data : numpy.ndarray
        The tensor data.
    symmetric_axes : tuple of int or list of tuple of int
        A single symmetry group ``(0, 1)`` or a list ``[(0, 1), (2, 3)]``.

    Returns
    -------
    SymmetricTensor

    Raises
    ------
    SymmetryError
        If the data does not satisfy the claimed symmetry.
    """
    if isinstance(symmetric_axes, tuple):
        groups: list[tuple[int, ...]] = [symmetric_axes]
    else:
        groups = list(symmetric_axes)

    validate_symmetry(data, groups)
    return SymmetricTensor(data, groups)
