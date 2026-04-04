"""Symmetric tensor support: SymmetryInfo, SymmetricTensor, as_symmetric."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import numpy as np

from mechestim.errors import SymmetryError


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

    # -- slicing returns plain ndarray --

    def __getitem__(self, key):  # type: ignore[override]
        result = super().__getitem__(key)
        if isinstance(result, np.ndarray):
            return np.asarray(result)
        return result

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
