"""Symmetric tensor support: SymmetryInfo, SymmetricTensor, as_symmetric."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, factorial
from typing import Sequence

import numpy as np

from mechestim.errors import SymmetryError


@dataclass(frozen=True)
class SymmetryInfo:
    """Metadata about tensor symmetry groups.

    Parameters
    ----------
    symmetric_dims : list of tuple of int
        Groups of dimension indices that are symmetric under permutation.
    shape : tuple of int
        Full tensor shape.
    """

    symmetric_dims: list[tuple[int, ...]]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        # Normalize each group to a sorted tuple.
        normalized = [tuple(sorted(g)) for g in self.symmetric_dims]
        # frozen=True means we must use object.__setattr__
        object.__setattr__(self, "symmetric_dims", normalized)

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
        for group in self.symmetric_dims:
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
        for group in self.symmetric_dims:
            result *= factorial(len(group))
        return result
