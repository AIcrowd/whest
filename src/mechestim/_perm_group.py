"""Permutation groups for exact symmetry representation.

Provides ``Permutation`` and ``PermutationGroup`` with the same internal
representation (integer array form) as sympy.combinatorics, enabling
zero-friction interop via ``as_sympy()`` / ``from_sympy()``.

Core algorithms:
- Dimino's algorithm for group enumeration from generators
  (Butler & McKay, Comm. in Algebra, 1983)
- Burnside's lemma for orbit counting (Burnside, 1897)

The API naming and array-form convention follow sympy's combinatorics
module for interoperability. No sympy code is used; this is an
independent implementation of standard algorithms.
"""

from __future__ import annotations

import math
from functools import reduce


class Permutation:
    """A permutation on {0, 1, ..., n-1} in array form.

    ``_array_form[i]`` is the image of ``i`` under the permutation.
    Same convention as ``sympy.combinatorics.Permutation``.
    """

    __slots__ = ("_array_form",)

    def __init__(self, array_form: list[int] | tuple[int, ...]) -> None:
        self._array_form = tuple(array_form)

    @property
    def size(self) -> int:
        return len(self._array_form)

    @property
    def array_form(self) -> list[int]:
        """Copy of the array form (matches sympy API)."""
        return list(self._array_form)

    @property
    def is_identity(self) -> bool:
        return all(self._array_form[i] == i for i in range(len(self._array_form)))

    @classmethod
    def identity(cls, size: int) -> Permutation:
        return cls(range(size))

    @classmethod
    def from_cycle(cls, size: int, cycle: list[int]) -> Permutation:
        """Construct from a single cycle on {0, ..., size-1}."""
        arr = list(range(size))
        for i in range(len(cycle)):
            arr[cycle[i]] = cycle[(i + 1) % len(cycle)]
        return cls(arr)

    def __mul__(self, other: Permutation) -> Permutation:
        """Compose: ``(self * other)[i] = self[other[i]]``."""
        return Permutation(tuple(self._array_form[other._array_form[i]] for i in range(self.size)))

    def __invert__(self) -> Permutation:
        inv = [0] * self.size
        for i, j in enumerate(self._array_form):
            inv[j] = i
        return Permutation(inv)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return NotImplemented
        return self._array_form == other._array_form

    def __hash__(self) -> int:
        return hash(self._array_form)

    def __repr__(self) -> str:
        return f"Permutation({list(self._array_form)})"

    @property
    def cyclic_form(self) -> list[tuple[int, ...]]:
        """Disjoint cycles, excluding fixed points. Smallest element first in each cycle."""
        visited: set[int] = set()
        cycles: list[tuple[int, ...]] = []
        for i in range(self.size):
            if i in visited or self._array_form[i] == i:
                visited.add(i)
                continue
            cycle: list[int] = []
            j = i
            while j not in visited:
                cycle.append(j)
                visited.add(j)
                j = self._array_form[j]
            cycles.append(tuple(cycle))
        return cycles

    @property
    def full_cyclic_form(self) -> list[tuple[int, ...]]:
        """Disjoint cycles, including fixed points as 1-cycles."""
        visited: set[int] = set()
        cycles: list[tuple[int, ...]] = []
        for i in range(self.size):
            if i in visited:
                continue
            cycle: list[int] = []
            j = i
            while j not in visited:
                cycle.append(j)
                visited.add(j)
                j = self._array_form[j]
            cycles.append(tuple(cycle))
        return cycles

    @property
    def cycle_structure(self) -> dict[int, int]:
        """Map cycle length -> count (excludes fixed points)."""
        result: dict[int, int] = {}
        for cycle in self.cyclic_form:
            length = len(cycle)
            result[length] = result.get(length, 0) + 1
        return result

    @property
    def order(self) -> int:
        """Order of this element: lcm of cycle lengths (1 for identity)."""
        lengths = [len(c) for c in self.full_cyclic_form]
        if not lengths:
            return 1
        return reduce(lambda a, b: a * b // math.gcd(a, b), lengths)

    # --- Sympy bridge ---

    def as_sympy(self):
        """Convert to ``sympy.combinatorics.Permutation``. Requires sympy."""
        try:
            from sympy.combinatorics import Permutation as SPermutation
        except ImportError:
            raise ImportError(
                "sympy is required for as_sympy(). Install with: pip install sympy"
            ) from None
        return SPermutation(self.array_form)

    @classmethod
    def from_sympy(cls, sp) -> Permutation:
        """Construct from a ``sympy.combinatorics.Permutation``."""
        return cls(sp.array_form)
