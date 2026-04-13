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


class Cycle:
    """Composable cycle builder, matching sympy's Cycle API.

    Build permutations by composing disjoint or overlapping cycles::

        Cycle(0, 2)(1, 3)     # → (0 2)(1 3)
        Permutation(Cycle(0, 2)(1, 3))  # → Permutation([2, 3, 0, 1])
    """

    __slots__ = ("_mapping",)

    def __init__(self, *cycle: int) -> None:
        self._mapping: dict[int, int] = {}
        if cycle:
            for i in range(len(cycle)):
                self._mapping[cycle[i]] = cycle[(i + 1) % len(cycle)]

    def __call__(self, *cycle: int) -> Cycle:
        """Compose another cycle, returning a new Cycle."""
        new = Cycle()
        new._mapping = dict(self._mapping)
        if cycle:
            new_cycle_map: dict[int, int] = {}
            for i in range(len(cycle)):
                new_cycle_map[cycle[i]] = cycle[(i + 1) % len(cycle)]
            combined: dict[int, int] = {}
            all_points = set(new._mapping) | set(new_cycle_map)
            for x in all_points:
                y = new._mapping.get(x, x)
                z = new_cycle_map.get(y, y)
                if z != x:
                    combined[x] = z
            new._mapping = combined
        return new

    def list(self, size: int | None = None) -> list[int]:
        """Return array form. Size inferred from max element + 1 if not given."""
        if size is None:
            size = max(self._mapping.keys(), default=-1) + 1
            size = max(size, max(self._mapping.values(), default=-1) + 1)
        arr = list(range(size))
        for k, v in self._mapping.items():
            if k < size:
                arr[k] = v
        return arr


class Permutation:
    """A permutation on {0, 1, ..., n-1} in array form.

    ``_array_form[i]`` is the image of ``i`` under the permutation.
    Same convention as ``sympy.combinatorics.Permutation``.
    """

    __slots__ = ("_array_form",)

    def __init__(
        self,
        array_form: list[int] | tuple[int, ...] | Cycle,
        size: int | None = None,
    ) -> None:
        if isinstance(array_form, Cycle):
            self._array_form = tuple(array_form.list(size))
        elif array_form and isinstance(array_form[0], (list, tuple)):
            # Cycle notation: list of lists
            c = Cycle()
            for cycle in array_form:
                c = c(*cycle)
            self._array_form = tuple(c.list(size))
        else:
            arr = list(array_form)
            if size is not None and size > len(arr):
                arr.extend(range(len(arr), size))
            self._array_form = tuple(arr)

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
        return Permutation(
            tuple(self._array_form[other._array_form[i]] for i in range(self.size))
        )

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

    def __call__(self, i: int) -> int:
        """Apply the permutation: ``perm(i)`` returns the image of ``i``."""
        return self._array_form[i]

    def support(self) -> set[int]:
        """Set of non-fixed points."""
        return {i for i in range(self.size) if self._array_form[i] != i}

    def parity(self) -> int:
        """Parity: 0 if even, 1 if odd."""
        return sum(len(c) - 1 for c in self.cyclic_form) % 2

    def signature(self) -> int:
        """Signature: +1 if even, -1 if odd."""
        return 1 if self.parity() == 0 else -1

    def transpositions(self) -> list[tuple[int, int]]:
        """Decompose into a product of transpositions.

        Each cycle (a, b, c, ...) becomes [(a, b), (a, c), ..., (a, ...)].
        Applying them left-to-right (each prepended) reconstructs the permutation.
        """
        result: list[tuple[int, int]] = []
        for cycle in self.cyclic_form:
            for i in range(1, len(cycle)):
                result.append((cycle[0], cycle[i]))
        return result

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


class PermutationGroup:
    """A permutation group on {0, ..., n-1} defined by generators.

    Same generator-based design as ``sympy.combinatorics.PermutationGroup``.
    For the small groups in einsum symmetry (typically < 100 elements), all
    elements are enumerated via Dimino's algorithm.
    """

    __slots__ = ("_generators", "_degree", "_axes", "_elements", "_order", "_labels")

    def __init__(
        self,
        *generators: Permutation,
        axes: tuple[int, ...] | None = None,
    ) -> None:
        if not generators:
            raise ValueError(
                "At least one generator required (use Permutation.identity(n) for the trivial group)"
            )
        degrees = {g.size for g in generators}
        if len(degrees) != 1:
            raise ValueError(f"All generators must have the same size, got {degrees}")
        self._generators = generators
        self._degree = generators[0].size
        self._axes = axes
        self._elements: list[Permutation] | None = None
        self._order: int | None = None
        self._labels: tuple[str, ...] | None = None

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def generators(self) -> list[Permutation]:
        return list(self._generators)

    @property
    def axes(self) -> tuple[int, ...] | None:
        return self._axes

    def elements(self) -> list[Permutation]:
        """All group elements via Dimino's algorithm. Cached."""
        if self._elements is not None:
            return self._elements
        self._elements = _dimino(self._generators)
        self._order = len(self._elements)
        return self._elements

    def order(self) -> int:
        """Number of elements in the group."""
        if self._order is not None:
            return self._order
        self._order = len(self.elements())
        return self._order

    def is_symmetric(self) -> bool:
        """True if this equals S_degree (the full symmetric group)."""
        return self.order() == math.factorial(self._degree)

    def orbits(self) -> list[frozenset[int]]:
        """Partition of {0, ..., degree-1} into orbits."""
        parent = list(range(self._degree))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for g in self._generators:
            for i in range(self._degree):
                if g._array_form[i] != i:
                    union(i, g._array_form[i])

        groups: dict[int, set[int]] = {}
        for i in range(self._degree):
            groups.setdefault(find(i), set()).add(i)
        return [frozenset(s) for s in groups.values()]

    def burnside_unique_count(self, size_dict: dict[int, int]) -> int:
        """Count unique tensor elements via Burnside's lemma.

        Parameters
        ----------
        size_dict : dict
            Maps position {0, ..., degree-1} to dimension size.
            Positions in the same orbit must have equal sizes.

        Returns
        -------
        int
            (1/|G|) * sum over g in G of product over each cycle c of g
            of size_dict[any element of c].
        """
        for orbit in self.orbits():
            sizes = {size_dict[i] for i in orbit}
            if len(sizes) != 1:
                raise ValueError(
                    f"Positions {orbit} are in the same orbit but have different "
                    f"dimension sizes {sizes}; all must have the same dimension size"
                )

        total_fixed = 0
        for g in self.elements():
            fixed = 1
            for cycle in g.full_cyclic_form:
                fixed *= size_dict[cycle[0]]
            total_fixed += fixed

        count, remainder = divmod(total_fixed, self.order())
        assert remainder == 0, (
            f"Burnside sum {total_fixed} not divisible by |G|={self.order()}"
        )
        return count

    # --- Convenience constructors ---

    @classmethod
    def symmetric(
        cls, k: int, *, axes: tuple[int, ...] | None = None
    ) -> PermutationGroup:
        """S_k: the full symmetric group. Generators: adjacent transpositions."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k == 1:
            return cls(Permutation.identity(1), axes=axes)
        gens = []
        for i in range(k - 1):
            arr = list(range(k))
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            gens.append(Permutation(arr))
        return cls(*gens, axes=axes)

    @classmethod
    def cyclic(cls, k: int, *, axes: tuple[int, ...] | None = None) -> PermutationGroup:
        """C_k: the cyclic group. Generator: the k-cycle (0 -> 1 -> ... -> k-1 -> 0)."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k == 1:
            return cls(Permutation.identity(1), axes=axes)
        gen = Permutation(list(range(1, k)) + [0])
        return cls(gen, axes=axes)

    @classmethod
    def dihedral(
        cls, k: int, *, axes: tuple[int, ...] | None = None
    ) -> PermutationGroup:
        """D_k: the dihedral group. Generators: k-cycle and reflection."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k <= 2:
            return cls.symmetric(k, axes=axes)
        rotation = Permutation(list(range(1, k)) + [0])
        refl_arr = [0] + list(range(k - 1, 0, -1))
        reflection = Permutation(refl_arr)
        return cls(rotation, reflection, axes=axes)

    # --- Sympy bridge ---

    def as_sympy(self):
        """Convert to ``sympy.combinatorics.PermutationGroup``. Requires sympy."""
        try:
            from sympy.combinatorics import PermutationGroup as SPermutationGroup
        except ImportError:
            raise ImportError(
                "sympy is required for as_sympy(). Install with: pip install sympy"
            ) from None
        return SPermutationGroup(*[g.as_sympy() for g in self._generators])

    @classmethod
    def from_sympy(
        cls, spg, *, axes: tuple[int, ...] | None = None
    ) -> PermutationGroup:
        """Construct from a ``sympy.combinatorics.PermutationGroup``."""
        gens = [Permutation.from_sympy(g) for g in spg.generators]
        return cls(*gens, axes=axes)

    def __repr__(self) -> str:
        axes_str = f", axes={self._axes}" if self._axes is not None else ""
        return f"PermutationGroup({', '.join(repr(g) for g in self._generators)}{axes_str})"


def _dimino(generators: tuple[Permutation, ...]) -> list[Permutation]:
    """Enumerate all group elements via Dimino's algorithm.

    Iteratively generates elements by composing known elements with generators
    until closure. Returns a list containing every element exactly once.
    """
    n = generators[0].size
    identity = Permutation.identity(n)
    elements = [identity]
    seen: set[Permutation] = {identity}

    for gen in generators:
        if gen in seen:
            continue
        coset = [gen]
        seen.add(gen)
        new_elements = [gen]
        while new_elements:
            next_new: list[Permutation] = []
            for elem in new_elements:
                for g in generators:
                    product = elem * g
                    if product not in seen:
                        seen.add(product)
                        next_new.append(product)
                    product_r = g * elem
                    if product_r not in seen:
                        seen.add(product_r)
                        next_new.append(product_r)
            new_elements = next_new
            coset.extend(next_new)
        elements.extend(coset)

    return elements
