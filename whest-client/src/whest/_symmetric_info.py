"""Pure-Python SymmetryInfo for the whest client.

This is a read-only metadata container that records symmetry information
reported by the server. It does NOT include SymmetricTensor or the
computed properties that require numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    symmetric_axes: list[tuple[int, ...]] = field(default_factory=list)
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        normalized = [tuple(sorted(g)) for g in self.symmetric_axes]
        object.__setattr__(self, "symmetric_axes", normalized)
