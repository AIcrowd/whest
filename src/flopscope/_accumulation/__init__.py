"""Symmetry-aware einsum accumulation cost — JS-mirrored α/M ladder.

Public surface:
    einsum_accumulation_cost(subscripts, *operands, partition_budget=None)
    AccumulationCost
    ComponentCost
    RegimeStep

Mirrors the JS engine in
``website/components/symmetry-aware-einsum-contractions/engine/``.
See ``.aicrowd/superpowers/specs/2026-05-07-symmetry-aware-einsum-cost-design.md``.
"""

# Re-exports populated incrementally by later tasks.
__all__ = []
