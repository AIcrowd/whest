"""AccumulationCost orchestrator + per-component cost wrapping.

Aggregates the ladder primitive (compute_accumulation) into
einsum-shaped cost reports. Future reduction code reuses run_ladder_per_component
and adds its own aggregator (aggregate_reduction) that uses different cost arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._burnside import size_aware_burnside
from ._components import Component
from ._ladder import (
    AccumulationResult,
    RegimeId,
    RegimeStep,
    Shape,
    compute_accumulation,
)


@dataclass(frozen=True)
class ComponentCost:
    """Per-component cost. alpha is None when this component's regime returned
    unavailable (partition budget exceeded with brute-force disabled by policy)."""
    labels: tuple[str, ...]
    va: tuple[str, ...]
    wa: tuple[str, ...]
    sizes: tuple[int, ...]

    m: int
    alpha: int | None
    dense_count: int

    regime_id: RegimeId
    shape: Shape

    group_name: str
    group_order: int

    regime_trace: tuple[RegimeStep, ...]
    unavailable_reason: str | None = None

    def describe(self) -> dict[str, str]:
        """LaTeX strings built on demand (Task 23 fills in the body)."""
        from ._cost_descriptions import describe_component
        return describe_component(self)


def run_ladder_per_component(
    components: Sequence[Component],
    *,
    partition_budget: int,
) -> tuple[ComponentCost, ...]:
    """For each Component, run the ladder + Burnside, return ComponentCosts.

    Pure transformation, no aggregation policy. Reused by both einsum and (future)
    reduction code paths.
    """
    out: list[ComponentCost] = []
    for c in components:
        result: AccumulationResult = compute_accumulation(
            labels=c.labels,
            va=c.va,
            wa=c.wa,
            elements=c.elements,
            generators=c.generators,
            sizes=c.sizes,
            visible_positions=c.visible_positions,
            partition_budget=partition_budget,
        )
        # M is always computable via Burnside, even when α is unavailable.
        if c.elements and len(c.elements) > 0:
            m = size_aware_burnside(c.elements, c.sizes)
        else:
            m = math.prod(c.sizes) if c.sizes else 1
        dense_count = math.prod(c.sizes) if c.sizes else 1

        unavailable_reason: str | None = None
        if result.regime_id == 'unavailable':
            # The "unavailable" trace step's reason is the ladder's last word.
            unavailable_step = next(
                (s for s in result.trace if s.regime_id == 'unavailable'), None,
            )
            if unavailable_step is not None:
                unavailable_reason = unavailable_step.reason

        out.append(ComponentCost(
            labels=c.labels,
            va=c.va,
            wa=c.wa,
            sizes=c.sizes,
            m=m,
            alpha=result.count,
            dense_count=dense_count,
            regime_id=result.regime_id,
            shape=result.shape,
            group_name=c.group_name,
            group_order=c.order,
            regime_trace=result.trace,
            unavailable_reason=unavailable_reason,
        ))
    return tuple(out)
