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


# ── AccumulationCost + aggregate_einsum ──────────────────────────────


import warnings

from flopscope.errors import CostFallbackWarning


@dataclass(frozen=True)
class AccumulationCost:
    """Whole-einsum cost. When any component is unavailable, total falls back to
    the dense baseline (k · dense_baseline) and a CostFallbackWarning fires."""
    total: int
    mu: int | None
    alpha: int | None
    m_total: int
    dense_baseline: int
    num_terms: int

    per_component: tuple[ComponentCost, ...]

    fallback_used: bool
    unavailable_components: tuple[int, ...] = ()
    unavailable_reason: str | None = None

    def describe(self) -> dict:
        """Human-readable + LaTeX summary, built on demand."""
        from ._cost_descriptions import describe_total
        return describe_total(self)

    @property
    def savings_ratio(self) -> float:
        """total / (k · dense_baseline). 1.0 means no savings; lower is better."""
        denom = self.num_terms * self.dense_baseline
        return self.total / denom if denom > 0 else 1.0


def aggregate_einsum(
    component_costs: Sequence[ComponentCost],
    *,
    num_terms: int,
    dense_baseline: int,
) -> AccumulationCost:
    """Aggregate per-component costs into the einsum cost: total = (k-1)·∏M + ∏α.

    Fallback policy: if any component has alpha=None, total = k · dense_baseline
    (the no-symmetry direct-event count) and a CostFallbackWarning fires.
    """
    failing = [i for i, c in enumerate(component_costs) if c.alpha is None]

    m_total = 1
    for c in component_costs:
        m_total *= c.m

    if not failing:
        alpha_product = 1
        for c in component_costs:
            assert c.alpha is not None  # for type narrowing
            alpha_product *= c.alpha
        mu = (num_terms - 1) * m_total
        total = mu + alpha_product
        return AccumulationCost(
            total=total,
            mu=mu,
            alpha=alpha_product,
            m_total=m_total,
            dense_baseline=dense_baseline,
            num_terms=num_terms,
            per_component=tuple(component_costs),
            fallback_used=False,
        )

    # Fallback: charge dense.
    fallback_total = num_terms * dense_baseline
    first_failing = component_costs[failing[0]]
    reason = first_failing.unavailable_reason or 'partition_budget exceeded'
    failing_labels = ', '.join(first_failing.labels)
    warnings.warn(
        CostFallbackWarning(
            f'einsum: component {list(failing)} ({failing_labels}) returned '
            f'unavailable — charging dense cost {fallback_total} = '
            f'{num_terms} × {dense_baseline}. Failing reason: {reason}. '
            f'Raise via flopscope.configure(partition_budget=...) to attempt '
            f'exact counting.'
        ),
        stacklevel=4,
    )
    return AccumulationCost(
        total=fallback_total,
        mu=None,
        alpha=None,
        m_total=m_total,
        dense_baseline=dense_baseline,
        num_terms=num_terms,
        per_component=tuple(component_costs),
        fallback_used=True,
        unavailable_components=tuple(failing),
        unavailable_reason=reason,
    )


# ── End-to-end orchestrator ──────────────────────────────────────────


from collections.abc import Sequence as _Seq
from typing import Any as _Any

from flopscope._perm_group import SymmetryGroup as _SymGroup

from ._bipartite import build_bipartite, build_incidence_matrix
from ._components import decompose_into_components
from ._detection import build_full_group, run_sigma_loop
from ._wreath import enumerate_wreath


def _build_size_map(
    input_parts: _Seq[str],
    shapes: _Seq[_Seq[int]],
) -> dict[str, int]:
    """Build label → size from operand shapes. Validates that each label
    appears with consistent sizes across operands."""
    size_map: dict[str, int] = {}
    for part, shape in zip(input_parts, shapes, strict=True):
        for label, dim in zip(part, shape, strict=True):
            existing = size_map.get(label)
            if existing is not None and existing != dim:
                raise ValueError(
                    f"label '{label}' has inconsistent sizes {existing} and {dim} "
                    f"across operands"
                )
            size_map[label] = dim
    return size_map


def _operand_names_from_identity_pattern(
    num_ops: int,
    identity_pattern: tuple[tuple[int, ...], ...] | None,
) -> tuple[str, ...]:
    """Generate operand names that respect the identity pattern.
    Operands sharing the same id (per identity_pattern) get the same name."""
    if identity_pattern is None:
        return tuple(f'op_{i}' for i in range(num_ops))
    name_of: dict[int, str] = {}
    for group in identity_pattern:
        shared_name = f'op_grp_{group[0]}'
        for pos in group:
            name_of[pos] = shared_name
    return tuple(name_of.get(i, f'op_{i}') for i in range(num_ops))


def _per_op_symmetry_for_wreath(sym: _Any) -> _Any:
    """Pass declared symmetry through to enumerate_h. SymmetryGroup objects
    pass through unchanged; strings (like 'symmetric') also pass through."""
    return sym


def compute_accumulation_cost(
    *,
    canonical_subscripts: str,
    input_parts: _Seq[str],
    output_subscript: str,
    shapes: _Seq[_Seq[int]],
    per_op_symmetries: _Seq[_Any] | None,
    identity_pattern: tuple[tuple[int, ...], ...] | None,
    partition_budget: int | None = None,
) -> AccumulationCost:
    """End-to-end whole-expression cost.

    Inputs:
      canonical_subscripts: the einsum string after canonicalization
        (e.g. 'ij,jk->ik').
      input_parts: per-operand subscript strings.
      output_subscript: output labels (may be empty).
      shapes: per-operand shape tuples.
      per_op_symmetries: parallel to operands; SymmetryGroup objects, strings
        ('symmetric', 'cyclic', 'dihedral'), or None.
      identity_pattern: tuple of operand-position tuples that share id.
      partition_budget: per-component partition cap; defaults to global setting.
    """
    from flopscope._config import get_setting

    num_ops = len(input_parts)
    if per_op_symmetries is None:
        per_op_symmetries = (None,) * num_ops
    if partition_budget is None:
        partition_budget = int(get_setting('partition_budget'))

    operand_names = _operand_names_from_identity_pattern(num_ops, identity_pattern)

    graph = build_bipartite(
        subscripts=tuple(input_parts),
        output=output_subscript,
        operand_names=operand_names,
    )
    matrix_data = build_incidence_matrix(graph)

    # Build per-operand wreath inputs.
    axis_ranks = tuple(len(part) for part in input_parts)
    u_offsets = tuple(sum(axis_ranks[:i]) for i in range(num_ops))
    grouped_ops: set[int] = set()
    for grp in graph.identical_groups:
        grouped_ops.update(grp)
    singleton_groups = [(i,) for i in range(num_ops) if i not in grouped_ops]
    identical_groups_all = (*graph.identical_groups, *singleton_groups)

    wreath_elements = list(enumerate_wreath(
        identical_groups=identical_groups_all,
        per_op_symmetry=tuple(_per_op_symmetry_for_wreath(s) for s in per_op_symmetries),
        axis_ranks=axis_ranks,
        u_offsets=u_offsets,
    ))

    sigma_results = run_sigma_loop(graph, matrix_data, tuple(wreath_elements))
    detected = build_full_group(sigma_results, all_labels=graph.all_labels)

    # Decompose into components.
    size_map = _build_size_map(input_parts, shapes)
    sizes = tuple(size_map[lbl] for lbl in graph.all_labels)
    components = decompose_into_components(
        detected_group=detected,
        v_labels=graph.free_labels,
        w_labels=graph.summed_labels,
        sizes=sizes,
    )

    component_costs = run_ladder_per_component(
        components, partition_budget=partition_budget,
    )
    dense_baseline = math.prod(sizes) if sizes else 1
    return aggregate_einsum(
        component_costs=component_costs,
        num_terms=num_ops,
        dense_baseline=dense_baseline,
    )
