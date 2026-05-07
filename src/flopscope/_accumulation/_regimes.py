"""The four mixed-shape accumulation regimes plus functionalProjection.

Port of website/components/symmetry-aware-einsum-contractions/engine/regimes/*.js.

Each regime is a Regime instance with `recognize(ctx) -> Verdict` and
`compute(ctx) -> RegimeOutput`. The dispatcher in _ladder.py iterates them
in priority order; functionalProjection is checked first (Stage 2a) and the
remaining three form the mixed-shape ladder (Stage 2b).
"""

from __future__ import annotations

from ._burnside import size_aware_burnside
from ._ladder import Regime, RegimeContext, RegimeOutput, Verdict
from ._output_orbit import projection_is_functional


# ── functionalProjection ─────────────────────────────────────────────


def _functional_projection_recognize(ctx: RegimeContext) -> Verdict:
    if projection_is_functional(ctx.elements, ctx.visible_positions):
        return Verdict(
            fired=True,
            reason='each product orbit reaches exactly one stored output representative',
        )
    return Verdict(
        fired=False,
        reason='some pointwise symmetry moves an output label into a summed label',
    )


def _functional_projection_compute(ctx: RegimeContext) -> RegimeOutput:
    count = size_aware_burnside(ctx.elements, ctx.sizes)
    return RegimeOutput(
        count=count,
        sub_steps=(
            {
                'step': 'projection-functional',
                'reason': 'G preserves V setwise; projection descends to output reps',
                'count': count,
            },
        ),
    )


FUNCTIONAL_PROJECTION_REGIME: Regime = Regime(
    id='functionalProjection',
    recognize=_functional_projection_recognize,
    compute=_functional_projection_compute,
)


# ── singleton (B.5: |V| = 1) ─────────────────────────────────────────


def _label_orbit(elements, label_idx: int) -> list[int]:
    """G-orbit of a single label position."""
    seen = {label_idx}
    changed = True
    while changed:
        changed = False
        for g in elements:
            for p in list(seen):
                q = g.array_form[p]
                if q not in seen:
                    seen.add(q)
                    changed = True
    return sorted(seen)


def _cycles_on_subset(perm, subset: list[int]) -> int:
    """Count cycles of perm restricted to a perm-invariant subset."""
    subset_set = set(subset)
    seen: set[int] = set()
    cycles = 0
    for start in subset:
        if start in seen:
            continue
        cycles += 1
        cur = start
        while cur not in seen:
            if cur not in subset_set:
                raise ValueError('subset not invariant under perm')
            seen.add(cur)
            cur = perm.array_form[cur]
    return cycles


def _subset_cycle_product(perm, subset: list[int], sizes) -> int:
    """∏ n_c over cycles of perm in `subset`. Each cycle's labels must share a size."""
    subset_set = set(subset)
    seen: set[int] = set()
    product = 1
    for start in subset:
        if start in seen:
            continue
        cycle: list[int] = []
        cur = start
        while cur not in seen:
            if cur not in subset_set:
                raise ValueError('subset not invariant under perm')
            seen.add(cur)
            cycle.append(cur)
            cur = perm.array_form[cur]
        n0 = sizes[cycle[0]]
        for i in cycle:
            if sizes[i] != n0:
                raise ValueError('singleton: cycle in R has mixed sizes')
        product *= n0
    return product


def _singleton_recognize(ctx: RegimeContext) -> Verdict:
    if len(ctx.va) == 1:
        return Verdict(fired=True, reason='|V| = 1')
    return Verdict(fired=False, reason=f'|V| = {len(ctx.va)}, not 1')


def _singleton_compute(ctx: RegimeContext) -> RegimeOutput:
    v_pos = ctx.visible_positions[0]
    omega = _label_orbit(ctx.elements, v_pos)
    n_omega = ctx.sizes[v_pos]
    for idx in omega:
        if ctx.sizes[idx] != n_omega:
            raise ValueError(
                f'singleton: orbit of label has mixed sizes at {ctx.labels[idx]}'
            )
    omega_set = set(omega)
    rest = [i for i in range(len(ctx.labels)) if i not in omega_set]

    total = 0
    for g in ctx.elements:
        rest_factor = _subset_cycle_product(g, rest, ctx.sizes)
        c_omega = _cycles_on_subset(g, omega)
        total += rest_factor * (n_omega ** c_omega - (n_omega - 1) ** c_omega)
    count = (n_omega * total) // len(ctx.elements)
    return RegimeOutput(count=count, sub_steps=())


SINGLETON_REGIME: Regime = Regime(
    id='singleton',
    recognize=_singleton_recognize,
    compute=_singleton_compute,
)


MIXED_REGIMES: tuple[Regime, ...] = (
    SINGLETON_REGIME,
)
