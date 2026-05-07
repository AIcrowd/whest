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


# Mixed regimes are added in subsequent tasks.
MIXED_REGIMES: tuple[Regime, ...] = ()
