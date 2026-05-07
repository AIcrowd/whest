"""Tests for the four mixed regimes in _regimes.py."""

import pytest

from flopscope._accumulation._ladder import RegimeContext
from flopscope._accumulation._regimes import FUNCTIONAL_PROJECTION_REGIME
from flopscope._perm_group import _Permutation as Permutation
from flopscope._perm_group import _dimino


def _ctx(*, labels, va, wa, elements, generators, sizes, visible_positions,
         partition_budget=100_000):
    return RegimeContext(
        labels=tuple(labels), va=tuple(va), wa=tuple(wa),
        elements=tuple(elements), generators=tuple(generators),
        sizes=tuple(sizes), visible_positions=tuple(visible_positions),
        partition_budget=partition_budget,
    )


# ── functionalProjection ──────────────────────────────────────────────


def test_functional_projection_fires_when_v_is_setwise_invariant():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    ctx = _ctx(
        labels=('i', 'j'), va=('i', 'j'), wa=(),
        elements=elements, generators=(swap,),
        sizes=(4, 4), visible_positions=(0, 1),
    )
    verdict = FUNCTIONAL_PROJECTION_REGIME.recognize(ctx)
    assert verdict.fired is True


def test_functional_projection_refuses_when_g_moves_v_to_w():
    cycle = Permutation([1, 2, 0])  # 0→1→2→0; visible {0,1} not preserved
    elements = _dimino((cycle,))
    ctx = _ctx(
        labels=('i', 'j', 'k'), va=('i', 'j'), wa=('k',),
        elements=elements, generators=(cycle,),
        sizes=(3, 3, 3), visible_positions=(0, 1),
    )
    verdict = FUNCTIONAL_PROJECTION_REGIME.recognize(ctx)
    assert verdict.fired is False
    assert 'output label into a summed label' in verdict.reason


def test_functional_projection_compute_returns_burnside_count():
    swap = Permutation([1, 0])
    identity = Permutation.identity(2)
    ctx = _ctx(
        labels=('i', 'j'), va=('i', 'j'), wa=(),
        elements=(identity, swap), generators=(swap,),
        sizes=(4, 4), visible_positions=(0, 1),
    )
    out = FUNCTIONAL_PROJECTION_REGIME.compute(ctx)
    # S_2 on (4, 4): 4·5/2 = 10
    assert out.count == 10
    assert len(out.sub_steps) == 1
    assert out.sub_steps[0]['step'] == 'projection-functional'
    assert out.sub_steps[0]['count'] == 10
