"""Tests for FlopscopePathInfo wrapper."""

from flopscope._accumulation._cost import AccumulationCost, ComponentCost
from flopscope._accumulation._path_info import FlopscopePathInfo


def _trivial_cost():
    c = ComponentCost(
        labels=("i",),
        va=("i",),
        wa=(),
        sizes=(3,),
        m=3,
        alpha=3,
        dense_count=3,
        regime_id="trivial",
        shape="trivial",
        group_name="trivial",
        group_order=1,
        regime_trace=(),
    )
    return AccumulationCost(
        total=6,
        mu=3,
        alpha=3,
        m_total=3,
        dense_baseline=3,
        num_terms=2,
        per_component=(c,),
        fallback_used=False,
    )


def test_path_info_wrapper_carries_accumulation():
    fpi = FlopscopePathInfo.from_inner(inner=None, accumulation=_trivial_cost())
    assert fpi.accumulation is not None
    assert fpi.accumulation.total == 6


def test_path_info_optimized_cost_returns_accumulation_total():
    fpi = FlopscopePathInfo.from_inner(inner=None, accumulation=_trivial_cost())
    assert fpi.optimized_cost == 6


def test_path_info_falls_back_to_inner_when_no_accumulation():
    """If no accumulation attached, optimized_cost should fall back to the
    wrapped inner PathInfo's optimized_cost (legacy behavior)."""

    class _FakeInner:
        optimized_cost = 99
        path = []
        steps = []

    fpi = FlopscopePathInfo.from_inner(inner=_FakeInner(), accumulation=None)
    assert fpi.optimized_cost == 99
