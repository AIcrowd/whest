"""Tests for _cost.py — ComponentCost, run_ladder_per_component, and aggregate_einsum."""

import math

from flopscope._accumulation._components import Component
from flopscope._accumulation._cost import ComponentCost, run_ladder_per_component
from flopscope._perm_group import _Permutation as Permutation
from flopscope._perm_group import _dimino


def _trivial_component(labels=('i',), sizes=(3,)):
    return Component(
        indices=tuple(range(len(labels))),
        labels=labels,
        va=labels,
        wa=(),
        sizes=sizes,
        visible_positions=tuple(range(len(labels))),
        generators=(),
        elements=(Permutation.identity(len(labels)),),
        order=1,
        group_name='trivial',
    )


def test_component_cost_trivial():
    component = _trivial_component()
    [cost] = run_ladder_per_component((component,), partition_budget=100_000)
    assert isinstance(cost, ComponentCost)
    assert cost.m == 3
    assert cost.alpha == 3
    assert cost.dense_count == 3
    assert cost.regime_id == 'trivial'
    assert cost.shape == 'trivial'
    assert cost.group_name == 'trivial'


def test_component_cost_s2_visible():
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    component = Component(
        indices=(0, 1),
        labels=('i', 'j'),
        va=('i', 'j'),
        wa=(),
        sizes=(4, 4),
        visible_positions=(0, 1),
        generators=(swap,),
        elements=tuple(elements),
        order=2,
        group_name='S2{i,j}',
    )
    [cost] = run_ladder_per_component((component,), partition_budget=100_000)
    # M = α (functional projection) = 4·5/2 = 10
    assert cost.m == 10
    assert cost.alpha == 10
    assert cost.regime_id == 'functionalProjection'
    assert cost.shape == 'allVisible'


def test_component_cost_unavailable_when_budget_zero():
    """A mixed S_2 component with partition_budget=0 → singleton fires before partitionCount."""
    swap = Permutation([1, 0])
    elements = _dimino((swap,))
    component = Component(
        indices=(0, 1), labels=('i', 'j'), va=('i',), wa=('j',),
        sizes=(6, 6), visible_positions=(0,),
        generators=(swap,), elements=tuple(elements), order=2,
        group_name='S2{i,j}',
    )
    [cost] = run_ladder_per_component((component,), partition_budget=0)
    # singleton fires before partitionCount, so this still gets a number.
    assert cost.alpha == 36
    assert cost.regime_id == 'singleton'


def test_component_cost_dense_count_is_product_of_sizes():
    # Use a trivial group so we don't need equal sizes (a swap on unequal sizes
    # is physically invalid because any cycle must have uniform dimension).
    component = Component(
        indices=(0, 1), labels=('i', 'j'), va=('i',), wa=('j',),
        sizes=(7, 11), visible_positions=(0,),
        generators=(), elements=(Permutation.identity(2),), order=1,
        group_name='trivial',
    )
    [cost] = run_ladder_per_component((component,), partition_budget=100_000)
    assert cost.dense_count == 77
