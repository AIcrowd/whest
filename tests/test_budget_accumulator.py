"""Tests for BudgetAccumulator and budget_data()."""

import pytest

from mechestim._budget import (
    BudgetContext,
    BudgetAccumulator,
    NamespaceRecord,
    _reset_global_default,
    budget_data,
    budget_reset,
)


def test_namespace_record_fields():
    rec = NamespaceRecord(
        namespace="train",
        flop_budget=1000,
        flops_used=500,
        op_log=[],
    )
    assert rec.namespace == "train"
    assert rec.flop_budget == 1000
    assert rec.flops_used == 500


def test_budget_data_unlabeled():
    with BudgetContext(flop_budget=1000, namespace="a", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
        ctx.deduct("mul", flop_cost=200, subscripts=None, shapes=())

    data = budget_data()
    assert data["flops_used"] == 300
    assert data["flop_budget"] == 1000
    assert data["flops_remaining"] == 700
    assert data["operations"]["add"]["flop_cost"] == 100
    assert data["operations"]["add"]["calls"] == 1
    assert data["operations"]["mul"]["flop_cost"] == 200


def test_budget_data_by_namespace():
    with BudgetContext(flop_budget=1000, namespace="train", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
    with BudgetContext(flop_budget=500, namespace="eval", quiet=True) as ctx:
        ctx.deduct("mul", flop_cost=50, subscripts=None, shapes=())

    data = budget_data(by_namespace=True)
    assert data["flops_used"] == 150
    assert data["flop_budget"] == 1500
    assert "train" in data["by_namespace"]
    assert data["by_namespace"]["train"]["flops_used"] == 100
    assert "eval" in data["by_namespace"]
    assert data["by_namespace"]["eval"]["flops_used"] == 50


def test_budget_data_accumulates_across_contexts():
    with BudgetContext(flop_budget=1000, namespace="a", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
    with BudgetContext(flop_budget=2000, namespace="b", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=300, subscripts=None, shapes=())

    data = budget_data()
    assert data["flops_used"] == 400
    assert data["operations"]["add"]["flop_cost"] == 400
    assert data["operations"]["add"]["calls"] == 2


def test_budget_reset():
    with BudgetContext(flop_budget=1000, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
    budget_reset()
    data = budget_data()
    assert data["flops_used"] == 0
    assert data["operations"] == {}
