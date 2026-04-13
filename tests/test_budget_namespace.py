"""Tests for namespace support on BudgetContext and OpRecord."""

from whest._budget import BudgetContext, OpRecord


def test_oprecord_has_namespace_field():
    rec = OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((10,),),
        flop_cost=10,
        cumulative=10,
        namespace="training",
    )
    assert rec.namespace == "training"


def test_oprecord_namespace_defaults_none():
    rec = OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((10,),),
        flop_cost=10,
        cumulative=10,
        namespace=None,
    )
    assert rec.namespace is None


def test_budget_context_accepts_namespace():
    with BudgetContext(flop_budget=1000, namespace="training") as ctx:
        assert ctx.namespace == "training"


def test_budget_context_namespace_defaults_none():
    with BudgetContext(flop_budget=1000) as ctx:
        assert ctx.namespace is None


def test_deduct_records_namespace():
    with BudgetContext(flop_budget=1000, namespace="inference") as ctx:
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=((5,),))
        assert ctx.op_log[0].namespace == "inference"


def test_deduct_records_none_namespace():
    with BudgetContext(flop_budget=1000) as ctx:
        ctx.deduct("add", flop_cost=10, subscripts=None, shapes=((5,),))
        assert ctx.op_log[0].namespace is None


def test_summary_includes_namespace_label():
    with BudgetContext(flop_budget=1000, namespace="training", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
        s = ctx.summary()
        assert "[training]" in s


def test_summary_without_namespace():
    with BudgetContext(flop_budget=1000, quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
        s = ctx.summary()
        assert "100" in s.replace(",", "")
        assert "[" not in s.split("\n")[0]  # no bracket in header
