"""End-to-end integration tests simulating participant usage."""

import numpy
import pytest

import mechestim as me


def test_simple_mlp_forward_pass():
    """Simulate a participant doing a forward pass through a small MLP."""
    numpy.random.seed(42)
    width = 16
    depth = 4
    weights = [
        me.array(numpy.random.randn(width, width) * numpy.sqrt(2.0 / width))
        for _ in range(depth)
    ]

    with me.BudgetContext(flop_budget=10**8) as budget:
        x = me.array(numpy.random.randn(100, width))
        for W in weights:
            x = me.einsum("bi,ji->bj", x, W)
            x = me.maximum(x, me.zeros_like(x))
        estimate = me.mean(x, axis=0)

        assert estimate.shape == (width,)
        assert budget.flops_used > 0
        summary = budget.summary()
        assert "einsum" in summary
        assert "maximum" in summary
        assert "mean" in summary
        einsum_ops = [r for r in budget.op_log if r.op_name == "einsum"]
        assert len(einsum_ops) == depth


def test_budget_tracking_accuracy():
    numpy.random.seed(42)
    A = me.array(numpy.random.randn(10, 20))
    B = me.array(numpy.random.randn(20, 30))

    with me.BudgetContext(flop_budget=10**8) as budget:
        me.einsum("ij,jk->ik", A, B)  # 10 * 20 * 30 = 6000 (FMA=1)
        me.exp(me.ones((100,)))  # 100
        me.sum(me.ones((50,)))  # 50
        assert budget.flops_used == 6000 + 100 + 50
        assert budget.flops_remaining == 10**8 - 6150


def test_flop_query_matches_execution():
    query_cost = me.flops.einsum_cost("ij,jk->ik", shapes=[(10, 20), (20, 30)])

    with me.BudgetContext(flop_budget=10**8) as budget:
        A = me.array(numpy.random.randn(10, 20))
        B = me.array(numpy.random.randn(20, 30))
        me.einsum("ij,jk->ik", A, B)

    assert budget.flops_used == query_cost


def test_mixed_free_and_counted():
    with me.BudgetContext(flop_budget=1000) as budget:
        x = me.zeros((10, 10))
        x = me.reshape(x, (100,))
        x = me.ones((5, 5))
        x = me.transpose(x)
        assert budget.flops_used == 0
        me.exp(me.ones((10,)))
        assert budget.flops_used == 10


def test_budget_exhaustion_mid_computation():
    with pytest.raises(me.BudgetExhaustedError) as exc_info:
        with me.BudgetContext(flop_budget=500) as budget:
            me.exp(me.ones((100,)))  # 100
            me.exp(me.ones((100,)))  # 200
            me.exp(me.ones((100,)))  # 300
            me.exp(me.ones((100,)))  # 400
            me.exp(me.ones((100,)))  # 500
            me.exp(me.ones((100,)))  # would be 600 — exceeds!
    assert exc_info.value.flops_remaining == 0
