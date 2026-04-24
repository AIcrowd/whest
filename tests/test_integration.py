"""End-to-end integration tests simulating participant usage."""

import numpy
import pytest

import whest as we


def test_simple_mlp_forward_pass():
    """Simulate a participant doing a forward pass through a small MLP."""
    numpy.random.seed(42)
    width = 16
    depth = 4
    weights = [
        we.array(numpy.random.randn(width, width) * numpy.sqrt(2.0 / width))
        for _ in range(depth)
    ]

    with we.BudgetContext(flop_budget=10**8) as budget:
        x = we.array(numpy.random.randn(100, width))
        for W in weights:
            x = we.einsum("bi,ji->bj", x, W)
            x = we.maximum(x, we.zeros_like(x))
        estimate = we.mean(x, axis=0)

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
    A = we.array(numpy.random.randn(10, 20))
    B = we.array(numpy.random.randn(20, 30))

    with we.BudgetContext(flop_budget=10**8) as budget:
        we.einsum("ij,jk->ik", A, B)  # 10 * 20 * 30 = 6000 (FMA=1)
        we.exp(we.ones((100,)))  # 100
        we.sum(we.ones((50,)))  # 50 − 1 = 49
        assert budget.flops_used == 6000 + 100 + 49
        assert budget.flops_remaining == 10**8 - 6149


def test_flop_query_matches_execution():
    query_cost = we.flops.einsum_cost("ij,jk->ik", shapes=[(10, 20), (20, 30)])

    with we.BudgetContext(flop_budget=10**8) as budget:
        A = we.array(numpy.random.randn(10, 20))  # 200 (array creation)
        B = we.array(numpy.random.randn(20, 30))  # 600 (array creation)
        we.einsum("ij,jk->ik", A, B)

    assert budget.flops_used == query_cost + 200 + 600


def test_mixed_free_and_counted():
    with we.BudgetContext(flop_budget=1000) as budget:
        x = we.zeros((10, 10))
        x = we.reshape(x, (100,))
        x = we.ones((5, 5))
        x = we.transpose(x)
        assert budget.flops_used == 0
        we.exp(we.ones((10,)))
        assert budget.flops_used == 10


def test_budget_exhaustion_mid_computation():
    with pytest.raises(we.BudgetExhaustedError) as exc_info:
        with we.BudgetContext(flop_budget=500) as budget:
            we.exp(we.ones((100,)))  # 100
            we.exp(we.ones((100,)))  # 200
            we.exp(we.ones((100,)))  # 300
            we.exp(we.ones((100,)))  # 400
            we.exp(we.ones((100,)))  # 500
            we.exp(we.ones((100,)))  # would be 600 — exceeds!
    assert exc_info.value.flops_remaining == 0
