import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestUnwrap:
    def test_result_matches_numpy(self):
        phase = numpy.array([0.0, 1.0, 2.0, 3.0, -3.0, -2.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import unwrap

            assert numpy.allclose(unwrap(phase), numpy.unwrap(phase))

    def test_cost(self):
        x = numpy.random.randn(20)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import unwrap

            unwrap(x)
            assert budget.flops_used == 20

    def test_outside_context_raises(self):
        from mechestim import unwrap

        with pytest.raises(NoBudgetContextError):
            unwrap(numpy.ones(5))
