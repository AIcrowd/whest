"""Unit tests for fnp.random cost formula vocabulary."""

import numpy as np

from flopscope._flops import _ceil_log2


class TestNumelOutput:
    def test_array_result(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(output)"]
        assert formula((), {}, np.zeros((10, 20))) == 200

    def test_scalar_result(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(output)"]
        assert formula((), {}, np.float64(1.0)) == 1

    def test_zero_size_returns_one(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(output)"]
        assert formula((), {}, np.zeros(0)) == 1


class TestNumelInput:
    def test_first_positional_array(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(input)"]
        a = np.arange(50)
        assert formula((a,), {}, None) == 50

    def test_zero_size_returns_one(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["numel(input)"]
        assert formula((np.zeros(0),), {}, None) == 1


class TestLength:
    def test_positional(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["length"]
        assert formula((42,), {}, b"x" * 42) == 42

    def test_keyword(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["length"]
        assert formula((), {"length": 7}, b"y" * 7) == 7

    def test_zero_returns_one(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["length"]
        assert formula((0,), {}, b"") == 1


class TestSortCost:
    def test_integer_population(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["sort_cost(n)"]
        # `a` passed as the integer 16 (population size, range [0, 16))
        assert formula((16,), {}, np.zeros(5)) == 16 * _ceil_log2(16)

    def test_array_population(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["sort_cost(n)"]
        a = np.arange(20)
        assert formula((a,), {}, np.zeros(5)) == 20 * _ceil_log2(20)

    def test_zero_d_array_population(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["sort_cost(n)"]
        a = np.array(16)  # 0-D scalar array; numpy.choice accepts as int(a)
        assert formula((a,), {}, np.zeros(5)) == 16 * _ceil_log2(16)


class TestChoiceCost:
    def test_with_replacement_uses_numel_output(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        result = np.zeros(20)
        cost = formula((100,), {"size": 20, "replace": True}, result)
        assert cost == 20

    def test_without_replacement_uses_sort_cost(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        result = np.zeros(5)
        cost = formula((16,), {"size": 5, "replace": False}, result)
        assert cost == 16 * _ceil_log2(16)

    def test_replace_passed_positionally(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        formula = COST_FORMULAS["choice_cost"]
        # Generator.choice signature: choice(a, size, replace, p, axis, shuffle)
        result = np.zeros(5)
        cost = formula((16, 5, False), {}, result)
        assert cost == 16 * _ceil_log2(16)


class TestRegistry:
    def test_all_named_formulas_present(self):
        from flopscope.numpy.random._cost_formulas import COST_FORMULAS

        # Spec-mandated formula names
        assert set(COST_FORMULAS.keys()) >= {
            "numel(output)",
            "numel(input)",
            "length",
            "sort_cost(n)",
            "choice_cost",
        }
