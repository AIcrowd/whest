"""Tests for whest.errors module."""

import pytest

from whest.errors import (
    BudgetExhaustedError,
    NoBudgetContextError,
    SymmetryError,
    WhestError,
    WhestServerError,
    WhestWarning,
    raise_from_response,
)

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_budget_exhausted_is_whest_error(self):
        assert issubclass(BudgetExhaustedError, WhestError)

    def test_no_budget_context_is_whest_error(self):
        assert issubclass(NoBudgetContextError, WhestError)

    def test_symmetry_error_is_whest_error(self):
        assert issubclass(SymmetryError, WhestError)

    def test_server_error_is_whest_error(self):
        assert issubclass(WhestServerError, WhestError)

    def test_warning_is_user_warning(self):
        assert issubclass(WhestWarning, UserWarning)

    def test_whest_error_is_exception(self):
        assert issubclass(WhestError, Exception)


# ---------------------------------------------------------------------------
# BudgetExhaustedError
# ---------------------------------------------------------------------------


class TestBudgetExhaustedError:
    def test_message_contains_op_name(self):
        err = BudgetExhaustedError(op_name="matmul", flop_cost=100, flops_remaining=50)
        assert "matmul" in str(err)

    def test_explicit_message_used_verbatim(self):
        err = BudgetExhaustedError(message="custom message here")
        assert str(err) == "custom message here"

    def test_default_construction(self):
        err = BudgetExhaustedError()
        assert isinstance(err, WhestError)

    def test_formatted_message_includes_cost(self):
        err = BudgetExhaustedError(op_name="svd", flop_cost=200, flops_remaining=10)
        msg = str(err)
        assert "200" in msg or "svd" in msg

    def test_can_be_raised_and_caught(self):
        with pytest.raises(BudgetExhaustedError):
            raise BudgetExhaustedError(op_name="dot", flop_cost=1, flops_remaining=0)

    def test_can_be_caught_as_whest_error(self):
        with pytest.raises(WhestError):
            raise BudgetExhaustedError(op_name="dot")


# ---------------------------------------------------------------------------
# NoBudgetContextError
# ---------------------------------------------------------------------------


class TestNoBudgetContextError:
    def test_default_message_mentions_budget_context(self):
        err = NoBudgetContextError()
        assert "BudgetContext" in str(err)

    def test_can_be_raised(self):
        with pytest.raises(NoBudgetContextError):
            raise NoBudgetContextError()

    def test_custom_message(self):
        err = NoBudgetContextError("wrap it")
        assert "wrap it" in str(err)


# ---------------------------------------------------------------------------
# SymmetryError
# ---------------------------------------------------------------------------


class TestSymmetryError:
    def test_is_whest_error(self):
        assert issubclass(SymmetryError, WhestError)

    def test_default_construction(self):
        err = SymmetryError()
        assert isinstance(err, SymmetryError)

    def test_explicit_message(self):
        err = SymmetryError(message="not symmetric")
        assert "not symmetric" in str(err)

    def test_dims_stored(self):
        err = SymmetryError(dims=(3, 4), max_deviation=0.5)
        assert isinstance(err, WhestError)

    def test_can_be_raised(self):
        with pytest.raises(SymmetryError):
            raise SymmetryError(dims=(2, 2), max_deviation=1e-6)


# ---------------------------------------------------------------------------
# WhestServerError
# ---------------------------------------------------------------------------


class TestWhestServerError:
    def test_is_whest_error(self):
        assert issubclass(WhestServerError, WhestError)

    def test_stores_message(self):
        err = WhestServerError("server blew up")
        assert "server blew up" in str(err)


# ---------------------------------------------------------------------------
# raise_from_response
# ---------------------------------------------------------------------------


class TestRaiseFromResponse:
    def test_raises_budget_exhausted(self):
        with pytest.raises(BudgetExhaustedError):
            raise_from_response("BudgetExhaustedError", "budget gone")

    def test_raises_no_budget_context(self):
        with pytest.raises(NoBudgetContextError):
            raise_from_response("NoBudgetContextError", "no context")

    def test_raises_symmetry_error(self):
        with pytest.raises(SymmetryError):
            raise_from_response("SymmetryError", "not symmetric")

    def test_raises_server_error(self):
        with pytest.raises(WhestServerError):
            raise_from_response("WhestServerError", "internal error")

    def test_raises_value_error(self):
        with pytest.raises(ValueError):
            raise_from_response("ValueError", "bad value")

    def test_raises_type_error(self):
        with pytest.raises(TypeError):
            raise_from_response("TypeError", "bad type")

    def test_raises_key_error(self):
        with pytest.raises(KeyError):
            raise_from_response("KeyError", "missing key")

    def test_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise_from_response("RuntimeError", "runtime failure")

    def test_invalid_request_maps_to_server_error(self):
        with pytest.raises(WhestServerError):
            raise_from_response("InvalidRequestError", "bad request")

    def test_unknown_type_raises_server_error(self):
        with pytest.raises(WhestServerError):
            raise_from_response("SomeUnknownError", "mystery error")

    def test_message_preserved_in_raised_exception(self):
        with pytest.raises(ValueError, match="specific message"):
            raise_from_response("ValueError", "specific message")

    def test_whest_error_message_preserved(self):
        with pytest.raises(BudgetExhaustedError) as exc_info:
            raise_from_response("BudgetExhaustedError", "budget is gone")
        assert "budget is gone" in str(exc_info.value)
