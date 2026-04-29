"""Tests for current symmetry deserialization and packing in the request handler."""

import numpy as np
import pytest
from whest_server._request_handler import RequestHandler
from whest_server._session import Session


@pytest.fixture
def handler():
    session = Session(flop_budget=int(1e12))
    yield RequestHandler(session)
    if session.is_open:
        session.close()


def test_resolve_symmetry_group(handler):
    import whest as we

    arg = {
        "__symmetry_group__": {
            "generators": [[1, 0, 2], [0, 2, 1]],
            "axes": [0, 1, 2],
        }
    }
    result = handler._resolve_arg(arg)
    assert isinstance(result, we.SymmetryGroup)
    assert result.degree == 3
    assert result.axes == (0, 1, 2)


def test_pack_symmetric_tensor(handler):
    import whest as we

    arr = np.array([[1.0, 2.0], [2.0, 3.0]])
    st = we.as_symmetric(arr, symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)))
    result = handler._pack_result(st)
    assert result["status"] == "ok"
    assert "symmetry" in result["result"]
    assert result["result"]["symmetry"] == {"axes": [0, 1], "generators": [[1, 0]]}
    legacy_key = "symmetry" + "_info"
    assert legacy_key not in result["result"]


def test_pack_regular_ndarray(handler):
    arr = np.array([1.0, 2.0, 3.0])
    result = handler._pack_result(arr)
    assert result["status"] == "ok"
    assert "symmetry" not in result["result"]
