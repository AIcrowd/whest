"""Tests for new type deserialization and packing in the request handler."""

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


def test_resolve_permutation(handler):
    import whest as we

    arg = {"__permutation__": [2, 0, 1]}
    result = handler._resolve_arg(arg)
    assert isinstance(result, we.Permutation)
    assert result.array_form == [2, 0, 1]


def test_resolve_perm_group(handler):
    import whest as we

    arg = {
        "__perm_group__": {
            "generators": [[1, 0, 2], [0, 2, 1]],
            "degree": 3,
            "axes": [0, 1, 2],
        }
    }
    result = handler._resolve_arg(arg)
    assert isinstance(result, we.PermutationGroup)
    assert result.degree == 3


def test_pack_symmetric_tensor(handler):
    import whest as we

    arr = np.array([[1.0, 2.0], [2.0, 3.0]])
    st = we.as_symmetric(arr, symmetric_axes=[[0, 1]])
    result = handler._pack_result(st)
    assert result["status"] == "ok"
    assert "symmetry_info" in result["result"]
    si = result["result"]["symmetry_info"]
    assert "symmetric_axes" in si
    assert "shape" in si


def test_pack_regular_ndarray(handler):
    arr = np.array([1.0, 2.0, 3.0])
    result = handler._pack_result(arr)
    assert result["status"] == "ok"
    assert "symmetry_info" not in result["result"]
