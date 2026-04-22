"""Tests for encoding/decoding the current client symmetry transport surface."""

from whest._remote_array import RemoteArray, _encode_arg, _result_from_response
from whest._perm_group import SymmetryGroup


def test_encode_symmetry_group():
    group = SymmetryGroup.symmetric(axes=(0, 1, 2))
    encoded = _encode_arg(group)
    assert encoded == {
        "__symmetry_group__": {
            "axes": [0, 1, 2],
            "generators": [[1, 0, 2], [0, 2, 1]],
        }
    }


def test_encode_nested_symmetry_group_in_list():
    group = SymmetryGroup.symmetric(axes=(0, 1))
    encoded = _encode_arg([group, 42])
    assert encoded[0] == {
        "__symmetry_group__": {"axes": [0, 1], "generators": [[1, 0]]}
    }
    assert encoded[1] == 42


def test_result_with_symmetry_payload():
    resp = {
        "status": "ok",
        "result": {
            "id": "a5",
            "shape": [3, 3],
            "dtype": "float64",
            "symmetry": {
                "axes": [0, 1],
                "generators": [[1, 0]],
            },
        },
        "budget": {},
    }
    result = _result_from_response(resp)
    assert isinstance(result, RemoteArray)
    assert result.symmetry == SymmetryGroup.from_payload(
        {"axes": [0, 1], "generators": [[1, 0]]}
    )
    assert not hasattr(result, "symmetry_info")


def test_result_without_symmetry_payload():
    resp = {
        "status": "ok",
        "result": {"id": "a0", "shape": [2, 3], "dtype": "float64"},
        "budget": {},
    }
    result = _result_from_response(resp)
    assert isinstance(result, RemoteArray)
    assert result.symmetry is None
