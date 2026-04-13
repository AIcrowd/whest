"""Tests for encoding/decoding new types (Permutation, PermutationGroup, Cycle, SymmetryInfo)."""

from mechestim._remote_array import RemoteArray, _encode_arg, _result_from_response
from mechestim._symmetric_info import SymmetryInfo

from mechestim._perm_group import Cycle, Permutation, PermutationGroup


def test_encode_permutation():
    p = Permutation([2, 0, 1])
    encoded = _encode_arg(p)
    assert encoded == {"__permutation__": [2, 0, 1]}


def test_encode_cycle():
    c = Cycle(0, 2)(1, 3)
    encoded = _encode_arg(c)
    assert "__permutation__" in encoded
    assert isinstance(encoded["__permutation__"], list)


def test_encode_permutation_group():
    g = PermutationGroup.symmetric(3)
    encoded = _encode_arg(g)
    assert "__perm_group__" in encoded
    data = encoded["__perm_group__"]
    assert "generators" in data
    assert "degree" in data


def test_encode_nested_perm_in_list():
    p = Permutation([1, 0])
    encoded = _encode_arg([p, 42])
    assert encoded[0] == {"__permutation__": [1, 0]}
    assert encoded[1] == 42


def test_result_with_symmetry_info():
    resp = {
        "status": "ok",
        "result": {
            "id": "a5",
            "shape": [3, 3],
            "dtype": "float64",
            "symmetry_info": {
                "symmetric_axes": [[0, 1]],
                "shape": [3, 3],
            },
        },
        "budget": {},
    }
    result = _result_from_response(resp)
    assert isinstance(result, RemoteArray)
    assert result.symmetry_info is not None
    assert result.symmetry_info.symmetric_axes == [(0, 1)]
    assert result.symmetry_info.shape == (3, 3)


def test_result_without_symmetry_info():
    resp = {
        "status": "ok",
        "result": {"id": "a0", "shape": [2, 3], "dtype": "float64"},
        "budget": {},
    }
    result = _result_from_response(resp)
    assert isinstance(result, RemoteArray)
    assert result.symmetry_info is None
