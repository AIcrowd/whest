"""Unit tests for RemoteArray, RemoteScalar, and helpers.

These tests cover metadata, scalar operations, _result_from_response,
and _bytes_to_list.  Operator overloads and data-access methods that
require a running server are deferred to integration tests (Task 14).
"""

from __future__ import annotations

import struct

import pytest
from flopscope._remote_array import (
    RemoteArray,
    RemoteScalar,
    _bytes_to_list,
    _result_from_response,
)

# =========================================================================
# TestRemoteArrayMetadata
# =========================================================================


class TestRemoteArrayMetadata:
    """Cached metadata properties -- no server round-trip needed."""

    def test_shape(self):
        a = RemoteArray(handle_id="h1", shape=(3, 4), dtype="float64")
        assert a.shape == (3, 4)

    def test_dtype(self):
        a = RemoteArray(handle_id="h2", shape=(5,), dtype="int32")
        assert a.dtype == "int32"

    def test_ndim(self):
        a = RemoteArray(handle_id="h3", shape=(2, 3, 4), dtype="float64")
        assert a.ndim == 3

    def test_size(self):
        a = RemoteArray(handle_id="h4", shape=(2, 5), dtype="float64")
        assert a.size == 10

    def test_nbytes_float64(self):
        a = RemoteArray(handle_id="h5", shape=(4,), dtype="float64")
        assert a.nbytes == 4 * 8

    def test_nbytes_float32(self):
        a = RemoteArray(handle_id="h6", shape=(4,), dtype="float32")
        assert a.nbytes == 4 * 4

    def test_nbytes_int64(self):
        a = RemoteArray(handle_id="h7", shape=(3,), dtype="int64")
        assert a.nbytes == 3 * 8

    def test_len(self):
        a = RemoteArray(handle_id="h8", shape=(7, 3), dtype="float64")
        assert len(a) == 7

    def test_len_scalar_raises(self):
        a = RemoteArray(handle_id="h9", shape=(), dtype="float64")
        with pytest.raises(TypeError):
            len(a)

    def test_handle_id(self):
        a = RemoteArray(handle_id="abc-123", shape=(2,), dtype="float64")
        assert a.handle_id == "abc-123"


# =========================================================================
# TestRemoteScalar
# =========================================================================


class TestRemoteScalar:
    """RemoteScalar wraps a scalar value and behaves like a Python number."""

    def test_float_conversion(self):
        s = RemoteScalar(value=3.14, dtype="float64")
        assert float(s) == 3.14

    def test_int_conversion(self):
        s = RemoteScalar(value=7.0, dtype="float64")
        assert int(s) == 7

    def test_bool_truthy(self):
        s = RemoteScalar(value=1.0, dtype="float64")
        assert bool(s) is True

    def test_bool_falsy(self):
        s = RemoteScalar(value=0.0, dtype="float64")
        assert bool(s) is False

    def test_repr_contains_value(self):
        s = RemoteScalar(value=42.0, dtype="float64")
        assert "42.0" in repr(s)

    def test_arithmetic_float_add(self):
        s = RemoteScalar(value=2.5, dtype="float64")
        assert float(s) + 1 == 3.5

    def test_comparison_gt(self):
        s = RemoteScalar(value=5.0, dtype="float64")
        assert s > 3.0

    def test_comparison_lt(self):
        s = RemoteScalar(value=2.0, dtype="float64")
        assert s < 4.0

    def test_comparison_eq(self):
        s = RemoteScalar(value=3.0, dtype="float64")
        assert s == 3.0

    def test_comparison_le(self):
        s = RemoteScalar(value=3.0, dtype="float64")
        assert s <= 3.0
        assert s <= 4.0

    def test_comparison_ge(self):
        s = RemoteScalar(value=3.0, dtype="float64")
        assert s >= 3.0
        assert s >= 2.0

    def test_shape_is_empty_tuple(self):
        s = RemoteScalar(value=1.0, dtype="float64")
        assert s.shape == ()

    def test_ndim_is_zero(self):
        s = RemoteScalar(value=1.0, dtype="float64")
        assert s.ndim == 0

    def test_size_is_one(self):
        s = RemoteScalar(value=1.0, dtype="float64")
        assert s.size == 1

    def test_handle_id_is_none(self):
        s = RemoteScalar(value=1.0, dtype="float64")
        assert s.handle_id is None

    def test_tolist(self):
        s = RemoteScalar(value=9.5, dtype="float64")
        assert s.tolist() == 9.5

    def test_hash(self):
        s1 = RemoteScalar(value=3.0, dtype="float64")
        s2 = RemoteScalar(value=3.0, dtype="float64")
        assert hash(s1) == hash(s2)
        # Hashable means it can be a dict key
        d = {s1: "ok"}
        assert d[s2] == "ok"

    def test_str(self):
        s = RemoteScalar(value=1.5, dtype="float64")
        assert "1.5" in str(s)


# =========================================================================
# TestResultFromResponse
# =========================================================================


class TestResultFromResponse:
    """_result_from_response dispatches on result keys."""

    def test_single_array_result(self):
        resp = {
            "status": "ok",
            "result": {
                "id": "handle-abc",
                "shape": [3, 4],
                "dtype": "float64",
            },
        }
        out = _result_from_response(resp)
        assert isinstance(out, RemoteArray)
        assert out.handle_id == "handle-abc"
        assert out.shape == (3, 4)
        assert out.dtype == "float64"

    def test_scalar_result(self):
        resp = {
            "status": "ok",
            "result": {"value": 42.0},
        }
        out = _result_from_response(resp)
        assert isinstance(out, RemoteScalar)
        assert float(out) == 42.0

    def test_multi_result(self):
        resp = {
            "status": "ok",
            "result": {
                "multi": [
                    {"id": "h1", "shape": [2], "dtype": "float64"},
                    {"id": "h2", "shape": [3], "dtype": "float32"},
                ],
            },
        }
        out = _result_from_response(resp)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], RemoteArray)
        assert out[0].handle_id == "h1"
        assert out[1].dtype == "float32"


# =========================================================================
# TestBytesToList
# =========================================================================


class TestBytesToList:
    """_bytes_to_list converts raw bytes to nested Python lists without numpy."""

    def test_1d_float64(self):
        vals = [1.0, 2.0, 3.0]
        data = struct.pack("3d", *vals)
        result = _bytes_to_list(data, (3,), "float64")
        assert result == [1.0, 2.0, 3.0]

    def test_2d_float32(self):
        # 2x3 matrix
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        data = struct.pack("6f", *vals)
        result = _bytes_to_list(data, (2, 3), "float32")
        assert len(result) == 2
        assert len(result[0]) == 3
        # float32 has limited precision, so compare approximately
        assert result[0] == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)
        assert result[1] == pytest.approx([4.0, 5.0, 6.0], abs=1e-6)

    def test_scalar_empty_shape(self):
        data = struct.pack("d", 7.5)
        result = _bytes_to_list(data, (), "float64")
        assert result == 7.5

    def test_1d_int64(self):
        vals = [10, 20, 30]
        data = struct.pack("3q", *vals)
        result = _bytes_to_list(data, (3,), "int64")
        assert result == [10, 20, 30]

    def test_1d_int32(self):
        vals = [1, 2, 3, 4]
        data = struct.pack("4i", *vals)
        result = _bytes_to_list(data, (4,), "int32")
        assert result == [1, 2, 3, 4]

    def test_3d_float64(self):
        # 2x2x2
        vals = list(range(1, 9))
        data = struct.pack("8d", *[float(v) for v in vals])
        result = _bytes_to_list(data, (2, 2, 2), "float64")
        assert result == [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]

    # -- FIX 4: float16, complex64, complex128 support -----------------------

    def test_1d_float16(self):
        vals = [1.0, 2.0, 3.0]
        data = struct.pack("<3e", *vals)
        result = _bytes_to_list(data, (3,), "float16")
        assert result == pytest.approx([1.0, 2.0, 3.0], abs=1e-3)

    def test_1d_complex64(self):
        # complex64 = two float32 components per element
        vals = [1.0, 2.0, 3.0, 4.0]  # (1+2j), (3+4j)
        data = struct.pack("<4f", *vals)
        result = _bytes_to_list(data, (2,), "complex64")
        assert len(result) == 2
        assert result[0] == pytest.approx(complex(1.0, 2.0), abs=1e-6)
        assert result[1] == pytest.approx(complex(3.0, 4.0), abs=1e-6)

    def test_1d_complex128(self):
        # complex128 = two float64 components per element
        vals = [1.0, 2.0, 3.0, 4.0]  # (1+2j), (3+4j)
        data = struct.pack("<4d", *vals)
        result = _bytes_to_list(data, (2,), "complex128")
        assert len(result) == 2
        assert result[0] == complex(1.0, 2.0)
        assert result[1] == complex(3.0, 4.0)

    def test_scalar_complex128(self):
        data = struct.pack("<2d", 5.0, -3.0)
        result = _bytes_to_list(data, (), "complex128")
        assert result == complex(5.0, -3.0)


# =========================================================================
# TestResultFromResponseMixed (FIX 6)
# =========================================================================


class TestResultFromResponseMixed:
    """FIX 6: mixed multi results with both arrays and scalars."""

    def test_multi_with_scalar(self):
        resp = {
            "status": "ok",
            "result": {
                "multi": [
                    {"id": "h1", "shape": [2], "dtype": "float64"},
                    {"value": 3.14},
                ],
            },
        }
        out = _result_from_response(resp)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], RemoteArray)
        assert isinstance(out[1], RemoteScalar)
        assert float(out[1]) == pytest.approx(3.14)


# =========================================================================
# TestRemoteArrayReverseOps (FIX 5)
# =========================================================================


class TestRemoteArrayReverseOps:
    """FIX 5: reverse operators exist on RemoteArray."""

    def test_has_rfloordiv(self):
        assert hasattr(RemoteArray, "__rfloordiv__")

    def test_has_rmod(self):
        assert hasattr(RemoteArray, "__rmod__")

    def test_has_rpow(self):
        assert hasattr(RemoteArray, "__rpow__")

    def test_has_rmatmul(self):
        assert hasattr(RemoteArray, "__rmatmul__")
