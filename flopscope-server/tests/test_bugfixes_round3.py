"""Server-side tests for round-3 bugfixes."""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._array_store import ArrayStore
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def session():
    s = Session(flop_budget=1_000_000)
    yield s
    if s.is_open:
        s.close()


@pytest.fixture()
def handler(session):
    return RequestHandler(session)


# =========================================================================
# Fix 2: _decode_index_key on RequestHandler handles handle dicts
# =========================================================================


class TestFix2ServerDecodeIndexKey:
    """Instance method _decode_index_key resolves handle dicts."""

    def test_decode_handle_dict(self, handler, session):
        idx_arr = np.array([2, 0, 1], dtype=np.int64)
        idx_handle = session.store_array(idx_arr)
        result = handler._decode_index_key({"__handle__": idx_handle})
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, idx_arr)

    def test_decode_handle_dict_bytes_key(self, handler, session):
        idx_arr = np.array([1, 0], dtype=np.int64)
        idx_handle = session.store_array(idx_arr)
        result = handler._decode_index_key({b"__handle__": idx_handle})
        np.testing.assert_array_equal(result, idx_arr)

    def test_decode_slice_still_works(self, handler):
        result = handler._decode_index_key({"__slice__": [1, 5, 2]})
        assert result == slice(1, 5, 2)

    def test_decode_int_still_works(self, handler):
        assert handler._decode_index_key(3) == 3

    def test_fancy_index_getitem(self, handler, session):
        """Full round-trip: arr[idx_array] via __getitem__."""
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        arr_handle = session.store_array(arr)
        idx = np.array([3, 1, 0], dtype=np.int64)
        idx_handle = session.store_array(idx)

        resp = handler.handle(
            {
                "op": "__getitem__",
                "args": [{"__handle__": arr_handle}, {"__handle__": idx_handle}],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        result_handle = resp["result"]["id"]
        result_arr = session.get_array(result_handle)
        np.testing.assert_array_equal(result_arr, [40.0, 20.0, 10.0])


# =========================================================================
# Fix 3: astype handling on server
# =========================================================================


class TestFix3AstypeServer:
    """Server handles astype op as a special case."""

    def test_astype_float_to_int(self, handler, session):
        arr = np.array([1.5, 2.7, 3.1])
        handle = session.store_array(arr)
        resp = handler.handle(
            {
                "op": "astype",
                "args": [{"__handle__": handle}, "int64"],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        result_handle = resp["result"]["id"]
        result = session.get_array(result_handle)
        np.testing.assert_array_equal(result, [1, 2, 3])
        assert result.dtype == np.int64

    def test_astype_int_to_float(self, handler, session):
        arr = np.array([1, 2, 3], dtype=np.int64)
        handle = session.store_array(arr)
        resp = handler.handle(
            {
                "op": "astype",
                "args": [{"__handle__": handle}, "float32"],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        assert resp["result"]["dtype"] == "float32"

    def test_astype_bytes_dtype(self, handler, session):
        """dtype may arrive as bytes from msgpack."""
        arr = np.array([1.0, 2.0])
        handle = session.store_array(arr)
        resp = handler.handle(
            {
                "op": "astype",
                "args": [{"__handle__": handle}, b"int32"],
                "kwargs": {},
            }
        )
        assert resp["status"] == "ok"
        assert resp["result"]["dtype"] == "int32"

    def test_astype_in_whitelist(self):
        from flopscope_server._protocol import WHITELIST

        assert "astype" in WHITELIST


# =========================================================================
# Fix 7: ArrayStore count limit
# =========================================================================


class TestFix7ArrayStoreLimit:
    """ArrayStore raises MemoryError when limit is reached."""

    def test_put_within_limit(self):
        store = ArrayStore()
        for i in range(10):
            store.put(np.array([i]))
        assert store.count == 10

    def test_put_exceeds_limit(self, monkeypatch):
        """Exceeding the limit raises MemoryError."""
        import flopscope_server._array_store as mod

        monkeypatch.setattr(mod, "MAX_ARRAY_COUNT", 5)
        store = ArrayStore()
        for i in range(5):
            store.put(np.array([i]))
        with pytest.raises(MemoryError, match="array store limit"):
            store.put(np.array([999]))

    def test_free_then_put_succeeds(self, monkeypatch):
        """After freeing arrays, new ones can be stored."""
        import flopscope_server._array_store as mod

        monkeypatch.setattr(mod, "MAX_ARRAY_COUNT", 3)
        store = ArrayStore()
        handles = []
        for i in range(3):
            handles.append(store.put(np.array([i])))
        # At limit
        with pytest.raises(MemoryError):
            store.put(np.array([99]))
        # Free one
        store.free([handles[0]])
        # Now we can add
        store.put(np.array([99]))
        assert store.count == 3

    def test_handler_returns_error_on_memory_error(self, handler, session, monkeypatch):
        """MemoryError from ArrayStore is caught and returned as error response."""
        import flopscope_server._array_store as mod

        monkeypatch.setattr(mod, "MAX_ARRAY_COUNT", 2)
        # Use up slots
        session.store_array(np.array([1.0]))
        session.store_array(np.array([2.0]))
        # Next store should fail
        resp = handler.handle(
            {
                "op": "ones",
                "args": [[3]],
                "kwargs": {},
            }
        )
        # The server wraps the MemoryError in an error response
        assert resp["status"] == "error"


# =========================================================================
# Fix 3 (server): resolve_arg recursive for lists (already works)
# =========================================================================


class TestResolveArgRecursive:
    """_resolve_arg handles nested lists with handle dicts."""

    def test_resolve_list_of_handles(self, handler, session):
        a = session.store_array(np.array([1.0, 2.0]))
        b = session.store_array(np.array([3.0, 4.0]))
        resolved = handler._resolve_arg([{"__handle__": a}, {"__handle__": b}])
        assert len(resolved) == 2
        np.testing.assert_array_equal(resolved[0], [1.0, 2.0])
        np.testing.assert_array_equal(resolved[1], [3.0, 4.0])
