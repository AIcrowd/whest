# Mechestim Client-Server Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split mechestim into a lightweight client (no numpy) and a backend server connected via ZeroMQ + msgpack, so adversarial competition participants cannot access numpy directly.

**Architecture:** Three packages — `mechestim` (existing, unchanged), `mechestim-client` (drop-in proxy, no numpy), `mechestim-server` (daemon with numpy + mechestim). Client sends operation requests over ZeroMQ, server executes via real mechestim, returns opaque array handles. Client's `RemoteArray` transparently proxies all operations including Python operators and data access.

**Tech Stack:** Python 3.10+, ZeroMQ (pyzmq), msgpack, existing mechestim + numpy on server side.

**Spec:** `docs/superpowers/specs/2026-04-01-client-server-architecture-design.md`

---

## File Structure

### mechestim-server (new package)

```
mechestim-server/
├── pyproject.toml
├── src/
│   └── mechestim_server/
│       ├── __init__.py              # Version, public API
│       ├── __main__.py              # Entry point: python -m mechestim_server
│       ├── _array_store.py          # ArrayStore: handle→ndarray mapping + ID generation
│       ├── _session.py              # Session: ArrayStore + BudgetContext + CommsTracker
│       ├── _comms_tracker.py        # Per-request timing + session-level aggregation
│       ├── _request_handler.py      # Dispatch msgpack requests → mechestim calls
│       ├── _protocol.py             # Message schema validation, encode/decode helpers
│       └── _server.py               # ZMQ REP loop, session lifecycle, timeout reaping
└── tests/
    ├── test_array_store.py
    ├── test_session.py
    ├── test_comms_tracker.py
    ├── test_request_handler.py
    ├── test_protocol.py
    └── test_server.py
```

### mechestim-client (new package)

```
mechestim-client/
├── pyproject.toml
├── src/
│   └── mechestim/                   # Same package name as real mechestim
│       ├── __init__.py              # Auto-generated proxy exports, constants, dtypes
│       ├── _remote_array.py         # RemoteArray + RemoteScalar transparent proxies
│       ├── _connection.py           # ZMQ REQ socket, env var config, send/recv
│       ├── _protocol.py             # Argument encoding (handles→IDs), response decoding
│       ├── _budget.py               # BudgetContext proxy (delegates to server)
│       ├── _comms_tracker.py        # Client-side round-trip timing
│       ├── _registry.py             # Static function registry (categories only, no numpy)
│       ├── _getattr.py              # Module-level __getattr__ for blacklisted/unknown funcs
│       ├── errors.py                # Same exception classes as real mechestim
│       ├── flops.py                 # Proxy to server-side cost queries
│       ├── linalg/
│       │   └── __init__.py          # svd proxy + __getattr__
│       ├── random/
│       │   └── __init__.py          # randn, normal, etc. proxies + __getattr__
│       └── fft/
│           └── __init__.py          # Blacklisted stubs, same errors
└── tests/
    ├── test_remote_array.py
    ├── test_connection.py
    ├── test_protocol.py
    ├── test_budget.py
    ├── test_comms_tracker.py
    ├── test_registry.py
    ├── test_operators.py
    ├── test_submodules.py
    └── test_integration.py          # Full end-to-end with real server
```

---

## Task 1: Scaffold Both Packages

**Files:**
- Create: `mechestim-server/pyproject.toml`
- Create: `mechestim-server/src/mechestim_server/__init__.py`
- Create: `mechestim-client/pyproject.toml`
- Create: `mechestim-client/src/mechestim/__init__.py`

- [ ] **Step 1: Create mechestim-server package scaffold**

```toml
# mechestim-server/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mechestim-server"
version = "0.1.0"
description = "Backend server for mechestim client-server architecture"
requires-python = ">=3.10"
dependencies = [
    "mechestim>=0.2.0",
    "numpy>=2.1.0,<2.2.0",
    "pyzmq>=26.0.0",
    "msgpack>=1.0.0",
]

[project.scripts]
mechestim-server = "mechestim_server.__main__:main"
```

```python
# mechestim-server/src/mechestim_server/__init__.py
"""Mechestim backend server — executes numpy operations on behalf of remote clients."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Create mechestim-client package scaffold**

```toml
# mechestim-client/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mechestim-client"
version = "0.1.0"
description = "Lightweight mechestim client — drop-in replacement with no numpy dependency"
requires-python = ">=3.10"
dependencies = [
    "pyzmq>=26.0.0",
    "msgpack>=1.0.0",
]
```

```python
# mechestim-client/src/mechestim/__init__.py
"""mechestim — transparent proxy to a remote mechestim server."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Verify both packages install**

Run:
```bash
cd mechestim-server && pip install -e . && cd ..
cd mechestim-client && pip install -e . && cd ..
python -c "import mechestim_server; print(mechestim_server.__version__)"
```
Expected: `0.1.0`

Note: You cannot have both `mechestim` (real) and `mechestim-client` installed simultaneously — they share the same package name. For development, use a venv per package.

- [ ] **Step 4: Commit**

```bash
git add mechestim-server/ mechestim-client/
git commit -m "feat: scaffold mechestim-server and mechestim-client packages"
```

---

## Task 2: Protocol Layer — Message Encoding/Decoding

The protocol layer defines the message format used by both client and server. Each side has its own `_protocol.py` since they have different dependencies, but the wire format is identical.

**Files:**
- Create: `mechestim-server/src/mechestim_server/_protocol.py`
- Create: `mechestim-server/tests/test_protocol.py`

- [ ] **Step 1: Write failing tests for protocol encode/decode**

```python
# mechestim-server/tests/test_protocol.py
import msgpack
import numpy as np

from mechestim_server._protocol import (
    encode_response,
    decode_request,
    encode_error_response,
    encode_fetch_response,
    validate_request,
    InvalidRequestError,
)


class TestDecodeRequest:
    def test_decode_op_request(self):
        raw = msgpack.packb({
            "op": "add",
            "args": ["a1", "a2"],
            "kwargs": {},
            "request_id": "r001",
        })
        req = decode_request(raw)
        assert req["op"] == "add"
        assert req["args"] == ["a1", "a2"]
        assert req["request_id"] == "r001"

    def test_decode_create_from_data(self):
        data = np.ones((2, 3), dtype=np.float64).tobytes()
        raw = msgpack.packb({
            "op": "create_from_data",
            "data": data,
            "shape": [2, 3],
            "dtype": "float64",
            "request_id": "r002",
        })
        req = decode_request(raw)
        assert req["op"] == "create_from_data"
        assert req["shape"] == [2, 3]
        assert isinstance(req["data"], bytes)

    def test_decode_budget_open(self):
        raw = msgpack.packb({
            "op": "budget_open",
            "flop_budget": 10_000_000,
            "request_id": "r003",
        })
        req = decode_request(raw)
        assert req["op"] == "budget_open"
        assert req["flop_budget"] == 10_000_000

    def test_decode_invalid_msgpack(self):
        import pytest
        with pytest.raises(InvalidRequestError):
            decode_request(b"not valid msgpack \xff\xff")

    def test_decode_missing_op(self):
        import pytest
        raw = msgpack.packb({"args": ["a1"]})
        with pytest.raises(InvalidRequestError, match="missing 'op'"):
            decode_request(raw)


class TestValidateRequest:
    def test_validate_known_op(self):
        validate_request({"op": "add", "args": ["a1", "a2"]})

    def test_validate_unknown_op_raises(self):
        import pytest
        with pytest.raises(InvalidRequestError, match="unknown op"):
            validate_request({"op": "hack_the_planet", "args": []})

    def test_validate_budget_ops(self):
        validate_request({"op": "budget_open", "flop_budget": 1000})
        validate_request({"op": "budget_close"})
        validate_request({"op": "budget_status"})

    def test_validate_fetch(self):
        validate_request({"op": "fetch", "id": "a1"})

    def test_validate_free(self):
        validate_request({"op": "free", "ids": ["a1", "a2"]})


class TestEncodeResponse:
    def test_encode_op_result(self):
        resp = encode_response(
            result={"id": "a3", "shape": [256], "dtype": "float64"},
            budget={"flops_used": 256, "flops_remaining": 9999744},
            comms_overhead_ns=48000,
        )
        decoded = msgpack.unpackb(resp)
        assert decoded["status"] == "ok"
        assert decoded["result"]["id"] == "a3"
        assert decoded["budget"]["flops_used"] == 256
        assert decoded["budget"]["comms_overhead_ns"] == 48000

    def test_encode_error_response(self):
        resp = encode_error_response(
            error_type="BudgetExhaustedError",
            message="Budget exhausted: used 10M of 10M FLOPs",
        )
        decoded = msgpack.unpackb(resp)
        assert decoded["status"] == "error"
        assert decoded["error_type"] == "BudgetExhaustedError"

    def test_encode_fetch_response(self):
        data = b"\x00" * 2048
        resp = encode_fetch_response(
            data=data,
            shape=[256],
            dtype="float64",
            comms_overhead_ns=120000,
        )
        decoded = msgpack.unpackb(resp)
        assert decoded["status"] == "ok"
        assert decoded["data"] == data
        assert decoded["shape"] == [256]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-server && python -m pytest tests/test_protocol.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement server protocol module**

```python
# mechestim-server/src/mechestim_server/_protocol.py
"""Message encoding/decoding for the mechestim wire protocol."""

from __future__ import annotations

import msgpack

from mechestim._registry import REGISTRY

# Operations that are part of the protocol but not in the mechestim registry
_PROTOCOL_OPS = frozenset({
    "budget_open", "budget_close", "budget_status",
    "fetch", "fetch_slice", "free",
    "create_from_data",
})

# All valid op names: registry functions + protocol ops
_VALID_OPS = frozenset(REGISTRY.keys()) | _PROTOCOL_OPS


class InvalidRequestError(Exception):
    """Raised when a request is malformed or contains an unknown op."""


def decode_request(raw: bytes) -> dict:
    """Decode a msgpack-encoded request. Raises InvalidRequestError on failure."""
    try:
        msg = msgpack.unpackb(raw, raw=True)
    except (msgpack.UnpackException, ValueError) as e:
        raise InvalidRequestError(f"invalid msgpack: {e}") from e

    # msgpack with raw=True returns bytes keys; normalize to str
    msg = {
        (k.decode() if isinstance(k, bytes) else k): (
            v.decode() if isinstance(v, bytes) and k in (b"op", b"dtype", b"request_id") else
            [vi.decode() if isinstance(vi, bytes) else vi for vi in v] if isinstance(v, list) and k in (b"args", b"ids") else
            v
        )
        for k, v in msg.items()
    }

    if "op" not in msg:
        raise InvalidRequestError("missing 'op' field in request")

    return msg


def validate_request(msg: dict) -> None:
    """Validate that the request op is in the whitelist. Raises InvalidRequestError."""
    op = msg.get("op", "")
    if op not in _VALID_OPS:
        raise InvalidRequestError(f"unknown op: {op!r}")


def encode_response(
    result: dict | None = None,
    budget: dict | None = None,
    comms_overhead_ns: int = 0,
) -> bytes:
    """Encode a successful operation response."""
    resp = {"status": "ok"}
    if result is not None:
        resp["result"] = result
    if budget is not None:
        budget["comms_overhead_ns"] = comms_overhead_ns
        resp["budget"] = budget
    return msgpack.packb(resp)


def encode_error_response(error_type: str, message: str) -> bytes:
    """Encode an error response."""
    return msgpack.packb({
        "status": "error",
        "error_type": error_type,
        "message": message,
    })


def encode_fetch_response(
    data: bytes,
    shape: list[int],
    dtype: str,
    comms_overhead_ns: int = 0,
) -> bytes:
    """Encode a fetch response with raw array data."""
    return msgpack.packb({
        "status": "ok",
        "data": data,
        "shape": shape,
        "dtype": dtype,
        "comms_overhead_ns": comms_overhead_ns,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-server && python -m pytest tests/test_protocol.py -v`
Expected: All PASS

Note: The `decode_request` function's key normalization is intentionally simple. msgpack with `raw=True` returns bytes keys for performance. We selectively decode string fields (op, dtype, request_id) and string list items (args handle IDs). Binary data fields (like `data` for create_from_data) stay as bytes.

- [ ] **Step 5: Commit**

```bash
git add mechestim-server/src/mechestim_server/_protocol.py mechestim-server/tests/test_protocol.py
git commit -m "feat(server): add wire protocol encode/decode with whitelist validation"
```

---

## Task 3: Server ArrayStore

**Files:**
- Create: `mechestim-server/src/mechestim_server/_array_store.py`
- Create: `mechestim-server/tests/test_array_store.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-server/tests/test_array_store.py
import numpy as np
import pytest

from mechestim_server._array_store import ArrayStore


class TestArrayStore:
    def test_put_and_get(self):
        store = ArrayStore()
        arr = np.ones((3, 4), dtype=np.float64)
        handle = store.put(arr)
        assert handle == "a0"
        retrieved = store.get(handle)
        np.testing.assert_array_equal(retrieved, arr)

    def test_sequential_ids(self):
        store = ArrayStore()
        h0 = store.put(np.zeros(1))
        h1 = store.put(np.zeros(2))
        h2 = store.put(np.zeros(3))
        assert h0 == "a0"
        assert h1 == "a1"
        assert h2 == "a2"

    def test_get_missing_raises(self):
        store = ArrayStore()
        with pytest.raises(KeyError, match="a99"):
            store.get("a99")

    def test_free_single(self):
        store = ArrayStore()
        h = store.put(np.zeros(5))
        store.free([h])
        with pytest.raises(KeyError):
            store.get(h)

    def test_free_multiple(self):
        store = ArrayStore()
        h0 = store.put(np.zeros(1))
        h1 = store.put(np.zeros(2))
        h2 = store.put(np.zeros(3))
        store.free([h0, h2])
        with pytest.raises(KeyError):
            store.get(h0)
        np.testing.assert_array_equal(store.get(h1), np.zeros(2))
        with pytest.raises(KeyError):
            store.get(h2)

    def test_clear(self):
        store = ArrayStore()
        store.put(np.zeros(1))
        store.put(np.zeros(2))
        assert store.count == 2
        store.clear()
        assert store.count == 0

    def test_count(self):
        store = ArrayStore()
        assert store.count == 0
        store.put(np.zeros(1))
        assert store.count == 1

    def test_metadata(self):
        store = ArrayStore()
        arr = np.ones((10, 20), dtype=np.float32)
        h = store.put(arr)
        meta = store.metadata(h)
        assert meta["id"] == h
        assert meta["shape"] == [10, 20]
        assert meta["dtype"] == "float32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-server && python -m pytest tests/test_array_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement ArrayStore**

```python
# mechestim-server/src/mechestim_server/_array_store.py
"""Server-side array storage with monotonic handle IDs."""

from __future__ import annotations

import numpy as np


class ArrayStore:
    """Maps string handle IDs to numpy arrays. One store per session."""

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._counter: int = 0

    def put(self, arr: np.ndarray) -> str:
        """Store an array, return its handle ID."""
        handle = f"a{self._counter}"
        self._counter += 1
        self._arrays[handle] = arr
        return handle

    def get(self, handle: str) -> np.ndarray:
        """Retrieve an array by handle. Raises KeyError if not found."""
        try:
            return self._arrays[handle]
        except KeyError:
            raise KeyError(f"unknown array handle: {handle!r}") from None

    def metadata(self, handle: str) -> dict:
        """Return handle metadata (id, shape, dtype) without copying data."""
        arr = self.get(handle)
        return {
            "id": handle,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    def free(self, handles: list[str]) -> None:
        """Remove arrays by handle. Silently ignores unknown handles."""
        for h in handles:
            self._arrays.pop(h, None)

    def clear(self) -> None:
        """Remove all arrays."""
        self._arrays.clear()

    @property
    def count(self) -> int:
        """Number of arrays currently stored."""
        return len(self._arrays)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-server && python -m pytest tests/test_array_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-server/src/mechestim_server/_array_store.py mechestim-server/tests/test_array_store.py
git commit -m "feat(server): add ArrayStore with monotonic handle IDs"
```

---

## Task 4: Server CommsTracker

**Files:**
- Create: `mechestim-server/src/mechestim_server/_comms_tracker.py`
- Create: `mechestim-server/tests/test_comms_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-server/tests/test_comms_tracker.py
from mechestim_server._comms_tracker import CommsTracker


class TestCommsTracker:
    def test_initial_state(self):
        tracker = CommsTracker()
        summary = tracker.summary()
        assert summary["request_count"] == 0
        assert summary["fetch_count"] == 0
        assert summary["total_bytes_sent"] == 0
        assert summary["total_bytes_received"] == 0
        assert summary["total_comms_overhead_ns"] == 0
        assert summary["total_compute_time_ns"] == 0

    def test_record_request(self):
        tracker = CommsTracker()
        tracker.record_request(
            bytes_received=100,
            bytes_sent=200,
            comms_overhead_ns=5000,
            compute_time_ns=20000,
            is_fetch=False,
        )
        summary = tracker.summary()
        assert summary["request_count"] == 1
        assert summary["fetch_count"] == 0
        assert summary["total_bytes_received"] == 100
        assert summary["total_bytes_sent"] == 200
        assert summary["total_comms_overhead_ns"] == 5000
        assert summary["total_compute_time_ns"] == 20000

    def test_record_fetch(self):
        tracker = CommsTracker()
        tracker.record_request(
            bytes_received=50,
            bytes_sent=16384,
            comms_overhead_ns=8000,
            compute_time_ns=0,
            is_fetch=True,
        )
        summary = tracker.summary()
        assert summary["request_count"] == 1
        assert summary["fetch_count"] == 1

    def test_accumulates(self):
        tracker = CommsTracker()
        for _ in range(3):
            tracker.record_request(
                bytes_received=100,
                bytes_sent=200,
                comms_overhead_ns=1000,
                compute_time_ns=5000,
                is_fetch=False,
            )
        summary = tracker.summary()
        assert summary["request_count"] == 3
        assert summary["total_bytes_received"] == 300
        assert summary["total_bytes_sent"] == 600
        assert summary["total_comms_overhead_ns"] == 3000
        assert summary["total_compute_time_ns"] == 15000

    def test_overhead_ratio(self):
        tracker = CommsTracker()
        tracker.record_request(
            bytes_received=0,
            bytes_sent=0,
            comms_overhead_ns=2000,
            compute_time_ns=8000,
            is_fetch=False,
        )
        summary = tracker.summary()
        assert summary["overhead_ratio"] == pytest.approx(0.2)

    def test_overhead_ratio_zero_total(self):
        tracker = CommsTracker()
        summary = tracker.summary()
        assert summary["overhead_ratio"] == 0.0


import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-server && python -m pytest tests/test_comms_tracker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CommsTracker**

```python
# mechestim-server/src/mechestim_server/_comms_tracker.py
"""Track communication overhead per session."""

from __future__ import annotations


class CommsTracker:
    """Accumulates per-request timing and byte counts for a session."""

    def __init__(self) -> None:
        self.request_count: int = 0
        self.fetch_count: int = 0
        self.total_bytes_sent: int = 0
        self.total_bytes_received: int = 0
        self.total_comms_overhead_ns: int = 0
        self.total_compute_time_ns: int = 0

    def record_request(
        self,
        *,
        bytes_received: int,
        bytes_sent: int,
        comms_overhead_ns: int,
        compute_time_ns: int,
        is_fetch: bool,
    ) -> None:
        """Record metrics for a single request-response cycle."""
        self.request_count += 1
        if is_fetch:
            self.fetch_count += 1
        self.total_bytes_received += bytes_received
        self.total_bytes_sent += bytes_sent
        self.total_comms_overhead_ns += comms_overhead_ns
        self.total_compute_time_ns += compute_time_ns

    def summary(self) -> dict:
        """Return session-level summary dict."""
        total = self.total_comms_overhead_ns + self.total_compute_time_ns
        return {
            "request_count": self.request_count,
            "fetch_count": self.fetch_count,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_comms_overhead_ns": self.total_comms_overhead_ns,
            "total_compute_time_ns": self.total_compute_time_ns,
            "overhead_ratio": (self.total_comms_overhead_ns / total) if total > 0 else 0.0,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-server && python -m pytest tests/test_comms_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-server/src/mechestim_server/_comms_tracker.py mechestim-server/tests/test_comms_tracker.py
git commit -m "feat(server): add CommsTracker for per-session overhead tracking"
```

---

## Task 5: Server Session

**Files:**
- Create: `mechestim-server/src/mechestim_server/_session.py`
- Create: `mechestim-server/tests/test_session.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-server/tests/test_session.py
import numpy as np
import pytest

from mechestim_server._session import Session


class TestSession:
    def test_create_session(self):
        session = Session(flop_budget=10_000)
        assert session.is_open
        assert session.budget_remaining == 10_000

    def test_store_and_retrieve_array(self):
        session = Session(flop_budget=10_000)
        arr = np.ones((5,), dtype=np.float64)
        handle = session.store_array(arr)
        assert handle == "a0"
        retrieved = session.get_array(handle)
        np.testing.assert_array_equal(retrieved, arr)

    def test_array_metadata(self):
        session = Session(flop_budget=10_000)
        handle = session.store_array(np.zeros((3, 4), dtype=np.float32))
        meta = session.array_metadata(handle)
        assert meta["shape"] == [3, 4]
        assert meta["dtype"] == "float32"

    def test_close_frees_arrays(self):
        session = Session(flop_budget=10_000)
        h = session.store_array(np.zeros(5))
        summary = session.close()
        assert not session.is_open
        assert "budget_summary" in summary
        assert "comms_summary" in summary
        with pytest.raises(KeyError):
            session.get_array(h)

    def test_close_twice_raises(self):
        session = Session(flop_budget=10_000)
        session.close()
        with pytest.raises(RuntimeError, match="already closed"):
            session.close()

    def test_budget_status(self):
        session = Session(flop_budget=10_000)
        status = session.budget_status()
        assert status["flop_budget"] == 10_000
        assert status["flops_used"] == 0
        assert status["flops_remaining"] == 10_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-server && python -m pytest tests/test_session.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Session**

```python
# mechestim-server/src/mechestim_server/_session.py
"""Session: ties together ArrayStore, BudgetContext, and CommsTracker."""

from __future__ import annotations

import numpy as np

import mechestim as me

from mechestim_server._array_store import ArrayStore
from mechestim_server._comms_tracker import CommsTracker


class Session:
    """One session per participant connection. Manages arrays, budget, and comms tracking."""

    def __init__(self, flop_budget: int) -> None:
        self._array_store = ArrayStore()
        self._comms_tracker = CommsTracker()
        self._flop_budget = flop_budget
        self._budget_ctx: me.BudgetContext | None = None
        self._is_open = True

        # Enter the budget context immediately
        self._budget_ctx = me.BudgetContext(flop_budget=flop_budget, quiet=True)
        self._budget_ctx.__enter__()

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def budget_remaining(self) -> int:
        if self._budget_ctx is None:
            return 0
        return self._budget_ctx.flops_remaining

    @property
    def budget_context(self) -> me.BudgetContext:
        """The active BudgetContext for executing operations."""
        if self._budget_ctx is None:
            raise RuntimeError("session is closed")
        return self._budget_ctx

    @property
    def comms_tracker(self) -> CommsTracker:
        return self._comms_tracker

    def store_array(self, arr: np.ndarray) -> str:
        """Store a numpy array, return its handle ID."""
        return self._array_store.put(arr)

    def get_array(self, handle: str) -> np.ndarray:
        """Retrieve a numpy array by handle."""
        return self._array_store.get(handle)

    def array_metadata(self, handle: str) -> dict:
        """Get metadata for an array handle."""
        return self._array_store.metadata(handle)

    def free_arrays(self, handles: list[str]) -> None:
        """Free specific arrays."""
        self._array_store.free(handles)

    def budget_status(self) -> dict:
        """Return current budget state."""
        ctx = self.budget_context
        return {
            "flop_budget": ctx.flop_budget,
            "flops_used": ctx.flops_used,
            "flops_remaining": ctx.flops_remaining,
        }

    def close(self) -> dict:
        """Close the session: exit budget context, free all arrays, return summary."""
        if not self._is_open:
            raise RuntimeError("session already closed")

        self._is_open = False

        # Exit the budget context
        budget_summary = self._budget_ctx.summary()
        self._budget_ctx.__exit__(None, None, None)
        self._budget_ctx = None

        # Free all arrays
        self._array_store.clear()

        return {
            "budget_summary": budget_summary,
            "comms_summary": self._comms_tracker.summary(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-server && python -m pytest tests/test_session.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-server/src/mechestim_server/_session.py mechestim-server/tests/test_session.py
git commit -m "feat(server): add Session tying ArrayStore, BudgetContext, and CommsTracker"
```

---

## Task 6: Server Request Handler

The request handler dispatches decoded messages to real mechestim functions.

**Files:**
- Create: `mechestim-server/src/mechestim_server/_request_handler.py`
- Create: `mechestim-server/tests/test_request_handler.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-server/tests/test_request_handler.py
import numpy as np
import pytest

from mechestim_server._request_handler import RequestHandler
from mechestim_server._session import Session


class TestRequestHandler:
    def setup_method(self):
        self.session = Session(flop_budget=1_000_000)
        self.handler = RequestHandler(self.session)

    def teardown_method(self):
        if self.session.is_open:
            self.session.close()

    def test_handle_zeros(self):
        result = self.handler.handle({"op": "zeros", "args": [[3, 4]]})
        assert result["status"] == "ok"
        handle = result["result"]["id"]
        assert result["result"]["shape"] == [3, 4]
        assert result["result"]["dtype"] == "float64"
        arr = self.session.get_array(handle)
        np.testing.assert_array_equal(arr, np.zeros((3, 4)))

    def test_handle_ones(self):
        result = self.handler.handle({"op": "ones", "args": [[2, 2]]})
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        np.testing.assert_array_equal(arr, np.ones((2, 2)))

    def test_handle_unary_exp(self):
        # First create an array
        h_in = self.session.store_array(np.array([1.0, 2.0, 3.0]))
        result = self.handler.handle({"op": "exp", "args": [h_in]})
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        np.testing.assert_allclose(arr, np.exp([1.0, 2.0, 3.0]))

    def test_handle_binary_add(self):
        h1 = self.session.store_array(np.array([1.0, 2.0]))
        h2 = self.session.store_array(np.array([3.0, 4.0]))
        result = self.handler.handle({"op": "add", "args": [h1, h2]})
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        np.testing.assert_array_equal(arr, [4.0, 6.0])

    def test_handle_binary_with_scalar(self):
        h = self.session.store_array(np.array([1.0, 2.0]))
        result = self.handler.handle({"op": "multiply", "args": [h, 3.0]})
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        np.testing.assert_array_equal(arr, [3.0, 6.0])

    def test_handle_reduction_sum(self):
        h = self.session.store_array(np.array([1.0, 2.0, 3.0]))
        result = self.handler.handle({"op": "sum", "args": [h]})
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        assert float(arr) == 6.0

    def test_handle_einsum(self):
        h1 = self.session.store_array(np.eye(3, dtype=np.float64))
        h2 = self.session.store_array(np.ones((3,), dtype=np.float64))
        result = self.handler.handle({"op": "einsum", "args": ["ij,j->i", h1, h2]})
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        np.testing.assert_array_equal(arr, [1.0, 1.0, 1.0])

    def test_handle_create_from_data(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64).tobytes()
        result = self.handler.handle({
            "op": "create_from_data",
            "data": data,
            "shape": [3],
            "dtype": "float64",
        })
        assert result["status"] == "ok"
        arr = self.session.get_array(result["result"]["id"])
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_handle_fetch(self):
        h = self.session.store_array(np.array([1.0, 2.0], dtype=np.float64))
        result = self.handler.handle({"op": "fetch", "id": h})
        assert result["status"] == "ok"
        assert result["shape"] == [2]
        assert result["dtype"] == "float64"
        assert isinstance(result["data"], bytes)
        assert len(result["data"]) == 16  # 2 * 8 bytes

    def test_handle_free(self):
        h = self.session.store_array(np.zeros(5))
        result = self.handler.handle({"op": "free", "ids": [h]})
        assert result["status"] == "ok"
        with pytest.raises(KeyError):
            self.session.get_array(h)

    def test_handle_budget_status(self):
        result = self.handler.handle({"op": "budget_status"})
        assert result["status"] == "ok"
        assert result["result"]["flop_budget"] == 1_000_000

    def test_handle_unknown_handle_returns_error(self):
        result = self.handler.handle({"op": "exp", "args": ["nonexistent"]})
        assert result["status"] == "error"
        assert "unknown array handle" in result["message"]

    def test_handle_budget_exhausted_returns_error(self):
        session = Session(flop_budget=1)
        handler = RequestHandler(session)
        h = session.store_array(np.ones(1000))
        result = handler.handle({"op": "exp", "args": [h]})
        assert result["status"] == "error"
        assert result["error_type"] == "BudgetExhaustedError"
        session.close()

    def test_budget_included_in_response(self):
        h = self.session.store_array(np.ones(10))
        result = self.handler.handle({"op": "exp", "args": [h]})
        assert "budget" in result
        assert result["budget"]["flops_used"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-server && python -m pytest tests/test_request_handler.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RequestHandler**

```python
# mechestim-server/src/mechestim_server/_request_handler.py
"""Dispatch incoming requests to mechestim functions."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import mechestim as me
from mechestim.errors import BudgetExhaustedError, NoBudgetContextError, SymmetryError

from mechestim_server._session import Session


class RequestHandler:
    """Handles a single decoded request dict, returns a response dict."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def handle(self, request: dict) -> dict:
        """Dispatch a request and return a response dict (not yet msgpack-encoded)."""
        op = request.get("op", "")
        try:
            if op == "fetch":
                return self._handle_fetch(request)
            elif op == "fetch_slice":
                return self._handle_fetch_slice(request)
            elif op == "free":
                return self._handle_free(request)
            elif op == "budget_status":
                return self._handle_budget_status()
            elif op == "create_from_data":
                return self._handle_create_from_data(request)
            else:
                return self._handle_op(request)
        except BudgetExhaustedError as e:
            return {"status": "error", "error_type": "BudgetExhaustedError", "message": str(e)}
        except NoBudgetContextError as e:
            return {"status": "error", "error_type": "NoBudgetContextError", "message": str(e)}
        except SymmetryError as e:
            return {"status": "error", "error_type": "SymmetryError", "message": str(e)}
        except (ValueError, TypeError) as e:
            return {"status": "error", "error_type": type(e).__name__, "message": str(e)}
        except KeyError as e:
            return {"status": "error", "error_type": "KeyError", "message": str(e)}
        except Exception as e:
            # Catch-all: sanitize message to avoid leaking numpy internals
            return {"status": "error", "error_type": "MechEstimServerError", "message": f"internal server error: {type(e).__name__}"}

    def _resolve_args(self, args: list) -> list:
        """Resolve handle IDs to numpy arrays. Non-string args pass through."""
        resolved = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith("a") and arg[1:].isdigit():
                resolved.append(self._session.get_array(arg))
            else:
                resolved.append(arg)
        return resolved

    def _handle_op(self, request: dict) -> dict:
        """Execute a mechestim function by name."""
        op = request["op"]
        raw_args = request.get("args", [])
        kwargs = request.get("kwargs", {})

        # Resolve handle references to actual numpy arrays
        args = self._resolve_args(raw_args)

        # Look up the mechestim function
        func = _get_mechestim_func(op)

        # Execute
        result = func(*args, **kwargs)

        # Store result and return metadata
        if isinstance(result, np.ndarray):
            handle = self._session.store_array(result)
            meta = self._session.array_metadata(handle)
        elif isinstance(result, tuple):
            # Multi-output ops (e.g., modf, linalg.svd)
            handles = []
            for r in result:
                if isinstance(r, np.ndarray):
                    handles.append(self._session.array_metadata(self._session.store_array(r)))
                else:
                    handles.append(r)
            meta = {"multi": handles}
        else:
            # Scalar result
            meta = {"value": result}

        budget = self._session.budget_status()
        return {"status": "ok", "result": meta, "budget": budget}

    def _handle_create_from_data(self, request: dict) -> dict:
        """Create an array from raw bytes sent by the client."""
        data = request["data"]
        shape = tuple(request["shape"])
        dtype = np.dtype(request["dtype"])
        arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        handle = self._session.store_array(arr)
        meta = self._session.array_metadata(handle)
        budget = self._session.budget_status()
        return {"status": "ok", "result": meta, "budget": budget}

    def _handle_fetch(self, request: dict) -> dict:
        """Return raw array bytes to the client."""
        handle = request["id"]
        arr = self._session.get_array(handle)
        return {
            "status": "ok",
            "data": arr.tobytes(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    def _handle_fetch_slice(self, request: dict) -> dict:
        """Return a slice of an array."""
        handle = request["id"]
        arr = self._session.get_array(handle)
        slices = tuple(slice(*s) if isinstance(s, list) else s for s in request["slices"])
        sliced = arr[slices]
        if isinstance(sliced, np.ndarray):
            return {
                "status": "ok",
                "data": sliced.tobytes(),
                "shape": list(sliced.shape),
                "dtype": str(sliced.dtype),
            }
        else:
            # Scalar indexing
            return {"status": "ok", "data": None, "value": float(sliced)}

    def _handle_free(self, request: dict) -> dict:
        """Free arrays by handle."""
        self._session.free_arrays(request["ids"])
        return {"status": "ok"}

    def _handle_budget_status(self) -> dict:
        """Return current budget state."""
        return {"status": "ok", "result": self._session.budget_status()}


def _get_mechestim_func(op_name: str):
    """Look up a mechestim function by name, supporting dotted paths (e.g., 'linalg.svd')."""
    parts = op_name.split(".")
    obj = me
    for part in parts:
        obj = getattr(obj, part)
    return obj
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-server && python -m pytest tests/test_request_handler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-server/src/mechestim_server/_request_handler.py mechestim-server/tests/test_request_handler.py
git commit -m "feat(server): add RequestHandler dispatching requests to mechestim functions"
```

---

## Task 7: Server ZMQ Loop

**Files:**
- Create: `mechestim-server/src/mechestim_server/_server.py`
- Create: `mechestim-server/src/mechestim_server/__main__.py`
- Create: `mechestim-server/tests/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-server/tests/test_server.py
"""Integration test: start a real server, connect a ZMQ client, send requests."""
import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

from mechestim_server._server import MechestimServer


@pytest.fixture
def server_url():
    return "tcp://127.0.0.1:15555"


@pytest.fixture
def server(server_url):
    srv = MechestimServer(url=server_url, session_timeout_s=5)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    time.sleep(0.1)  # Wait for server to bind
    yield srv
    srv.stop()
    thread.join(timeout=2)


@pytest.fixture
def client(server_url, server):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(server_url)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    yield sock
    sock.close()
    ctx.term()


def send_recv(sock, msg):
    sock.send(msgpack.packb(msg))
    return msgpack.unpackb(sock.recv())


class TestServer:
    def test_budget_open_and_status(self, client):
        resp = send_recv(client, {"op": "budget_open", "flop_budget": 10_000})
        assert resp[b"status"] == b"ok"

        resp = send_recv(client, {"op": "budget_status"})
        assert resp[b"status"] == b"ok"
        assert resp[b"result"][b"flop_budget"] == 10_000

    def test_create_and_fetch(self, client):
        send_recv(client, {"op": "budget_open", "flop_budget": 1_000_000})

        resp = send_recv(client, {"op": "zeros", "args": [[3, 4]]})
        assert resp[b"status"] == b"ok"
        handle = resp[b"result"][b"id"]

        resp = send_recv(client, {"op": "fetch", "id": handle})
        assert resp[b"status"] == b"ok"
        data = resp[b"data"]
        arr = np.frombuffer(data, dtype=np.float64).reshape(3, 4)
        np.testing.assert_array_equal(arr, np.zeros((3, 4)))

    def test_operation_chain(self, client):
        send_recv(client, {"op": "budget_open", "flop_budget": 1_000_000})

        r1 = send_recv(client, {"op": "ones", "args": [[5]]})
        h1 = r1[b"result"][b"id"]

        r2 = send_recv(client, {"op": "exp", "args": [h1]})
        h2 = r2[b"result"][b"id"]

        r3 = send_recv(client, {"op": "fetch", "id": h2})
        arr = np.frombuffer(r3[b"data"], dtype=np.float64)
        np.testing.assert_allclose(arr, np.exp(np.ones(5)))

    def test_budget_close_returns_summary(self, client):
        send_recv(client, {"op": "budget_open", "flop_budget": 1_000_000})
        r1 = send_recv(client, {"op": "ones", "args": [[10]]})
        send_recv(client, {"op": "exp", "args": [r1[b"result"][b"id"]]})

        resp = send_recv(client, {"op": "budget_close"})
        assert resp[b"status"] == b"ok"
        assert b"budget_summary" in resp[b"result"]
        assert b"comms_summary" in resp[b"result"]

    def test_error_no_session(self, client):
        resp = send_recv(client, {"op": "zeros", "args": [[3]]})
        assert resp[b"status"] == b"error"

    def test_invalid_op(self, client):
        send_recv(client, {"op": "budget_open", "flop_budget": 1_000})
        resp = send_recv(client, {"op": "hack_the_planet", "args": []})
        assert resp[b"status"] == b"error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-server && python -m pytest tests/test_server.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MechestimServer**

```python
# mechestim-server/src/mechestim_server/_server.py
"""ZMQ REP server loop with session management."""

from __future__ import annotations

import time

import msgpack
import zmq

from mechestim_server._protocol import (
    InvalidRequestError,
    decode_request,
    encode_error_response,
    encode_response,
    validate_request,
)
from mechestim_server._request_handler import RequestHandler
from mechestim_server._session import Session


class MechestimServer:
    """ZeroMQ REP server that handles mechestim operations."""

    def __init__(self, url: str = "ipc:///tmp/mechestim.sock", session_timeout_s: float = 60.0) -> None:
        self._url = url
        self._session_timeout_s = session_timeout_s
        self._session: Session | None = None
        self._handler: RequestHandler | None = None
        self._running = False

    def run(self) -> None:
        """Start the server loop. Blocks until stop() is called."""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(self._url)
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s poll interval for clean shutdown
        self._running = True

        try:
            while self._running:
                try:
                    raw = socket.recv()
                except zmq.Again:
                    # Timeout — check for session reaping
                    self._check_session_timeout()
                    continue

                response = self._process_request(raw)
                socket.send(response)
        finally:
            socket.close()
            ctx.term()

    def stop(self) -> None:
        """Signal the server to stop."""
        self._running = False

    def _process_request(self, raw: bytes) -> bytes:
        """Decode, validate, dispatch, encode. Returns msgpack bytes."""
        t0 = time.perf_counter_ns()

        try:
            request = decode_request(raw)
        except InvalidRequestError as e:
            return encode_error_response("InvalidRequestError", str(e))

        t1 = time.perf_counter_ns()

        op = request.get("op", "")

        # Handle session lifecycle ops
        if op == "budget_open":
            return self._handle_budget_open(request, deser_ns=t1 - t0)
        elif op == "budget_close":
            return self._handle_budget_close(deser_ns=t1 - t0)

        # All other ops require an active session
        if self._session is None or not self._session.is_open:
            return encode_error_response("NoBudgetContextError", "No active session. Send budget_open first.")

        try:
            validate_request(request)
        except InvalidRequestError as e:
            return encode_error_response("InvalidRequestError", str(e))

        # Dispatch to request handler
        result = self._handler.handle(request)

        t2 = time.perf_counter_ns()

        # Encode response
        response = msgpack.packb(result)

        t3 = time.perf_counter_ns()

        # Track comms overhead
        comms_ns = (t1 - t0) + (t3 - t2)
        compute_ns = t2 - t1
        is_fetch = op in ("fetch", "fetch_slice")
        self._session.comms_tracker.record_request(
            bytes_received=len(raw),
            bytes_sent=len(response),
            comms_overhead_ns=comms_ns,
            compute_time_ns=compute_ns,
            is_fetch=is_fetch,
        )

        return response

    def _handle_budget_open(self, request: dict, deser_ns: int) -> bytes:
        """Open a new session."""
        if self._session is not None and self._session.is_open:
            return encode_error_response("RuntimeError", "A session is already open. Send budget_close first.")

        flop_budget = request.get("flop_budget", 0)
        self._session = Session(flop_budget=flop_budget)
        self._handler = RequestHandler(self._session)
        self._last_activity = time.monotonic()

        return encode_response(
            result={"session": "opened"},
            budget=self._session.budget_status(),
            comms_overhead_ns=deser_ns,
        )

    def _handle_budget_close(self, deser_ns: int) -> bytes:
        """Close the active session and return summary."""
        if self._session is None or not self._session.is_open:
            return encode_error_response("RuntimeError", "No active session to close.")

        summary = self._session.close()
        self._session = None
        self._handler = None

        return msgpack.packb({"status": "ok", "result": summary})

    def _check_session_timeout(self) -> None:
        """Reap the session if it has been inactive too long."""
        if self._session is not None and self._session.is_open:
            if hasattr(self, "_last_activity"):
                if time.monotonic() - self._last_activity > self._session_timeout_s:
                    self._session.close()
                    self._session = None
                    self._handler = None
```

- [ ] **Step 4: Implement __main__.py entry point**

```python
# mechestim-server/src/mechestim_server/__main__.py
"""Entry point: python -m mechestim_server"""

from __future__ import annotations

import argparse
import sys

from mechestim_server._server import MechestimServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechestim backend server")
    parser.add_argument(
        "--url",
        default="ipc:///tmp/mechestim.sock",
        help="ZeroMQ bind URL (default: ipc:///tmp/mechestim.sock)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Session inactivity timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    print(f"mechestim-server starting on {args.url}", file=sys.stderr)
    server = MechestimServer(url=args.url, session_timeout_s=args.timeout)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nmechestim-server shutting down", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mechestim-server && python -m pytest tests/test_server.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add mechestim-server/src/mechestim_server/_server.py mechestim-server/src/mechestim_server/__main__.py mechestim-server/tests/test_server.py
git commit -m "feat(server): add ZMQ server loop with session lifecycle and timeout reaping"
```

---

## Task 8: Client Errors Module

**Files:**
- Create: `mechestim-client/src/mechestim/errors.py`
- Create: `mechestim-client/tests/test_errors.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-client/tests/test_errors.py
from mechestim.errors import (
    MechEstimError,
    BudgetExhaustedError,
    NoBudgetContextError,
    SymmetryError,
    MechEstimWarning,
    MechEstimServerError,
    raise_from_response,
)


class TestErrors:
    def test_budget_exhausted_is_mechestim_error(self):
        err = BudgetExhaustedError("exp", flop_cost=100, flops_remaining=50)
        assert isinstance(err, MechEstimError)
        assert "exp" in str(err)

    def test_no_budget_context(self):
        err = NoBudgetContextError()
        assert "BudgetContext" in str(err)

    def test_symmetry_error(self):
        err = SymmetryError((0, 1), 0.01)
        assert isinstance(err, MechEstimError)

    def test_server_error(self):
        err = MechEstimServerError("internal error")
        assert isinstance(err, MechEstimError)

    def test_warning_is_user_warning(self):
        assert issubclass(MechEstimWarning, UserWarning)


class TestRaiseFromResponse:
    def test_raises_budget_exhausted(self):
        import pytest
        with pytest.raises(BudgetExhaustedError, match="some message"):
            raise_from_response("BudgetExhaustedError", "some message")

    def test_raises_no_budget_context(self):
        import pytest
        with pytest.raises(NoBudgetContextError):
            raise_from_response("NoBudgetContextError", "No active BudgetContext")

    def test_raises_value_error(self):
        import pytest
        with pytest.raises(ValueError, match="bad value"):
            raise_from_response("ValueError", "bad value")

    def test_raises_server_error_for_unknown(self):
        import pytest
        with pytest.raises(MechEstimServerError, match="weird"):
            raise_from_response("WeirdError", "weird")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-client && python -m pytest tests/test_errors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement client errors**

```python
# mechestim-client/src/mechestim/errors.py
"""Exception classes mirroring real mechestim errors, plus raise_from_response."""

from __future__ import annotations


class MechEstimError(Exception):
    """Base exception for mechestim."""


class BudgetExhaustedError(MechEstimError):
    """Raised when an operation would exceed the FLOP budget."""

    def __init__(self, op_name: str = "", *, flop_cost: int = 0, flops_remaining: int = 0, message: str = ""):
        if message:
            super().__init__(message)
        else:
            super().__init__(
                f"{op_name} would cost {flop_cost:,} FLOPs but only {flops_remaining:,} remain"
            )


class NoBudgetContextError(MechEstimError):
    """Raised when a counted operation is used outside a BudgetContext."""

    def __init__(self, message: str = ""):
        super().__init__(message or "No active BudgetContext. Wrap your code in `with mechestim.BudgetContext(...):` ")


class SymmetryError(MechEstimError):
    """Raised when a symmetry claim is violated."""

    def __init__(self, dims: tuple[int, ...] = (), max_deviation: float = 0.0, message: str = ""):
        if message:
            super().__init__(message)
        else:
            super().__init__(
                f"Tensor not symmetric along dims {dims}: max deviation = {max_deviation}"
            )


class MechEstimWarning(UserWarning):
    """Warning for NaN/Inf results."""


class MechEstimServerError(MechEstimError):
    """Raised for server-side errors that don't map to a specific mechestim exception."""


# Map error type names to exception classes
_ERROR_MAP: dict[str, type] = {
    "BudgetExhaustedError": BudgetExhaustedError,
    "NoBudgetContextError": NoBudgetContextError,
    "SymmetryError": SymmetryError,
    "MechEstimServerError": MechEstimServerError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "InvalidRequestError": MechEstimServerError,
}


def raise_from_response(error_type: str, message: str) -> None:
    """Re-raise a server error on the client side."""
    exc_class = _ERROR_MAP.get(error_type, MechEstimServerError)
    if exc_class in (BudgetExhaustedError, NoBudgetContextError, SymmetryError):
        raise exc_class(message=message)
    raise exc_class(message)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-client && python -m pytest tests/test_errors.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-client/src/mechestim/errors.py mechestim-client/tests/test_errors.py
git commit -m "feat(client): add error classes mirroring real mechestim + raise_from_response"
```

---

## Task 9: Client Connection Layer

**Files:**
- Create: `mechestim-client/src/mechestim/_connection.py`
- Create: `mechestim-client/src/mechestim/_protocol.py`
- Create: `mechestim-client/tests/test_connection.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-client/tests/test_connection.py
import os
import pytest
from unittest.mock import patch, MagicMock

from mechestim._connection import get_connection, Connection


class TestConnectionConfig:
    def test_default_url(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MECHESTIM_SERVER_URL", None)
            conn = Connection()
            assert conn.url == "ipc:///tmp/mechestim.sock"

    def test_url_from_env(self):
        with patch.dict(os.environ, {"MECHESTIM_SERVER_URL": "tcp://10.0.0.5:5555"}):
            conn = Connection()
            assert conn.url == "tcp://10.0.0.5:5555"


class TestProtocol:
    def test_encode_op(self):
        from mechestim._protocol import encode_request
        raw = encode_request("add", args=["a1", "a2"])
        import msgpack
        msg = msgpack.unpackb(raw)
        assert msg[b"op"] == b"add"

    def test_encode_with_kwargs(self):
        from mechestim._protocol import encode_request
        raw = encode_request("zeros", args=[[3, 4]], kwargs={"dtype": "float32"})
        import msgpack
        msg = msgpack.unpackb(raw)
        assert msg[b"kwargs"][b"dtype"] == b"float32"

    def test_encode_create_from_data(self):
        from mechestim._protocol import encode_create_from_data
        raw = encode_create_from_data(b"\x00" * 24, [3], "float64")
        import msgpack
        msg = msgpack.unpackb(raw)
        assert msg[b"op"] == b"create_from_data"
        assert msg[b"shape"] == [3]

    def test_encode_budget_open(self):
        from mechestim._protocol import encode_budget_open
        raw = encode_budget_open(10_000)
        import msgpack
        msg = msgpack.unpackb(raw)
        assert msg[b"op"] == b"budget_open"
        assert msg[b"flop_budget"] == 10_000

    def test_decode_response(self):
        import msgpack
        from mechestim._protocol import decode_response
        raw = msgpack.packb({"status": "ok", "result": {"id": "a0", "shape": [3], "dtype": "float64"}})
        resp = decode_response(raw)
        assert resp["status"] == "ok"
        assert resp["result"]["id"] == "a0"

    def test_decode_error_response(self):
        import msgpack
        from mechestim._protocol import decode_response
        raw = msgpack.packb({"status": "error", "error_type": "BudgetExhaustedError", "message": "budget gone"})
        resp = decode_response(raw)
        assert resp["status"] == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-client && python -m pytest tests/test_connection.py -v`
Expected: FAIL

- [ ] **Step 3: Implement client protocol**

```python
# mechestim-client/src/mechestim/_protocol.py
"""Client-side message encoding/decoding."""

from __future__ import annotations

import msgpack


def encode_request(op: str, args: list | None = None, kwargs: dict | None = None) -> bytes:
    """Encode an operation request."""
    msg = {"op": op}
    if args is not None:
        msg["args"] = args
    if kwargs:
        msg["kwargs"] = kwargs
    return msgpack.packb(msg)


def encode_create_from_data(data: bytes, shape: list[int], dtype: str) -> bytes:
    """Encode a create_from_data request."""
    return msgpack.packb({
        "op": "create_from_data",
        "data": data,
        "shape": shape,
        "dtype": dtype,
    })


def encode_budget_open(flop_budget: int) -> bytes:
    """Encode a budget_open request."""
    return msgpack.packb({"op": "budget_open", "flop_budget": flop_budget})


def encode_budget_close() -> bytes:
    """Encode a budget_close request."""
    return msgpack.packb({"op": "budget_close"})


def encode_budget_status() -> bytes:
    """Encode a budget_status request."""
    return msgpack.packb({"op": "budget_status"})


def encode_fetch(handle_id: str) -> bytes:
    """Encode a fetch request."""
    return msgpack.packb({"op": "fetch", "id": handle_id})


def encode_free(handles: list[str]) -> bytes:
    """Encode a free request."""
    return msgpack.packb({"op": "free", "ids": handles})


def decode_response(raw: bytes) -> dict:
    """Decode a msgpack response, normalizing bytes keys to strings."""
    msg = msgpack.unpackb(raw)
    return _normalize_keys(msg)


def _normalize_keys(obj):
    """Recursively convert bytes keys/values to strings where appropriate."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): _normalize_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    elif isinstance(obj, bytes):
        # Try to decode as string; if it looks like binary data (array bytes), keep as bytes
        try:
            decoded = obj.decode("utf-8")
            # Heuristic: if it decoded cleanly and is short, it's probably a string
            return decoded
        except UnicodeDecodeError:
            return obj
    return obj
```

- [ ] **Step 4: Implement Connection**

```python
# mechestim-client/src/mechestim/_connection.py
"""ZMQ client connection to mechestim server."""

from __future__ import annotations

import os
import time

import zmq

from mechestim._protocol import decode_response
from mechestim.errors import raise_from_response, MechEstimServerError

_DEFAULT_URL = "ipc:///tmp/mechestim.sock"
_DEFAULT_TIMEOUT_MS = 30_000

# Module-level singleton
_connection: Connection | None = None


class Connection:
    """ZMQ REQ socket wrapper with auto-connect."""

    def __init__(self, url: str | None = None, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> None:
        self.url = url or os.environ.get("MECHESTIM_SERVER_URL", _DEFAULT_URL)
        self._timeout_ms = timeout_ms
        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def _ensure_connected(self) -> zmq.Socket:
        if self._socket is None:
            self._ctx = zmq.Context()
            self._socket = self._ctx.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.connect(self.url)
        return self._socket

    def send_recv(self, raw_request: bytes) -> dict:
        """Send a raw msgpack request, receive and decode the response.

        Returns the decoded response dict. Raises on error responses.
        """
        sock = self._ensure_connected()
        t_start = time.perf_counter_ns()

        try:
            sock.send(raw_request)
            raw_response = sock.recv()
        except zmq.Again:
            raise MechEstimServerError("server timeout: no response within timeout period")

        t_end = time.perf_counter_ns()

        resp = decode_response(raw_response)

        # Attach client-side round-trip timing
        resp["_round_trip_ns"] = t_end - t_start
        resp["_response_bytes"] = len(raw_response)
        resp["_request_bytes"] = len(raw_request)

        # Auto-raise on error responses
        if resp.get("status") == "error":
            raise_from_response(resp.get("error_type", "MechEstimServerError"), resp.get("message", "unknown error"))

        return resp

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None


def get_connection() -> Connection:
    """Get or create the module-level connection singleton."""
    global _connection
    if _connection is None:
        _connection = Connection()
    return _connection


def reset_connection() -> None:
    """Close and reset the connection singleton (for testing)."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mechestim-client && python -m pytest tests/test_connection.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add mechestim-client/src/mechestim/_connection.py mechestim-client/src/mechestim/_protocol.py mechestim-client/tests/test_connection.py
git commit -m "feat(client): add ZMQ connection layer and wire protocol encoding"
```

---

## Task 10: Client RemoteArray and RemoteScalar

**Files:**
- Create: `mechestim-client/src/mechestim/_remote_array.py`
- Create: `mechestim-client/tests/test_remote_array.py`

This is the largest task — the transparent proxy that makes the entire system work.

- [ ] **Step 1: Write failing tests**

```python
# mechestim-client/tests/test_remote_array.py
"""Tests for RemoteArray require a running server. Use the server fixture."""
import struct
import threading
import time

import msgpack
import pytest
import zmq

from mechestim._remote_array import RemoteArray, RemoteScalar


class TestRemoteArrayMetadata:
    """Test metadata access (no server needed — cached locally)."""

    def test_shape(self):
        arr = RemoteArray(handle_id="a0", shape=(3, 4), dtype="float64")
        assert arr.shape == (3, 4)

    def test_dtype(self):
        arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float32")
        assert arr.dtype == "float32"

    def test_ndim(self):
        arr = RemoteArray(handle_id="a0", shape=(3, 4, 5), dtype="float64")
        assert arr.ndim == 3

    def test_size(self):
        arr = RemoteArray(handle_id="a0", shape=(3, 4), dtype="float64")
        assert arr.size == 12

    def test_nbytes(self):
        arr = RemoteArray(handle_id="a0", shape=(3,), dtype="float64")
        assert arr.nbytes == 24  # 3 * 8

    def test_len(self):
        arr = RemoteArray(handle_id="a0", shape=(5, 3), dtype="float64")
        assert len(arr) == 5

    def test_len_scalar_raises(self):
        arr = RemoteArray(handle_id="a0", shape=(), dtype="float64")
        with pytest.raises(TypeError):
            len(arr)


class TestRemoteScalar:
    def test_float_conversion(self):
        s = RemoteScalar(value=3.14, dtype="float64")
        assert float(s) == pytest.approx(3.14)

    def test_int_conversion(self):
        s = RemoteScalar(value=42, dtype="int64")
        assert int(s) == 42

    def test_bool_conversion(self):
        assert bool(RemoteScalar(value=1.0, dtype="float64")) is True
        assert bool(RemoteScalar(value=0.0, dtype="float64")) is False

    def test_repr(self):
        s = RemoteScalar(value=3.14, dtype="float64")
        assert "3.14" in repr(s)

    def test_arithmetic(self):
        s = RemoteScalar(value=5.0, dtype="float64")
        assert float(s) + 1 == 6.0

    def test_comparison(self):
        s = RemoteScalar(value=5.0, dtype="float64")
        assert s > 3.0
        assert s < 10.0
        assert s == 5.0

    def test_shape_is_empty_tuple(self):
        s = RemoteScalar(value=5.0, dtype="float64")
        assert s.shape == ()

    def test_ndim_is_zero(self):
        s = RemoteScalar(value=5.0, dtype="float64")
        assert s.ndim == 0

    def test_tolist(self):
        s = RemoteScalar(value=5.0, dtype="float64")
        assert s.tolist() == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-client && python -m pytest tests/test_remote_array.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RemoteArray and RemoteScalar**

```python
# mechestim-client/src/mechestim/_remote_array.py
"""Transparent proxy classes for server-side arrays."""

from __future__ import annotations

import struct
from typing import Any

# dtype name → (struct format char, byte size)
_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "float16": ("e", 2),
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int8": ("b", 1),
    "int16": ("h", 2),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint8": ("B", 1),
    "bool": ("?", 1),
    "bool_": ("?", 1),
    "complex64": ("", 8),   # 2 * float32
    "complex128": ("", 16),  # 2 * float64
}


def _dtype_itemsize(dtype: str) -> int:
    """Get byte size for a dtype string."""
    info = _DTYPE_INFO.get(dtype)
    if info:
        return info[1]
    # Fallback: assume 8 bytes
    return 8


def _bytes_to_list(data: bytes, shape: tuple[int, ...], dtype: str) -> Any:
    """Convert raw bytes to a nested Python list matching the given shape.

    Works without numpy — uses struct.unpack.
    """
    info = _DTYPE_INFO.get(dtype)
    if info is None or info[0] == "":
        # For complex or unknown types, return flat list of raw values
        itemsize = _dtype_itemsize(dtype)
        count = len(data) // itemsize
        # Best effort: treat as float64
        values = list(struct.unpack(f"<{count}d", data))
    else:
        fmt_char = info[0]
        itemsize = info[1]
        count = len(data) // itemsize
        values = list(struct.unpack(f"<{count}{fmt_char}", data))

    if len(shape) == 0:
        return values[0] if values else 0
    elif len(shape) == 1:
        return values
    else:
        # Reshape into nested lists
        return _reshape_flat(values, shape)


def _reshape_flat(flat: list, shape: tuple[int, ...]) -> list:
    """Reshape a flat list into nested lists of given shape."""
    if len(shape) == 1:
        return flat[:shape[0]]
    stride = 1
    for dim in shape[1:]:
        stride *= dim
    return [_reshape_flat(flat[i * stride:(i + 1) * stride], shape[1:]) for i in range(shape[0])]


class RemoteScalar:
    """A scalar value returned from a server operation. Behaves like a Python number."""

    def __init__(self, value: int | float, dtype: str) -> None:
        self._value = value
        self._dtype = dtype

    @property
    def shape(self) -> tuple:
        return ()

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def handle_id(self) -> str | None:
        return None

    def tolist(self) -> int | float:
        return self._value

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return repr(self._value)

    def __str__(self) -> str:
        return str(self._value)

    # Comparison operators
    def __eq__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value == other._value
        return self._value == other

    def __lt__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return self._value < other_val

    def __le__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return self._value <= other_val

    def __gt__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return self._value > other_val

    def __ge__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, RemoteScalar) else other
        return self._value >= other_val

    def __hash__(self) -> int:
        return hash(self._value)


class RemoteArray:
    """Transparent proxy for a server-side numpy array.

    Metadata (shape, dtype) is cached locally. Data is fetched on demand
    when the participant reads values (print, indexing, tolist, etc.).
    """

    def __init__(self, handle_id: str, shape: tuple[int, ...], dtype: str) -> None:
        self._handle_id = handle_id
        self._shape = tuple(shape) if not isinstance(shape, tuple) else shape
        self._dtype = dtype

    @property
    def handle_id(self) -> str:
        return self._handle_id

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        return self.size * _dtype_itemsize(self._dtype)

    def __len__(self) -> int:
        if len(self._shape) == 0:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    # --- Data access (auto-fetch from server) ---

    def _fetch_data(self) -> tuple[bytes, tuple[int, ...], str]:
        """Fetch raw array data from the server."""
        from mechestim._connection import get_connection
        from mechestim._protocol import encode_fetch

        conn = get_connection()
        resp = conn.send_recv(encode_fetch(self._handle_id))
        return resp["data"], tuple(resp["shape"]), resp["dtype"]

    def tolist(self) -> Any:
        """Fetch array data and convert to a Python list."""
        data, shape, dtype = self._fetch_data()
        return _bytes_to_list(data, shape, dtype)

    def __repr__(self) -> str:
        """Fetch data and format like numpy."""
        try:
            data = self.tolist()
            return f"array({data})"
        except Exception:
            return f"RemoteArray(id={self._handle_id!r}, shape={self._shape}, dtype={self._dtype!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def __float__(self) -> float:
        """Convert 0-d or single-element array to float."""
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        data, shape, dtype = self._fetch_data()
        return float(_bytes_to_list(data, shape, dtype))

    def __int__(self) -> int:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        data, shape, dtype = self._fetch_data()
        return int(_bytes_to_list(data, shape, dtype))

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError("truth value of array with more than one element is ambiguous")
        data, shape, dtype = self._fetch_data()
        return bool(_bytes_to_list(data, shape, dtype))

    def __iter__(self):
        """Fetch data and iterate."""
        lst = self.tolist()
        return iter(lst)

    def __getitem__(self, key):
        """Fetch slice or element from server."""
        from mechestim._connection import get_connection
        from mechestim._protocol import encode_request

        # For simple integer/slice indexing, use fetch_slice
        # For now, fetch full array and index locally
        data, shape, dtype = self._fetch_data()
        full = _bytes_to_list(data, shape, dtype)
        result = full[key] if not isinstance(full, (int, float)) else full
        return result

    # --- Operator overloads (dispatch to server) ---

    def _dispatch_op(self, op_name: str, *args):
        """Send an operation to the server and return a new RemoteArray or RemoteScalar."""
        from mechestim._connection import get_connection
        from mechestim._protocol import encode_request

        # Encode args: RemoteArray → handle_id, RemoteScalar → value, else pass through
        encoded_args = []
        for arg in args:
            if isinstance(arg, RemoteArray):
                encoded_args.append(arg.handle_id)
            elif isinstance(arg, RemoteScalar):
                encoded_args.append(arg._value)
            else:
                encoded_args.append(arg)

        conn = get_connection()
        resp = conn.send_recv(encode_request(op_name, args=encoded_args))
        return _result_from_response(resp)

    # Arithmetic
    def __add__(self, other):
        return self._dispatch_op("add", self, other)

    def __radd__(self, other):
        return self._dispatch_op("add", other, self)

    def __sub__(self, other):
        return self._dispatch_op("subtract", self, other)

    def __rsub__(self, other):
        return self._dispatch_op("subtract", other, self)

    def __mul__(self, other):
        return self._dispatch_op("multiply", self, other)

    def __rmul__(self, other):
        return self._dispatch_op("multiply", other, self)

    def __truediv__(self, other):
        return self._dispatch_op("true_divide", self, other)

    def __rtruediv__(self, other):
        return self._dispatch_op("true_divide", other, self)

    def __floordiv__(self, other):
        return self._dispatch_op("floor_divide", self, other)

    def __mod__(self, other):
        return self._dispatch_op("remainder", self, other)

    def __pow__(self, other):
        return self._dispatch_op("power", self, other)

    def __matmul__(self, other):
        return self._dispatch_op("matmul", self, other)

    def __neg__(self):
        return self._dispatch_op("negative", self)

    def __abs__(self):
        return self._dispatch_op("abs", self)

    # Comparison operators (return RemoteArray of bools)
    def __eq__(self, other):
        return self._dispatch_op("equal", self, other)

    def __ne__(self, other):
        return self._dispatch_op("not_equal", self, other)

    def __lt__(self, other):
        return self._dispatch_op("less", self, other)

    def __le__(self, other):
        return self._dispatch_op("less_equal", self, other)

    def __gt__(self, other):
        return self._dispatch_op("greater", self, other)

    def __ge__(self, other):
        return self._dispatch_op("greater_equal", self, other)


def _result_from_response(resp: dict) -> RemoteArray | RemoteScalar | tuple:
    """Convert a server response to the appropriate client-side object."""
    result = resp.get("result", {})

    if "value" in result:
        # Scalar result
        return RemoteScalar(value=result["value"], dtype="float64")
    elif "multi" in result:
        # Multi-output (tuple of arrays)
        items = []
        for item in result["multi"]:
            if isinstance(item, dict) and "id" in item:
                items.append(RemoteArray(
                    handle_id=item["id"],
                    shape=tuple(item["shape"]),
                    dtype=item["dtype"],
                ))
            else:
                items.append(item)
        return tuple(items)
    elif "id" in result:
        # Single array result
        return RemoteArray(
            handle_id=result["id"],
            shape=tuple(result["shape"]),
            dtype=result["dtype"],
        )
    else:
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-client && python -m pytest tests/test_remote_array.py -v`
Expected: All PASS (metadata and scalar tests pass; operator tests that need a server are in integration tests)

- [ ] **Step 5: Commit**

```bash
git add mechestim-client/src/mechestim/_remote_array.py mechestim-client/tests/test_remote_array.py
git commit -m "feat(client): add RemoteArray and RemoteScalar transparent proxy classes"
```

---

## Task 11: Client BudgetContext Proxy

**Files:**
- Create: `mechestim-client/src/mechestim/_budget.py`
- Create: `mechestim-client/tests/test_budget.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-client/tests/test_budget.py
import pytest

from mechestim._budget import BudgetContext


class TestBudgetContextProxy:
    def test_attributes(self):
        ctx = BudgetContext.__new__(BudgetContext)
        ctx._flop_budget = 10_000
        ctx._flops_used = 0
        ctx._is_open = False
        assert ctx.flop_budget == 10_000
        assert ctx.flops_used == 0
        assert ctx.flops_remaining == 10_000

    def test_update_from_server(self):
        ctx = BudgetContext.__new__(BudgetContext)
        ctx._flop_budget = 10_000
        ctx._flops_used = 0
        ctx._is_open = True
        ctx._update_budget({"flop_budget": 10_000, "flops_used": 500, "flops_remaining": 9500})
        assert ctx.flops_used == 500
        assert ctx.flops_remaining == 9500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-client && python -m pytest tests/test_budget.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BudgetContext proxy**

```python
# mechestim-client/src/mechestim/_budget.py
"""Client-side BudgetContext that proxies to the server."""

from __future__ import annotations

from typing import Any

from mechestim._connection import get_connection
from mechestim._protocol import encode_budget_open, encode_budget_close, encode_budget_status
from mechestim.errors import NoBudgetContextError


class OpRecord:
    """Placeholder for operation records. Full log lives on server."""

    def __init__(self, op_name: str, flop_cost: int, cumulative: int) -> None:
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.cumulative = cumulative


class BudgetContext:
    """Transparent proxy for server-side BudgetContext.

    Usage is identical to the real mechestim BudgetContext:

        with me.BudgetContext(flop_budget=10_000_000) as budget:
            x = me.zeros((256,))
            y = me.exp(x)
            print(budget.flops_used)
            print(budget.summary())
    """

    def __init__(self, flop_budget: int, flop_multiplier: float = 1.0, quiet: bool = False) -> None:
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._quiet = quiet
        self._flops_used = 0
        self._is_open = False
        self._close_summary: dict | None = None

    @property
    def flop_budget(self) -> int:
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        return self._flop_budget - self._flops_used

    @property
    def flop_multiplier(self) -> float:
        return self._flop_multiplier

    def _update_budget(self, budget_info: dict) -> None:
        """Update local budget cache from a server response."""
        if "flops_used" in budget_info:
            self._flops_used = budget_info["flops_used"]

    def summary(self) -> str:
        """Get the budget summary from the server."""
        if self._close_summary is not None:
            return self._close_summary.get("budget_summary", "session closed")

        conn = get_connection()
        resp = conn.send_recv(encode_budget_status())
        self._update_budget(resp.get("result", {}))
        budget = resp.get("result", {})
        return (
            f"BudgetContext: {budget.get('flops_used', 0):,} / {budget.get('flop_budget', 0):,} FLOPs used "
            f"({budget.get('flops_remaining', 0):,} remaining)"
        )

    def __enter__(self) -> BudgetContext:
        conn = get_connection()
        resp = conn.send_recv(encode_budget_open(self._flop_budget))
        self._is_open = True
        budget_info = resp.get("budget", {})
        self._update_budget(budget_info)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._is_open:
            conn = get_connection()
            resp = conn.send_recv(encode_budget_close())
            self._is_open = False
            self._close_summary = resp.get("result", {})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-client && python -m pytest tests/test_budget.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-client/src/mechestim/_budget.py mechestim-client/tests/test_budget.py
git commit -m "feat(client): add BudgetContext proxy delegating to server"
```

---

## Task 12: Client Registry and Module API Surface

**Files:**
- Create: `mechestim-client/src/mechestim/_registry.py`
- Create: `mechestim-client/src/mechestim/_getattr.py`
- Modify: `mechestim-client/src/mechestim/__init__.py`
- Create: `mechestim-client/src/mechestim/flops.py`
- Create: `mechestim-client/src/mechestim/linalg/__init__.py`
- Create: `mechestim-client/src/mechestim/random/__init__.py`
- Create: `mechestim-client/src/mechestim/fft/__init__.py`
- Create: `mechestim-client/tests/test_registry.py`

- [ ] **Step 1: Copy the registry categories from the real mechestim**

The client needs a static copy of the function registry — just the names and categories, no numpy dependency.

```bash
# Extract category data from the real registry
cd /path/to/mechestim
python -c "
from mechestim._registry import REGISTRY, REGISTRY_META
import json
slim = {name: {'category': entry['category'], 'module': entry['module']} for name, entry in REGISTRY.items()}
print(json.dumps({'meta': REGISTRY_META, 'registry': slim}, indent=2))
" > /path/to/mechestim-client/src/mechestim/_registry_data.json
```

Alternatively, build the registry statically in `_registry.py` by importing and re-exporting the category mapping. For the implementation, create a Python file that contains the registry as a dict literal:

```python
# mechestim-client/src/mechestim/_registry.py
"""Static function registry — categories only, no numpy dependency.

This file is generated from the real mechestim registry. Keep in sync
by running: python scripts/export_client_registry.py
"""

from __future__ import annotations

REGISTRY_META = {
    "numpy_version": "2.1.3",
    "last_updated": "2026-04-01",
}

# Category constants
COUNTED_UNARY = "counted_unary"
COUNTED_BINARY = "counted_binary"
COUNTED_REDUCTION = "counted_reduction"
COUNTED_CUSTOM = "counted_custom"
FREE = "free"
BLACKLISTED = "blacklisted"

# Function name → category
# This is a subset — only the category is needed by the client
FUNCTION_CATEGORIES: dict[str, str] = {}  # Populated by sync script


def is_valid_op(name: str) -> bool:
    """Check if a function name is a valid mechestim operation."""
    return name in FUNCTION_CATEGORIES


def get_category(name: str) -> str | None:
    """Get the category of a function, or None if unknown."""
    return FUNCTION_CATEGORIES.get(name)
```

Write a script to populate `FUNCTION_CATEGORIES` from the real registry:

```python
# scripts/export_client_registry.py
"""Export the mechestim registry for the client package."""

import json
from mechestim._registry import REGISTRY

categories = {name: entry["category"] for name, entry in REGISTRY.items()}
print(f"FUNCTION_CATEGORIES = {json.dumps(categories, indent=4)}")
```

Run this script and paste the output into `_registry.py` replacing the empty dict.

- [ ] **Step 2: Write failing test for registry**

```python
# mechestim-client/tests/test_registry.py
from mechestim._registry import is_valid_op, get_category, FUNCTION_CATEGORIES


class TestClientRegistry:
    def test_registry_not_empty(self):
        assert len(FUNCTION_CATEGORIES) > 100

    def test_known_ops(self):
        assert is_valid_op("add")
        assert is_valid_op("exp")
        assert is_valid_op("zeros")
        assert is_valid_op("einsum")

    def test_unknown_op(self):
        assert not is_valid_op("hack_the_planet")

    def test_categories(self):
        assert get_category("add") == "counted_binary"
        assert get_category("exp") == "counted_unary"
        assert get_category("zeros") == "free"
        assert get_category("sum") == "counted_reduction"

    def test_blacklisted(self):
        assert get_category("save") == "blacklisted"
        assert get_category("load") == "blacklisted"
```

- [ ] **Step 3: Implement __getattr__ for helpful error messages**

```python
# mechestim-client/src/mechestim/_getattr.py
"""Module-level __getattr__ for blacklisted/unknown function access."""

from __future__ import annotations

from mechestim._registry import get_category, BLACKLISTED


def make_module_getattr(module_prefix: str, module_label: str):
    """Create a __getattr__ that provides helpful errors for unknown attributes."""

    def __getattr__(name: str) -> None:
        qualified = f"{module_prefix}{name}" if module_prefix else name
        category = get_category(qualified)

        if category == BLACKLISTED:
            raise AttributeError(
                f"'{module_label}.{name}' is intentionally not supported in mechestim. "
                f"This function is blacklisted because it has no meaningful FLOP cost model."
            )
        elif category is not None:
            raise AttributeError(
                f"'{module_label}.{name}' is registered but not yet implemented in the client."
            )
        else:
            raise AttributeError(
                f"module '{module_label}' has no attribute '{name}'"
            )

    return __getattr__
```

- [ ] **Step 4: Implement flops proxy**

```python
# mechestim-client/src/mechestim/flops.py
"""Proxy to server-side FLOP cost queries."""

from __future__ import annotations

from mechestim._connection import get_connection
from mechestim._protocol import encode_request


def einsum_cost(subscripts: str, shapes: list[tuple[int, ...]], **kwargs) -> int:
    conn = get_connection()
    resp = conn.send_recv(encode_request("flops.einsum_cost", args=[subscripts, shapes], kwargs=kwargs))
    return resp["result"]["value"]


def svd_cost(m: int, n: int, k: int | None = None) -> int:
    conn = get_connection()
    args = [m, n] if k is None else [m, n, k]
    resp = conn.send_recv(encode_request("flops.svd_cost", args=args))
    return resp["result"]["value"]


def pointwise_cost(shape: tuple[int, ...]) -> int:
    """Calculate locally — no server round-trip needed."""
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)


def reduction_cost(input_shape: tuple[int, ...], axis: int | None = None) -> int:
    """Calculate locally — no server round-trip needed."""
    result = 1
    for dim in input_shape:
        result *= dim
    return max(result, 1)
```

- [ ] **Step 5: Implement submodule proxies**

```python
# mechestim-client/src/mechestim/linalg/__init__.py
"""mechestim.linalg — proxied to server."""

from mechestim._remote_array import RemoteArray, _result_from_response
from mechestim._connection import get_connection
from mechestim._protocol import encode_request
from mechestim._getattr import make_module_getattr


def svd(a, k=None):
    """Proxied SVD."""
    args = [a.handle_id if isinstance(a, RemoteArray) else a]
    kwargs = {} if k is None else {"k": k}
    conn = get_connection()
    resp = conn.send_recv(encode_request("linalg.svd", args=args, kwargs=kwargs))
    return _result_from_response(resp)


__all__ = ["svd"]
__getattr__ = make_module_getattr(module_prefix="linalg.", module_label="mechestim.linalg")
```

```python
# mechestim-client/src/mechestim/random/__init__.py
"""mechestim.random — proxied to server (all 0 FLOPs)."""

from mechestim._remote_array import RemoteArray, _result_from_response
from mechestim._connection import get_connection
from mechestim._protocol import encode_request


def _random_proxy(func_name):
    def wrapper(*args, **kwargs):
        conn = get_connection()
        resp = conn.send_recv(encode_request(f"random.{func_name}", args=list(args), kwargs=kwargs))
        return _result_from_response(resp)
    wrapper.__name__ = func_name
    wrapper.__qualname__ = f"random.{func_name}"
    return wrapper


randn = _random_proxy("randn")
normal = _random_proxy("normal")
uniform = _random_proxy("uniform")
rand = _random_proxy("rand")
seed = _random_proxy("seed")
choice = _random_proxy("choice")
permutation = _random_proxy("permutation")
shuffle = _random_proxy("shuffle")

__all__ = ["randn", "normal", "uniform", "rand", "seed", "choice", "permutation", "shuffle"]


def __getattr__(name: str):
    # Forward unknown random functions to server
    return _random_proxy(name)
```

```python
# mechestim-client/src/mechestim/fft/__init__.py
"""mechestim.fft — all blacklisted."""

from mechestim._getattr import make_module_getattr

__all__: list[str] = []
__getattr__ = make_module_getattr(module_prefix="fft.", module_label="mechestim.fft")
```

- [ ] **Step 6: Implement client __init__.py with auto-generated proxies**

```python
# mechestim-client/src/mechestim/__init__.py
"""mechestim — transparent proxy to a remote mechestim server.

Drop-in replacement for the real mechestim library. All operations are
dispatched to a backend server via ZeroMQ. Participants use this package
identically to the real mechestim.
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Errors (same classes as real mechestim) ---
from mechestim.errors import (
    BudgetExhaustedError,
    MechEstimError,
    MechEstimServerError,
    MechEstimWarning,
    NoBudgetContextError,
    SymmetryError,
)

# --- Budget ---
from mechestim._budget import BudgetContext, OpRecord

# --- RemoteArray as ndarray stand-in ---
from mechestim._remote_array import RemoteArray as ndarray, RemoteArray, RemoteScalar

# --- Submodules ---
from mechestim import fft, flops, linalg, random

# --- Registry ---
from mechestim._registry import FUNCTION_CATEGORIES, get_category

# --- Connection ---
from mechestim._connection import get_connection
from mechestim._protocol import encode_request
from mechestim._remote_array import _result_from_response

# --- Constants (no server round-trip) ---
import math as _math

pi = _math.pi
e = _math.e
inf = float("inf")
nan = float("nan")
newaxis = None

# --- Dtypes (string identifiers — no numpy needed) ---
float16 = "float16"
float32 = "float32"
float64 = "float64"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
bool_ = "bool_"
complex64 = "complex64"
complex128 = "complex128"


# --- Auto-generated proxy functions ---

def _make_proxy(op_name: str):
    """Create a proxy function that dispatches to the server."""

    def proxy(*args, **kwargs):
        conn = get_connection()
        # Encode args: RemoteArray → handle_id, RemoteScalar → value
        encoded_args = []
        for arg in args:
            if isinstance(arg, RemoteArray):
                encoded_args.append(arg.handle_id)
            elif isinstance(arg, RemoteScalar):
                encoded_args.append(arg._value)
            else:
                encoded_args.append(arg)
        resp = conn.send_recv(encode_request(op_name, args=encoded_args, kwargs=kwargs))
        return _result_from_response(resp)

    proxy.__name__ = op_name
    proxy.__qualname__ = f"mechestim.{op_name}"
    proxy.__doc__ = f"[mechestim-client] Proxied to server: {op_name}"
    return proxy


# Generate proxy for every registered function that isn't blacklisted
_SKIP_CATEGORIES = {"blacklisted", "unclassified"}
_ALREADY_DEFINED = set(dir())  # Don't overwrite explicit imports

for _name, _category in FUNCTION_CATEGORIES.items():
    if _category not in _SKIP_CATEGORIES and _name not in _ALREADY_DEFINED:
        # Skip dotted names (submodule functions like linalg.svd)
        if "." not in _name:
            globals()[_name] = _make_proxy(_name)


# Special case: array() needs to handle Python lists/raw data
def array(object, dtype=None, **kwargs):
    """Create an array from Python data. Sends raw bytes to server."""
    import struct as _struct

    # Convert Python list to flat bytes + shape
    if isinstance(object, (list, tuple)):
        flat, shape = _flatten(object)
        dtype_str = dtype if isinstance(dtype, str) else (dtype or "float64")
        fmt = {"float64": "d", "float32": "f", "int64": "q", "int32": "i", "int8": "b", "uint8": "B", "float16": "e", "bool_": "?"}
        fmt_char = fmt.get(dtype_str, "d")
        data = _struct.pack(f"<{len(flat)}{fmt_char}", *flat)
        from mechestim._protocol import encode_create_from_data
        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, list(shape), dtype_str))
        return _result_from_response(resp)
    elif isinstance(object, RemoteArray):
        return object  # Already a remote array
    else:
        raise TypeError(f"Cannot create array from {type(object)}")


def _flatten(obj, depth=0):
    """Flatten a nested list/tuple and return (flat_list, shape)."""
    if not isinstance(obj, (list, tuple)):
        return [obj], ()
    if len(obj) == 0:
        return [], (0,)
    first_flat, inner_shape = _flatten(obj[0], depth + 1)
    flat = list(first_flat)
    for item in obj[1:]:
        item_flat, _ = _flatten(item, depth + 1)
        flat.extend(item_flat)
    return flat, (len(obj),) + inner_shape


# Special case: einsum needs string subscripts handled
def einsum(subscripts: str, *operands, **kwargs):
    """Proxied einsum — subscripts string + array operands."""
    conn = get_connection()
    encoded_args = [subscripts]
    for op in operands:
        if isinstance(op, RemoteArray):
            encoded_args.append(op.handle_id)
        elif isinstance(op, RemoteScalar):
            encoded_args.append(op._value)
        else:
            encoded_args.append(op)
    resp = conn.send_recv(encode_request("einsum", args=encoded_args, kwargs=kwargs))
    return _result_from_response(resp)


# --- Module-level __getattr__ for blacklisted/unknown ---
from mechestim._getattr import make_module_getattr
__getattr__ = make_module_getattr(module_prefix="", module_label="mechestim")
```

- [ ] **Step 7: Run registry test**

Run: `cd mechestim-client && python -m pytest tests/test_registry.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add mechestim-client/src/mechestim/
git commit -m "feat(client): add full module API surface with auto-generated proxies, submodules, and registry"
```

---

## Task 13: Client-Side CommsTracker

**Files:**
- Create: `mechestim-client/src/mechestim/_comms_tracker.py`
- Create: `mechestim-client/tests/test_comms_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# mechestim-client/tests/test_comms_tracker.py
from mechestim._comms_tracker import ClientCommsTracker


class TestClientCommsTracker:
    def test_initial_state(self):
        tracker = ClientCommsTracker()
        assert tracker.total_round_trip_ns == 0
        assert tracker.request_count == 0

    def test_record(self):
        tracker = ClientCommsTracker()
        tracker.record(round_trip_ns=50_000, request_bytes=100, response_bytes=200)
        assert tracker.request_count == 1
        assert tracker.total_round_trip_ns == 50_000
        assert tracker.total_request_bytes == 100
        assert tracker.total_response_bytes == 200

    def test_accumulates(self):
        tracker = ClientCommsTracker()
        tracker.record(round_trip_ns=1000, request_bytes=10, response_bytes=20)
        tracker.record(round_trip_ns=2000, request_bytes=30, response_bytes=40)
        assert tracker.request_count == 2
        assert tracker.total_round_trip_ns == 3000
        assert tracker.total_request_bytes == 40
        assert tracker.total_response_bytes == 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mechestim-client && python -m pytest tests/test_comms_tracker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# mechestim-client/src/mechestim/_comms_tracker.py
"""Client-side communication overhead tracking."""

from __future__ import annotations


class ClientCommsTracker:
    """Tracks client-side round-trip timing and byte counts."""

    def __init__(self) -> None:
        self.request_count: int = 0
        self.total_round_trip_ns: int = 0
        self.total_request_bytes: int = 0
        self.total_response_bytes: int = 0

    def record(self, *, round_trip_ns: int, request_bytes: int, response_bytes: int) -> None:
        self.request_count += 1
        self.total_round_trip_ns += round_trip_ns
        self.total_request_bytes += request_bytes
        self.total_response_bytes += response_bytes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mechestim-client && python -m pytest tests/test_comms_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mechestim-client/src/mechestim/_comms_tracker.py mechestim-client/tests/test_comms_tracker.py
git commit -m "feat(client): add client-side comms overhead tracker"
```

---

## Task 14: End-to-End Integration Tests

**Files:**
- Create: `mechestim-client/tests/test_integration.py`

These tests start a real server and exercise the full client API.

- [ ] **Step 1: Write integration tests**

```python
# mechestim-client/tests/test_integration.py
"""Full end-to-end tests with a real mechestim server."""

import os
import threading
import time
import sys

import pytest

# Add mechestim-server to path for testing
SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "mechestim-server", "src")
sys.path.insert(0, SERVER_DIR)


@pytest.fixture(scope="module")
def server():
    """Start a real mechestim server on a test port."""
    from mechestim_server._server import MechestimServer

    url = "tcp://127.0.0.1:15556"
    srv = MechestimServer(url=url, session_timeout_s=30)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    time.sleep(0.2)

    # Point client at test server
    os.environ["MECHESTIM_SERVER_URL"] = url

    yield srv

    srv.stop()
    thread.join(timeout=2)


@pytest.fixture(autouse=True)
def reset_client_connection():
    """Reset client connection state between tests."""
    from mechestim._connection import reset_connection
    reset_connection()
    yield
    reset_connection()


class TestBasicOps:
    def test_zeros_and_fetch(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as budget:
            x = me.zeros((3, 4))
            assert x.shape == (3, 4)
            assert x.dtype == "float64"
            data = x.tolist()
            assert data == [[0.0] * 4] * 3

    def test_ones(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.ones((5,))
            assert x.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_array_from_list(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0, 3.0])
            assert x.shape == (3,)
            assert x.tolist() == [1.0, 2.0, 3.0]


class TestCountedOps:
    def test_exp(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as budget:
            x = me.ones((10,))
            y = me.exp(x)
            assert y.shape == (10,)
            data = y.tolist()
            assert all(abs(v - 2.718281828) < 0.001 for v in data)
            assert budget.flops_used == 10

    def test_add(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as budget:
            x = me.ones((5,))
            y = me.ones((5,))
            z = me.add(x, y)
            assert z.tolist() == [2.0, 2.0, 2.0, 2.0, 2.0]

    def test_sum_reduction(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0, 3.0])
            s = me.sum(x)
            assert float(s) == 6.0


class TestOperators:
    def test_add_operator(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.ones((3,))
            y = me.ones((3,))
            z = x + y
            assert z.tolist() == [2.0, 2.0, 2.0]

    def test_mul_scalar(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0, 3.0])
            z = x * 2.0
            assert z.tolist() == [2.0, 4.0, 6.0]

    def test_neg(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, -2.0, 3.0])
            z = -x
            assert z.tolist() == [-1.0, 2.0, -3.0]

    def test_matmul(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            a = me.array([[1.0, 0.0], [0.0, 1.0]])
            b = me.array([3.0, 4.0])
            c = a @ b
            assert c.tolist() == [3.0, 4.0]


class TestTransparency:
    """Tests that the client is truly transparent — no 'RemoteArray' visible."""

    def test_print_shows_values(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0])
            repr_str = repr(x)
            assert "array(" in repr_str
            assert "1.0" in repr_str

    def test_iteration(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([10.0, 20.0, 30.0])
            values = [v for v in x]
            assert values == [10.0, 20.0, 30.0]

    def test_indexing(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([10.0, 20.0, 30.0])
            assert x[0] == 10.0
            assert x[2] == 30.0

    def test_len(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.zeros((7,))
            assert len(x) == 7

    def test_float_conversion(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([42.0])
            assert float(x) == 42.0


class TestEinsum:
    def test_einsum_matvec(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            W = me.array([[1.0, 0.0], [0.0, 1.0]])
            x = me.array([3.0, 4.0])
            y = me.einsum("ij,j->i", W, x)
            assert y.tolist() == [3.0, 4.0]


class TestErrors:
    def test_budget_exhausted(self, server):
        import mechestim as me

        with pytest.raises(me.BudgetExhaustedError):
            with me.BudgetContext(flop_budget=1):
                x = me.ones((1000,))
                me.exp(x)  # Costs 1000 FLOPs, budget is 1

    def test_no_budget_context(self, server):
        import mechestim as me

        # Operations outside budget context should fail
        # (server has no active session)
        from mechestim.errors import NoBudgetContextError, MechEstimServerError
        with pytest.raises((NoBudgetContextError, MechEstimServerError)):
            me.zeros((3,))

    def test_blacklisted_function(self, server):
        import mechestim as me

        with pytest.raises(AttributeError, match="blacklisted"):
            me.save

    def test_unknown_function(self, server):
        import mechestim as me

        with pytest.raises(AttributeError):
            me.nonexistent_function


class TestBudgetTracking:
    def test_flops_tracked(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as budget:
            x = me.ones((100,))
            me.exp(x)  # 100 FLOPs
            me.exp(x)  # 100 more
            assert budget.flops_used == 200
            assert budget.flops_remaining == 999_800

    def test_summary(self, server):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as budget:
            x = me.ones((10,))
            me.exp(x)
            summary = budget.summary()
            assert "10" in summary  # 10 FLOPs used
```

- [ ] **Step 2: Run integration tests**

Run: `cd mechestim-client && python -m pytest tests/test_integration.py -v --timeout=30`
Expected: All PASS

Note: These tests start a real server thread. If any fail, check:
1. Is `mechestim-server` importable? (Its `src/` must be on sys.path)
2. Is port 15556 available?
3. Is the real `mechestim` package installed in the environment running the server?

- [ ] **Step 3: Commit**

```bash
git add mechestim-client/tests/test_integration.py
git commit -m "test: add full end-to-end integration tests for client-server"
```

---

## Task 15: Docker Compose Configuration

**Files:**
- Create: `docker/docker-compose.yml`
- Create: `docker/Dockerfile.server`
- Create: `docker/Dockerfile.participant`

- [ ] **Step 1: Create server Dockerfile**

```dockerfile
# docker/Dockerfile.server
FROM python:3.10-slim

WORKDIR /app

# Install mechestim (real lib) and server
COPY mechestim-server/ /app/mechestim-server/
COPY . /app/mechestim-lib/

RUN pip install --no-cache-dir /app/mechestim-lib/ /app/mechestim-server/

CMD ["python", "-m", "mechestim_server", "--url", "ipc:///tmp/mechestim.sock"]
```

- [ ] **Step 2: Create participant Dockerfile**

```dockerfile
# docker/Dockerfile.participant
FROM python:3.10-slim

WORKDIR /app

# Install ONLY the client — no numpy
COPY mechestim-client/ /app/mechestim-client/
RUN pip install --no-cache-dir /app/mechestim-client/

# Verify numpy is NOT installed
RUN python -c "import mechestim; print('client installed')" && \
    python -c "import numpy" 2>&1 | grep -q "No module" && \
    echo "GOOD: numpy is not installed"

ENV MECHESTIM_SERVER_URL=ipc:///tmp/mechestim.sock

# Participant code is mounted at runtime
CMD ["python", "/submission/run.py"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.server
    volumes:
      - mechestim-sock:/tmp
    restart: unless-stopped

  participant:
    build:
      context: ..
      dockerfile: docker/Dockerfile.participant
    volumes:
      - mechestim-sock:/tmp
      - ./submissions:/submission:ro
    environment:
      - MECHESTIM_SERVER_URL=ipc:///tmp/mechestim.sock
    depends_on:
      - backend

volumes:
  mechestim-sock:
```

- [ ] **Step 4: Test Docker setup**

```bash
cd docker
# Create a test submission
mkdir -p submissions
cat > submissions/run.py << 'PYEOF'
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    x = me.ones((100,))
    y = me.exp(x)
    z = x + y
    print(f"Result shape: {z.shape}")
    print(f"First value: {z[0]}")
    print(f"FLOPs used: {budget.flops_used}")
    print(f"Summary: {budget.summary()}")
    print("SUCCESS: client-server working!")
PYEOF

docker compose up --build
```
Expected: "SUCCESS: client-server working!" in output

- [ ] **Step 5: Commit**

```bash
git add docker/
git commit -m "feat: add Docker compose setup for client-server deployment"
```

---

## Summary

| Task | What it builds | Tests |
|---|---|---|
| 1 | Package scaffolding | Install check |
| 2 | Server protocol layer | 12 tests |
| 3 | Server ArrayStore | 9 tests |
| 4 | Server CommsTracker | 6 tests |
| 5 | Server Session | 6 tests |
| 6 | Server RequestHandler | 14 tests |
| 7 | Server ZMQ loop + entry point | 6 tests |
| 8 | Client errors | 9 tests |
| 9 | Client connection + protocol | 8 tests |
| 10 | Client RemoteArray/Scalar | 18 tests |
| 11 | Client BudgetContext proxy | 2 tests |
| 12 | Client registry + API surface + submodules | 5 tests |
| 13 | Client CommsTracker | 3 tests |
| 14 | Integration tests | 18 tests |
| 15 | Docker compose | Manual verification |

**Total: ~116 tests across 15 tasks**
