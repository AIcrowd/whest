# Mechestim Client-Server Architecture Design

**Date:** 2026-04-01
**Status:** Draft
**Goal:** Enable mechestim to run as a client-server system where adversarial participants use a lightweight client (no numpy) that transparently proxies operations to a backend server (has numpy + mechestim).

---

## Problem

We run a public competition where participants submit code that uses `mechestim` to perform numpy-backed computations with FLOP budgets. Participants are adversarial — they may attempt to monkey-patch numpy, bypass budget enforcement, or otherwise cheat. We need to remove numpy from the participant container entirely while maintaining an identical API experience.

## Constraints

- **No numpy in participant container.** The client package must have zero numpy dependency.
- **Transparent to participants.** `import mechestim as me` works identically. Participants cannot tell if they're running locally or via client-server.
- **Python operators work.** `x + y`, `x @ W`, `x > 0.5` all dispatch to server-side mechestim ops with FLOP tracking.
- **Debugging is transparent.** `print(x)`, `x[0]`, `x.tolist()` auto-fetch data invisibly.
- **Server-side authoritative budget.** The server is the single source of truth for FLOP accounting. No client-side budget state that could be tampered with.
- **Track communication overhead.** Separate computation time from serialization/network overhead for competition validation.
- **Support same-host (primary) and cross-host (fallback) deployment.**

---

## Architecture Overview

Three packages, two processes, one API:

```
PARTICIPANT CONTAINER (no numpy)          BACKEND SERVER (has numpy)
┌─────────────────────────────┐          ┌──────────────────────────────┐
│  participant_code.py        │          │  mechestim-server            │
│    import mechestim as me   │          │                              │
│    x = me.zeros((256,))     │          │  ┌────────────────────────┐  │
│    y = me.exp(x)            │          │  │  numpy + mechestim     │  │
│    z = x + y                │          │  │  (real computation)    │  │
│                             │          │  └────────────────────────┘  │
│  ┌────────────────────────┐ │          │                              │
│  │  mechestim-client      │ │  ZeroMQ  │  ┌────────────────────────┐  │
│  │  (drop-in replacement) │◄├──────────┤► │  Request handler       │  │
│  │                        │ │  msgpack  │  │  - execute ops         │  │
│  │  - RemoteArray proxy   │ │  IPC/TCP  │  │  - manage array store  │  │
│  │  - BudgetContext proxy │ │          │  │  - track budget        │  │
│  │  - operator overloads  │ │          │  │  - track comms overhead│  │
│  └────────────────────────┘ │          │  └────────────────────────┘  │
└─────────────────────────────┘          └──────────────────────────────┘
```

### Three Packages

- **`mechestim`** — the existing library, unchanged. Lives on the server. Has numpy.
- **`mechestim-client`** — drop-in replacement installed in participant containers. Same Python package name (`mechestim`). Dependencies: `pyzmq`, `msgpack` only.
- **`mechestim-server`** — backend daemon. Dependencies: `mechestim`, `numpy`, `pyzmq`, `msgpack`.

Both `mechestim` and `mechestim-client` install as the `mechestim` Python package. They are mutually exclusive — install one or the other.

---

## RemoteArray: The Transparent Proxy

`RemoteArray` replaces `numpy.ndarray` on the client side. Participants never see it as anything other than a normal array.

### Metadata (cached locally, no round-trip)

- `.shape`, `.dtype`, `.ndim`, `.size`, `.nbytes`

Note: `.T` (transpose) dispatches to the server as an operation, since it produces a new array.

### Operations (dispatch to server, return new RemoteArray)

- `me.exp(x)` → server executes, returns new RemoteArray
- `x + y` → `__add__` dispatches `me.add(x, y)`
- `x @ W` → `__matmul__` dispatches `me.matmul(x, W)`

### Data access (auto-fetch, invisible to participant)

- `print(x)` → `__repr__` fetches data, formats like numpy
- `x[0, 3]` → `__getitem__` fetches slice
- `float(x)` → `__float__` fetches scalar
- `x.tolist()` → fetches full array, returns Python list
- `for v in x:` → `__iter__` fetches and iterates
- `if x:` → `__bool__` fetches scalar truth value

### Operator Mapping (all FLOP-tracked on server)

| Python operator | Dispatches to | FLOP category |
|---|---|---|
| `x + y` | `me.add(x, y)` | counted_binary |
| `x - y` | `me.subtract(x, y)` | counted_binary |
| `x * y` | `me.multiply(x, y)` | counted_binary |
| `x / y` | `me.divide(x, y)` | counted_binary |
| `x ** y` | `me.power(x, y)` | counted_binary |
| `x @ y` | `me.matmul(x, y)` | counted_custom |
| `-x` | `me.negative(x)` | counted_unary |
| `abs(x)` | `me.abs(x)` | counted_unary |
| `x > y`, `x == y`, etc. | `me.greater(x, y)`, etc. | counted_binary |
| `x += y` | `me.add(x, y)` (rebind) | counted_binary |

### RemoteScalar

When a mechestim op returns a 0-d array (e.g., `me.sum(x)`), the server sends the value inline. The client wraps it in `RemoteScalar` — behaves like a Python float/int but is also a valid mechestim operand.

### What RemoteArray Does NOT Support

- Direct mutation (`x[0] = 5`) — arrays are immutable (same as current mechestim)
- Buffer protocol / `__array__()` — no numpy on the client

---

## Wire Protocol

ZeroMQ REQ/REP pattern. Every exchange is one request, one response. Serialized with msgpack.

### Argument Encoding

- **RemoteArray arguments** → string handle ID (e.g., `"a7"`)
- **Python scalars** → inline msgpack int/float
- **Raw array data** (for `me.array()`) → `create_from_data` with raw bytes + shape + dtype
- **Strings, tuples, None** → inline msgpack native types

The server resolves handle IDs to numpy arrays via its ArrayStore before executing.

### Message Types

| Type | Purpose | Example |
|---|---|---|
| `op` | Execute any mechestim function (including array creation like `zeros`, `ones`) | `{"op": "add", "args": ["a1", "a2"]}` |
| `create_from_data` | Client sends raw bytes to create array (used by `me.array()`) | `{"op": "create_from_data", "data": <bytes>, "shape": [10,10], "dtype": "float64"}` |
| `fetch` | Pull array data to client | `{"op": "fetch", "id": "a7"}` |
| `fetch_slice` | Pull a slice | `{"op": "fetch_slice", "id": "a7", "slices": [[0, 5]]}` |
| `budget_open` | Start a BudgetContext | `{"op": "budget_open", "flop_budget": 10000000}` |
| `budget_status` | Query budget state | `{"op": "budget_status"}` |
| `budget_close` | End context, get summary | `{"op": "budget_close"}` |
| `free` | Explicitly release handles | `{"op": "free", "ids": ["a1", "a2"]}` |

### Response Format

Every response includes budget state and comms overhead:

```python
{
    "status": "ok",
    "result": {
        "id": "a8",
        "shape": [256],
        "dtype": "float64"
    },
    "budget": {
        "flops_used": 256,
        "flops_remaining": 9999744,
        "comms_overhead_ns": 48000
    }
}
```

Fetch responses include raw bytes:

```python
{
    "status": "ok",
    "data": <raw bytes>,
    "shape": [256, 256],
    "dtype": "float64",
    "comms_overhead_ns": 120000
}
```

---

## Server-Side Array Store & Session Lifecycle

```
SERVER PROCESS
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  ArrayStore: {"a1": ndarray, "a2": ndarray, ...}        │
│  BudgetContext: flop_budget, flops_used, op_log          │
│  CommsTracker: bytes, timing, request counts             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Session Lifecycle

1. **`budget_open`** — creates session with ArrayStore, BudgetContext, CommsTracker. Returns session ID.
2. **Operations** — execute within session's BudgetContext. Arrays accumulate in ArrayStore.
3. **`budget_close`** — returns budget summary + comms summary. All arrays freed.
4. **Crash/timeout** — server reaps session after configurable timeout (default 60s). All arrays freed.

### GC is Session-Scoped

No reference counting. All arrays freed when session ends. This works because:
- Each participant submission is one session
- Sessions are short-lived (bounded by FLOP budget)
- No cross-session array sharing

### Handle ID Generation

Monotonic counter per session: `"a0"`, `"a1"`, `"a2"`, ... Simple, fast, debuggable.

---

## Transport Layer

### Same-Host (Primary)

ZeroMQ IPC transport via Unix socket:
```
ipc:///tmp/mechestim.sock
```
Latency: ~10-30 microseconds per call.

### Cross-Host (Fallback)

ZeroMQ TCP transport:
```
tcp://host:5555
```
Latency: ~0.2-1ms per call.

### Configuration

Single environment variable:
```
MECHESTIM_SERVER_URL=ipc:///tmp/mechestim.sock   # same host (default)
MECHESTIM_SERVER_URL=tcp://10.0.0.5:5555          # cross host
```

### Docker Compose (Same Host)

```yaml
services:
  participant:
    image: participant-sandbox
    volumes:
      - /tmp/mechestim.sock:/tmp/mechestim.sock
    environment:
      - MECHESTIM_SERVER_URL=ipc:///tmp/mechestim.sock

  backend:
    image: mechestim-server
    volumes:
      - /tmp/mechestim.sock:/tmp/mechestim.sock
```

### Timeouts

- Client: configurable per-request timeout (default 30s). Raises `MechEstimServerError`.
- Server: session reap timeout (default 60s inactivity).

---

## Communication Overhead Tracking

### Per-Request (Server-Side)

```
t0 → deserialize request (msgpack)  → t1   ← comms overhead
t1 → execute mechestim operation     → t2   ← computation time
t2 → serialize response (msgpack)    → t3   ← comms overhead

comms_overhead = (t1 - t0) + (t3 - t2)
compute_time   = (t2 - t1)
```

### Client-Side

- Round-trip wall time per request
- `network_latency = round_trip_time - server_total_time`

### Session Summary (returned on budget_close)

```python
{
    "budget_summary": { ... },          # normal mechestim budget summary
    "comms_summary": {
        "request_count": 47,
        "fetch_count": 3,
        "total_bytes_sent": 1_048_576,
        "total_bytes_received": 2_097_152,
        "total_comms_overhead_ns": 4_800_000,
        "total_compute_time_ns": 12_300_000,
        "total_network_latency_ns": 1_200_000,
        "overhead_ratio": 0.28
    }
}
```

---

## Client Package Structure

```
mechestim-client/
├── src/
│   └── mechestim/                  # Same module name as real lib
│       ├── __init__.py              # Identical exports, auto-generated proxies
│       ├── _remote_array.py         # RemoteArray + RemoteScalar
│       ├── _connection.py           # ZMQ connection management
│       ├── _protocol.py             # msgpack encode/decode
│       ├── _budget.py               # BudgetContext proxy
│       ├── _operators.py            # Operator overloads → server ops
│       ├── _comms_tracker.py        # Client-side overhead tracking
│       ├── errors.py                # Same exception classes as real mechestim
│       ├── flops.py                 # Proxy to server-side flop queries
│       ├── linalg/__init__.py       # svd → server dispatch
│       ├── random/__init__.py       # randn, normal → server dispatch
│       └── fft/__init__.py          # Blacklisted, same errors
├── pyproject.toml                   # deps: pyzmq, msgpack (NO numpy)
└── tests/
```

### Proxy Generation

Client reads from a static copy of the mechestim registry and auto-generates proxy stubs:
- `counted_unary`, `counted_binary`, `counted_reduction`, `counted_custom` → proxy that sends `{"op": name, ...}` to server
- `free` → same proxy protocol
- `blacklisted` → `__getattr__` raises same helpful `AttributeError` as real mechestim

Constants and dtypes (`me.pi`, `me.float32`, `me.ndarray`) are defined statically — no round-trip.

---

## Error Handling

### Error Propagation

Server catches mechestim exceptions, serializes them:

```python
{"status": "error", "error_type": "BudgetExhaustedError", "message": "..."}
```

Client re-raises as the identical exception type. Participant `try/except` works as expected.

### Error Types That Cross the Wire

- `BudgetExhaustedError`, `NoBudgetContextError`, `SymmetryError`
- `ValueError`, `TypeError` (invalid arguments)

### Errors That Don't Cross the Wire

- numpy internal errors → wrapped as generic `MechEstimServerError` with sanitized message (no numpy stack traces leak)

### Client-Side Errors (Never Hit Server)

- `AttributeError` for blacklisted/unknown functions

---

## Security

| Threat | Mitigation |
|---|---|
| Malformed msgpack | Validate message schema. Invalid → error response, no crash. |
| Unknown op name | Check against registry whitelist. Unknown → error. |
| Fake handle IDs | Lookup in ArrayStore. Missing → error response. |
| Request flooding | Bounded by FLOP budget. Once exhausted, all ops rejected. |
| Huge array creation | Configurable max array size. Reject if exceeded. |
| Multiple sessions | One session per connection. Second `budget_open` → error. |
| Code injection / pickle | msgpack only. No eval, no pickle, no code execution. |
| File system access | No file I/O ops in protocol. No file-access endpoints. |

**Key security property:** the protocol is a closed whitelist of operation names from the mechestim registry. No arbitrary code execution path exists.

---

## Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| ZeroMQ + msgpack over gRPC | Lower latency (~10-30us vs ~0.2ms), simpler setup, no codegen. Good enough for our single-client, single-server use case. |
| Opaque handles over full round-trips | Avoids sending large arrays back and forth for chained operations. Matches Ray's proven ObjectRef pattern. |
| Session-scoped GC over refcounting | Vastly simpler. Works because sessions are short-lived and bounded by FLOP budget. |
| Server-side authoritative budget | Adversarial participants can't tamper with budget enforcement. |
| Same Python package name | Enables true drop-in replacement with zero code changes for participants. |
| Auto-generated proxy stubs from registry | Keeps client in sync with server's function list. Single source of truth. |
| Monotonic handle IDs over UUIDs | Simpler, faster, more debuggable. Security comes from session isolation, not ID unpredictability. |

---

## Research References

- **Ray ObjectRef pattern**: Opaque handles with server-side data, proven at scale. Ray uses Pickle protocol 5 + shared memory for same-node zero-copy.
- **Ray session model**: Ownership-based reference counting. We simplified to session-scoped GC.
- **Transport benchmarks**: ZeroMQ IPC ~10-30us, gRPC ~0.2ms, shared memory sub-us. ZeroMQ is the sweet spot for our latency and complexity requirements.
- **Arrow Flight**: Best for large bulk transfers but pyarrow requires numpy (disqualified for client).
