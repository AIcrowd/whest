# Client-Server Model

## When to use this page

Use this page to understand how mechestim's client-server architecture works and why it exists.

## Why client-server?

In competition evaluation, participant code runs in an **isolated container** that cannot import NumPy directly. This prevents participants from bypassing FLOP counting by calling NumPy functions outside mechestim.

The client-server model enforces this isolation:

```mermaid
graph LR
  subgraph participant ["Participant Container"]
    direction TB
    p1["import mechestim as me"]
    p2["No NumPy here!<br/>Only client proxy"]
  end

  subgraph server ["Server Container"]
    direction TB
    s1["mechestim library<br/>(real NumPy)"]
    s2["Budget enforcement<br/>Array storage<br/>FLOP counting"]
  end

  participant -- "ZMQ · IPC / TCP" --> server
  server -- "ZMQ · IPC / TCP" --> participant

  style participant fill:#f8f9fa,stroke:#4051b5,stroke-width:2px,color:#1a1a2e
  style server fill:#f8f9fa,stroke:#4051b5,stroke-width:2px,color:#1a1a2e
  style p1 fill:#fff,stroke:#e5e7eb,color:#1a1a2e
  style p2 fill:#fff,stroke:#e5e7eb,color:#555
  style s1 fill:#fff,stroke:#e5e7eb,color:#1a1a2e
  style s2 fill:#fff,stroke:#e5e7eb,color:#555
```

## How it works

1. **Server** runs the real mechestim library backed by NumPy. It stores all arrays, enforces budgets, and counts FLOPs.

2. **Client** is a drop-in replacement (`import mechestim as me`) that proxies every operation to the server over ZMQ (msgpack-encoded messages).

3. **Arrays stay on the server.** The client holds lightweight `RemoteArray` handles that reference server-side data. When you call `me.einsum(...)`, the client sends the operation and handle IDs to the server, which executes it and returns a new handle.

4. **Budget enforcement happens server-side.** The client cannot manipulate FLOP counts.

## Communication protocol

- **Transport:** ZMQ (REQ/REP pattern)
- **Serialization:** msgpack with binary-safe array payloads
- **Default endpoint:** `ipc:///tmp/mechestim.sock` (configurable via `MECHESTIM_SERVER_URL`)
- **Timeout:** 30 seconds per request

## API compatibility

Code written for the local library works unchanged with the client:

```python
# This code works with BOTH the local library and the client
import mechestim as me

with me.BudgetContext(flop_budget=10**6) as budget:
    x = me.zeros((256,))
    W = me.random.randn(256, 256)
    h = me.einsum('ij,j->i', W, x)
    print(budget.summary())
```

## When to use which

| Use case | Package | Install path |
|----------|---------|-------------|
| Development, testing, research | `mechestim` (local library) | `uv add git+...` or `uv sync` from repo |
| Competition evaluation, sandboxed environments | `mechestim-client` + `mechestim-server` | Docker containers |

## Three packages in this repo

| Package | Location | Description |
|---------|----------|-------------|
| `mechestim` | `src/mechestim/` | Local library — full NumPy backend, direct execution |
| `mechestim-client` | `mechestim-client/` | Client proxy — no NumPy dependency, forwards ops to server |
| `mechestim-server` | `mechestim-server/` | Server — runs real mechestim, manages sessions and arrays |

## 📎 Related pages

- [Running with Docker](./docker.md) — set up client-server locally
- [Your First Budget](../getting-started/first-budget.md) — getting started with the local library
