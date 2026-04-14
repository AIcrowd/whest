# Running with Docker

## When to use this page

Use this page to run the client-server model locally, either with Docker Compose or manually.

## Prerequisites

- [Client-Server Model](./client-server.md) — understand why the architecture exists
- Docker and Docker Compose installed

## With Docker Compose

The `docker/` directory contains a ready-to-use setup:

```bash
cd docker
docker compose up --build
```

This starts two containers:

| Service | Image | Role |
|---------|-------|------|
| `backend` | `Dockerfile.server` | Runs whest server, listens on IPC socket |
| `participant` | `Dockerfile.participant-hardened` | Runs participant code with whest-client only |

The containers share an IPC socket volume for communication.

## Without Docker

From a source checkout, start both processes from the repository root so the
server can import the local `src/whest` package:

```bash
# Terminal 1: Start the server
PYTHONPATH=src:whest-server/src \
  uv run --with pyzmq --with msgpack \
  python -m whest_server --url ipc:///tmp/whest.sock
```

```bash
# Terminal 2: Run client code
export WHEST_SERVER_URL=ipc:///tmp/whest.sock
PYTHONPATH=whest-client/src \
  uv run --with pyzmq --with msgpack python your_script.py
```

For TCP (e.g., across machines):

```bash
# Server
PYTHONPATH=src:whest-server/src \
  uv run --with pyzmq --with msgpack \
  python -m whest_server --url tcp://0.0.0.0:15555

# Client
export WHEST_SERVER_URL=tcp://server-host:15555
PYTHONPATH=whest-client/src \
  uv run --with pyzmq --with msgpack python your_script.py
```

If you already have `whest-client` and `whest-server` installed into
separate environments, the shorter `cd ... && uv run ...` workflow also works.
The commands above are the reproducible source-checkout path.

## Time limit enforcement

Submissions run under two layers of time enforcement:

1. **In-library (cooperative):** `BudgetContext(wall_time_limit_s=N)` checks the deadline before and after each numpy call. When exceeded, it raises `TimeExhaustedError` with diagnostic info (which operation, elapsed time, configured limit). This is a UX feature — it gives participants a clean error message.

2. **Container-level (hard):** The Docker container enforces a kernel-level time limit via cgroups/rlimit. If the in-library check doesn't catch the overshoot (e.g., a single very long numpy call), the container delivers `SIGKILL`. This is the ultimate backstop — no Python code can escape it.

Signal-based preemption (`SIGALRM`) is deliberately not used because Python signal handlers cannot interrupt C extensions (numpy/LAPACK/BLAS), making them ineffective for exactly the operations where time limits matter most.

## ⚠️ Common pitfalls

**Symptom:** `Connection refused` or `timeout`

**Fix:** Ensure the server is running before starting the client. Check that `WHEST_SERVER_URL` matches the server's `--url` argument.

**Symptom:** Port conflict

**Fix:** Change the port in both the server `--url` and client `WHEST_SERVER_URL`.

## 📎 Related pages

- [Client-Server Model](./client-server.md) — architecture overview
- [Contributor Guide](../development/contributing.md) — local repo workflows
