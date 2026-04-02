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
| `backend` | `Dockerfile.server` | Runs mechestim server, listens on IPC socket |
| `participant` | `Dockerfile.participant` | Runs participant code with mechestim-client only |

The containers share an IPC socket volume for communication.

## Without Docker

Start the server manually:

```bash
# Terminal 1: Start the server
cd mechestim-server
uv run python -m mechestim_server --url ipc:///tmp/mechestim.sock
```

```bash
# Terminal 2: Run client code
export MECHESTIM_SERVER_URL=ipc:///tmp/mechestim.sock
cd mechestim-client
uv run python your_script.py
```

For TCP (e.g., across machines):

```bash
# Server
uv run python -m mechestim_server --url tcp://0.0.0.0:15555

# Client
export MECHESTIM_SERVER_URL=tcp://server-host:15555
uv run python your_script.py
```

## ⚠️ Common pitfalls

**Symptom:** `Connection refused` or `timeout`

**Fix:** Ensure the server is running before starting the client. Check that `MECHESTIM_SERVER_URL` matches the server's `--url` argument.

**Symptom:** Port conflict

**Fix:** Change the port in both the server `--url` and client `MECHESTIM_SERVER_URL`.

## 📎 Related pages

- [Client-Server Model](./client-server.md) — architecture overview
