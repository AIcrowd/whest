# Mechestim Docker Setup

Docker configuration for running mechestim participant submissions in a hardened, sandboxed environment.

## Architecture

```
┌─────────────────────┐     IPC socket      ┌─────────────────────┐
│   participant        │ ◄────────────────► │   backend           │
│   (hardened image)   │   /ipc/mechestim   │   (server image)    │
│                      │      .sock          │                     │
│  - mechestim-client  │                     │  - mechestim lib    │
│  - pyzmq, msgpack    │                     │  - mechestim-server │
│  - stripped stdlib   │                     │  - numpy            │
│  - no shell, no net  │                     │                     │
└─────────────────────┘                     └─────────────────────┘
```

**Participant container** runs untrusted submission code with a locked-down Python environment. All computation is dispatched to the backend via ZMQ over a Unix domain socket.

**Backend container** runs the real mechestim server with NumPy, executing operations and counting FLOPs.

## Quick Start

```bash
cd docker

# Build both images
/usr/local/bin/docker compose build

# Place your submission in submissions/run.py
cp submissions/run_basic.py submissions/run.py

# Run
/usr/local/bin/docker compose down -v && /usr/local/bin/docker compose up --abort-on-container-exit
```

## Test Submissions

Three test scripts are included in `submissions/`:

| File | Purpose |
|------|---------|
| `run_basic.py` | Smoke test — runs mechestim API (einsum, maximum, sum) with a FLOP budget |
| `run_adversarial.py` | 9 tests verifying all defense layers (filesystem, open restriction, network, shell) |
| `run_introspection.py` | 6 tests for Python introspection attacks (__subclasses__, eval+import, os.system) |

Run any test by copying it to `submissions/run.py`:

```bash
cp submissions/run_adversarial.py submissions/run.py
/usr/local/bin/docker compose down -v && /usr/local/bin/docker compose up --abort-on-container-exit
```

## Defense Layers

The hardened participant image uses four independent defense layers:

### 1. Filesystem Stripping
The Python stdlib is stripped from ~2500 files down to ~450, keeping only modules needed by mechestim-client and pyzmq. Dangerous modules like `ctypes`, `numpy`, and `tkinter` are deleted at build time.

### 2. open() Restriction
`builtins.open` is replaced with a wrapper that only allows read-only access to:
- `/submission/` (participant's own code)
- Python's stdlib and site-packages paths

All writes and reads to other paths (e.g., `/etc/passwd`) are blocked.

### 3. No Shell / No Binaries
All system binaries (`/bin/*`, `/usr/bin/*`, `/sbin/*`) except `python3` are removed from the image. `os.system()` and `subprocess.run()` have nothing to execute.

### 4. Docker-Level Hardening
Applied via `docker-compose.yml`:

| Flag | Effect |
|------|--------|
| `network_mode: none` | No network access whatsoever |
| `read_only: true` | Immutable container filesystem |
| `cap_drop: ALL` | No Linux capabilities |
| `no-new-privileges` | No privilege escalation |
| `pids_limit: 50` | Prevents fork bombs |
| `mem_limit: 256m` | Memory cap |
| `cpus: 1.0` | CPU cap |
| `tmpfs: /tmp:size=10m` | Tiny writable area |

## What Participants Can Do

- Use the full mechestim API (`import mechestim as me`)
- `print()` for debugging and budget summaries
- Basic Python: loops, lists, dicts, classes, generators, comprehensions
- `itertools`, `functools`, `collections`, `json`, `time`, `contextlib`
- Read their own files from `/submission/`

## What Participants Cannot Do

- Install packages (no pip, no shell, no compiler)
- Use numpy (not installed)
- Call native code (ctypes stripped)
- Access the network (network_mode: none)
- Spawn processes (no shell binaries in image)
- Write files (read-only filesystem)
- Read system files (open() restricted)

## Files

```
docker/
  docker-compose.yml                # Orchestration with hardening flags
  Dockerfile.participant-hardened   # Multi-stage hardened build
  Dockerfile.server                 # Server image (unchanged, trusted)
  Dockerfile.participant            # Original unhardened image (for debugging)
  entrypoint.py                     # Python entrypoint (no shell needed)
  entrypoint.sh                     # Original shell entrypoint (for debugging)
  lockdown/
    allowlist.py                    # Canonical list of allowed stdlib modules
    strip_stdlib.py                 # Build-time script: delete non-allowed stdlib
    sitecustomize.py                # Runtime hook: restrict open(), disable breakpoint
    generate_constants.py           # Extract math constants at build time
    collect_libs.py                 # Trace and collect needed shared libraries
  submissions/
    run_basic.py                    # Smoke test
    run_adversarial.py              # Defense layer tests (9 tests)
    run_introspection.py            # Introspection attack tests (6 tests)
```

## Development / Debugging

If something breaks in the hardened image, debug using the original unhardened setup:

1. Edit `docker-compose.yml` to use `Dockerfile.participant` instead of `Dockerfile.participant-hardened`
2. Remove the hardening flags (`network_mode`, `read_only`, etc.)
3. The same `submissions/run.py` will run with full Python available

## Design Decisions

**Why not disable exec/eval/compile?**
Python's own stdlib depends on them — `collections.namedtuple` calls `eval()`, `dataclasses` uses `compile()`, the import system uses `exec()`. Disabling them breaks Python itself. Instead, we rely on filesystem stripping and Docker hardening to make exec/eval harmless.

**Why is math/subprocess/pickle available?**
pyzmq has deep transitive import chains: `zmq` -> `platform` -> `subprocess` -> `selectors` -> `math`. These modules are kept by the stripping script because they're needed at import time. They're harmless in the locked-down container: `math` only has scalar operations (can't bypass matrix FLOP counting), `subprocess` has no shell to run, `pickle` has no untrusted data to deserialize.

**Why 126MB instead of 30MB?**
The `debian:trixie-slim` base is ~80MB (needed for glibc compatibility with the Python 3.10 binary). The remaining ~46MB is the Python runtime + stripped stdlib + site-packages. Further reduction would require building Python from source with a static binary.
