# Implementation Plan: API Docs Coverage

**Spec:** `docs/superpowers/specs/2026-04-02-api-docs-coverage-design.md`

## Steps

### Step 1: Write `scripts/generate_api_docs.py`

The script reads `REGISTRY` from `mechestim._registry` and has two modes:

**Generate mode (default):**
- Group registry entries by module field
- Map modules to mechestim source modules:
  - `numpy` ops with `counted_*` categories → already in `counted-ops.md` via `_pointwise`/`_einsum`
  - `numpy` ops with `free` category → already in `free-ops.md` via `_free_ops`
  - `numpy.linalg` → generate `docs/api/linalg.md` with directives for `mechestim.linalg._svd`, `._decompositions`, `._solvers`, `._properties`, `._compound`, `._aliases`
  - `numpy.fft` → generate `docs/api/fft.md` with directives for `mechestim.fft._transforms`, `mechestim.fft._free`
  - `numpy.random` → generate `docs/api/random.md` with directive for `mechestim.random`
  - `mechestim._polynomial` → generate `docs/api/polynomial.md`
  - `mechestim._window` → generate `docs/api/window.md`
  - `mechestim._unwrap` → include in `counted-ops.md` or generate standalone
- Generate `docs/reference/operation-audit.md` from full registry

**Verify mode (`--verify`):**
- Build a set of all non-blacklisted op names from registry
- For each `docs/api/*.md`, extract `:::` directives, import each module, collect public function names
- Special handling: `random` uses `__getattr__`, so verify random ops by checking registry module == `numpy.random` and that `docs/api/random.md` exists with a `:::` directive for the random module
- Report any ops not covered, exit 1 if any missing

### Step 2: Update `mkdocs.yml` nav

Add new pages under API Reference:
- Linear Algebra: `api/linalg.md`
- FFT: `api/fft.md`
- Random: `api/random.md`
- Polynomial: `api/polynomial.md`
- Window Functions: `api/window.md`

### Step 3: Run the script to generate all pages

```bash
uv run python scripts/generate_api_docs.py
```

### Step 4: Run verify mode to confirm coverage

```bash
uv run python scripts/generate_api_docs.py --verify
```

### Step 5: Test mkdocs build

```bash
uv run --extra docs mkdocs build
```

### Step 6: Commit everything
