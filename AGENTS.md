# Flopscope — repository guide for AI agents

This file orients automated coding agents working on this repository.
Read it before making non-trivial changes.

## What this library is

**Flopscope** is a NumPy-compatible math library that counts every FLOP
analytically against a configurable budget. It is the single product
this repository builds.

The Python public API is JAX-style:

```python
import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=1_000_000) as budget:
    a = fnp.array([1.0, 2.0, 3.0])
    b = fnp.einsum('i->', a)
    print(budget.summary())
```

- **`flopscope`** (top-level) — flopscope-specific primitives:
  `BudgetContext`, `configure`, `OpRecord`, `SymmetricTensor`,
  `PermutationGroup`, `FlopscopeArray`, errors, version metadata, plus
  the submodules `numpy`, `accounting`, `stats`.
- **`flopscope.numpy`** — counted numpy-shaped surface (`einsum`,
  `array`, `linalg`, `fft`, `random`, `testing`, `typing`, dtypes,
  constants). **Strict policy**: any name not implemented raises
  `AttributeError`. There is no transparent fallback to numpy —
  silently exposing uncounted ops would defeat FLOP accounting.
- **`flopscope.accounting`** — analytical cost helpers
  (`einsum_cost`, `pointwise_cost`, `reduction_cost`, …).
- **`flopscope.stats`** — statistical-distribution primitives
  (closer in spirit to scipy.stats than to a numpy submodule).

## Branding rules — the non-negotiables

The library was previously called **whest**. It is now **Flopscope**.
The rebrand is complete and intentional. Future agents should treat
the old name as if it never existed.

| Context | Form |
|---|---|
| Code identifier (Python, npm, file path, URL slug, env var, CSS class) | lowercase **`flopscope`** |
| Brand noun in user-facing prose, page titles, navigation labels | capitalized **`Flopscope`** |
| Wordmark (the rendered glyph) | always lowercase **`flopscope.`**, with `flop` in coral, `scope` in body ink, `.` in coral |

**Exception — `whestbench` is intentionally retained.** This is a
distinct sub-brand for the bench/estimator surface (color tokens
`--wb-*`). It is not a stale reference; do not rename it to
`flopscopebench`.

If you find a `whest` (case-insensitive) reference *anywhere else* in
the repo, it is a bug from an incomplete rebrand. Fix it.

## Repository layout

```
.
├── src/flopscope/            # Core Python library (the "flopscope" wheel)
│   ├── __init__.py           # Top-level: primitives only (no numpy ops)
│   ├── numpy/__init__.py     # JAX-style counted numpy surface
│   ├── numpy/{linalg,fft,random,testing,typing}/  # numpy submodules
│   ├── accounting.py         # Analytical cost helpers (was flops.py)
│   ├── stats/                # Statistical distributions (top-level)
│   ├── errors.py             # FlopscopeError, FlopscopeWarning, ...
│   ├── _registry.py          # Operation registry (single source of truth
│   │                         # for what flopscope implements)
│   └── data/                 # weights.json, default_weights.json
├── flopscope-client/         # Lightweight ZMQ client wheel (no numpy
│   └── src/flopscope/        # dep). Mirrors core's JAX-style layout:
│       ├── __init__.py       # primitives only
│       └── numpy/__init__.py # remote-proxy counted surface
├── flopscope-server/         # ZMQ server wheel that hosts the real
│   └── src/flopscope_server/ # flopscope library and serves clients.
├── tests/                    # pytest suite for the core library
├── examples/                 # Standalone usage examples
├── benchmarks/               # Op-level benchmark harness (perf calibration)
├── scripts/                  # generate_api_docs.py, sync_client.py,
│                             # numpy_audit.py, etc.
├── docker/                   # Container images for client/server
├── website/                  # Next.js + fumadocs documentation site
└── pyproject.toml            # name = "flopscope"
```

## Convention reminders

### Python imports in new code or examples

Use the JAX-style aliases consistently:

```python
import flopscope as flops          # primitives, errors, BudgetContext
import flopscope.numpy as fnp      # numpy-shaped counted ops
from flopscope import accounting   # cost helpers
```

Do not write `import flopscope as we` (legacy) or `import flopscope as me`.

### Strict no-fallback policy on `flopscope.numpy`

If a numpy name is unsupported, `flopscope.numpy.__getattr__` raises
`AttributeError`. Do not introduce a numpy fallback — silently
returning an uncounted op would defeat the library's purpose. If a
new name needs to be supported, add it to the registry and implement
it in the appropriate `_*` internal module.

### Env vars

All env vars are prefixed `FLOPSCOPE_`. Hard cut from the legacy
`WHEST_` prefix; no aliases.

### Wordmark markup

Anywhere the brand glyph appears, the markup is:

```html
<span class="flopscope-wordmark" aria-label="flopscope.">
  <span class="flopscope-wordmark__flop">flop</span>scope<span
    class="flopscope-wordmark__dot">.</span>
</span>
```

Existing call sites: `website/lib/layout.shared.tsx` (nav anchor),
`website/components/symmetry-aware-einsum-contractions/index.tsx`
(explorer hero),
`website/components/symmetry-aware-einsum-contractions/components/StickyBar.jsx`
(explorer sticky bar). All three reuse the same CSS contract — don't
fork the markup.

### API operation pages on the docs site

Match `numpy.org`'s reference layout:

1. H1 — full module path (`flopscope.numpy.einsum`); namespace muted,
   function name in coral.
2. Signature — short alias form (`fnp.einsum(...)`); param names
   italic, defaults coral, type hints stripped.
3. Brief summary, provenance, AREA / TYPE / NUMPY REF, COST,
   FLOPSCOPE CONTEXT (the flopscope-specific value-add band).
4. Untitled extended description prose (no `## Extended Summary`
   heading — suppressed in `OperationDocBody.tsx`).
5. Parameters / Returns / See also / Notes / Examples.

The renderer code lives in
`website/components/api-reference/` (`OperationDocPage.tsx`,
`OperationDocBody.tsx`, `OperationDocSignature.tsx`,
`parseSignature.ts`, `styles.module.css`). The signature parser
handles nested brackets and quoted type strings; if it can't parse
a signature it falls back to plaintext.

## Dev workflow cheat sheet

```bash
# Python core
uv venv && . .venv/bin/activate
uv pip install -e '.[dev]'
pytest                                # 3,300+ tests; 0 fail target

# Website
cd website && npm ci && npm run dev   # http://localhost:3000
cd website && npm test                # node test runner; 491+ tests

# Regenerate per-op API JSON after registry/docstring changes
python scripts/generate_api_docs.py

# Lint
ruff check . && ruff format --check .
```

## End-of-task checklist

Before committing brand-relevant changes, run:

```bash
grep -rinE '\bwhest\b' . --exclude-dir={.git,node_modules,__pycache__,.venv,.next,dist,build,.generated} \
    | grep -viE 'whestbench|AGENTS\.md'
```

Expected output: empty. The only `whest` references in the repo are
the deliberate `whestbench` sub-brand and this guide's didactic
mentions.
