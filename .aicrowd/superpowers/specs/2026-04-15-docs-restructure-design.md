# whest Docs Information Architecture Redesign

## Problem

The current docs are organized like a textbook — linear, exhaustive, and overwhelming. Pages try to cover too much (exploit-symmetry: 595 lines, 27 code blocks). The `## When to use this page` H2 heading wastes visual space. There's no progressive disclosure, no audience-specific entry points, and three overlapping static reference tables.

## Design Principles (learned from JAX, FastAPI, Polars, Rich)

1. **Time to First Wow** — 5 lines of code, visible FLOP output, within 60 seconds
2. **Three-tier architecture** — Tutorials (learn by doing) → Concepts (understand why) → Reference (look up)
3. **Code is the documentation** — show working code + output first, explain after
4. **One focused topic per page** — max ~200 lines, 3-5 minute read
5. **Progressive disclosure via structure** — sidebar hierarchy and page ordering, not UI widgets
6. **Bridge to familiar** — "it's NumPy, but every operation has a cost"

## Page Entry Pattern (all pages)

Replace the `## When to use this page` H2 with:

```mdx
*One sentence context in italic.*

**You will learn:**
- Bullet 1
- Bullet 2
- Bullet 3
```

No heading wasted. First real H2 is actual content.

## Sidebar Structure

```
Home (landing page)

Getting Started
  Installation                     ~150 words, 1 min
  Quickstart                       ~400 words, 2 min
  Competition Guide                ~700 words, 3 min  [NEW]

Guides
  Migrate from NumPy               ~400 words, 2 min
  Einsum Patterns                  ~800 words, 4 min
  Symmetry Savings                 ~800 words, 4 min  [TRIMMED from 3500]
  Linear Algebra                   ~500 words, 3 min
  FFT Operations                   ~500 words, 3 min
  Budget Planning & Debugging      ~800 words, 4 min  [MERGED from 2 pages]

Understanding whest
  How whest Works                  ~800 words, 4 min  [NEW — core architecture]
  The FLOP Counting Model          ~1200 words, 6 min [TRIMMED]
  Operation Categories             ~500 words, 3 min
  Symmetry Detection Deep Dive     ~2000 words, 10 min [MERGED algorithm + explorer]
  Calibration & Weights            ~1000 words, 5 min  [MERGED from 3 pages]

API Reference                      Interactive (unified ops + weights + costs)

Infrastructure
  Client-Server Model              ~500 words [SCOPED to competition eval]
  Docker Setup                     ~500 words

Development
  Contributing                     ~600 words
  NumPy Compatibility Testing      [MOVED from Concepts]

Cookbook                            [NEW — annotated examples from examples/]

Changelog
```

## What Changes

### Pages split (too long → focused)

| Current | Lines | Becomes |
|---------|-------|---------|
| exploit-symmetry.mdx (595) | Too long | `guides/symmetry.mdx` (~800 words practical) |
| subgraph-symmetry.mdx (551) | Algorithm | `understanding/symmetry-detection.mdx` (~2000 words, includes explorer) |
| calibrate-weights.mdx (338) | Split audiences | `understanding/calibration.mdx` (~1000 words, merged with empirical-weights) |
| flop-counting-model.mdx (334) | De-duplicate | Trimmed, weight system content goes to calibration page |

### Pages merged (related topics split across 2 pages)

| Current pages | Becomes |
|--------------|---------|
| plan-your-budget + debug-budget-overruns | `guides/budget-planning.mdx` |
| calibrate-weights + empirical-weights + weight sections of flop-counting-model | `understanding/calibration.mdx` |
| common-errors + api/errors | Single errors reference (in API Reference or standalone) |

### Pages removed (replaced by interactive API Reference)

| Current | Replacement |
|---------|-------------|
| reference/operation-audit.mdx (537 lines, 500-row static table) | Interactive API Reference |
| reference/cheat-sheet.mdx (330 lines, 238-row static table) | Interactive API Reference |
| reference/empirical-weights.mdx (data tables only) | Weight data merged into ops.json → API Reference |

### Pages added (missing content)

| New page | What it covers | Why |
|----------|---------------|-----|
| `getting-started/competition.mdx` | Budget limits, wall_time_limit_s, submission structure, competition-specific tips | 80% of users are competition participants with no dedicated entry point |
| `understanding/how-whest-works.mdx` | NumPy wrapping pattern, cost interception, budget tracking, the registry | Core architecture was missing — only client-server existed |
| `cookbook/` (multiple) | Annotated examples from `examples/` directory (11 scripts) | Examples exist but are invisible in docs |

### Unified API Reference

Extend the existing interactive ApiReference component:
- **Merge weight data** from `weights.json` into `ops.json` (add `weight` field per operation)
- **Add columns** to the component: weight value, full NumPy equivalent
- **Remove** the 3 static reference pages that duplicate this data
- The `generate_api_docs.py` script merges both JSON sources at build time

### Architecture section fix

- **Rename** "Architecture" → "Infrastructure" in sidebar
- **Add** "How whest Works" under "Understanding whest" — the primary architecture page explaining NumPy wrapping, cost interception, budget tracking
- **Scope** client-server page with a clear note: "This architecture is used for competition evaluation, where participant code runs in an isolated container"

## Page Count

| Current | Proposed |
|---------|----------|
| 25+ pages, many 400+ lines | ~22 pages, max ~200 lines each |
| 3 overlapping reference tables | 1 unified interactive reference |
| No competition guide | Dedicated competition entry point |
| No architecture overview | "How whest Works" page |
| 11 invisible example scripts | Cookbook section |

## Not in Scope

- Interactive code sandboxes (future sprint)
- Search improvements (Orama config is a separate issue)
- Dark mode polish (already addressed)
- Landing page redesign (already done)
