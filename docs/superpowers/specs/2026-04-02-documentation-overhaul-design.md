# Documentation Overhaul Design

**Date:** 2026-04-02
**Status:** Draft
**Audience:** General library users (not competition-specific)

## Goal

Rewrite the mechestim documentation to match the style, structure, and visual
identity of the
[network-estimation-challenge-internal](https://github.com/AIcrowd/network-estimation-challenge-internal)
reference repo. The result is a persona-driven docs site with getting-started
guides, how-to pages, concept explanations, API reference, operation audit,
client-server architecture guide, troubleshooting, and runnable example scripts.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Structure | Diataxis (tutorials / how-to / concepts / reference) | Matches reference repo, proven pattern |
| Page template | Adapted from reference | "When to use this page", "Prerequisites", "Usage", "What you'll see", "Common pitfalls", "Related pages" |
| Emojis | Yes, in section headers | Match reference repo style |
| Installation | `uv` only (no `pip`) | Avoids NumPy version confusion; `uv` resolves from lock file |
| Examples | Inline snippets in docs + separate `examples/` directory. All examples use `uv run`. | Both quick understanding and deeper learning |
| API docs | Keep mkdocstrings approach | Single source of truth in code docstrings |
| Build tool | MkDocs + Material (existing) | Already configured |
| Theme/logo | Match reference repo visual identity | Consistent branding across the challenge ecosystem |
| Math advice | None — docs show API usage, not algorithmic guidance | Audience is researchers who know the math |

## Visual Identity

Adopt the visual identity from the reference repo:

- **Logo:** Use the network estimation challenge logo from
  `/circuit-estimation/logos/` (copy into `docs/assets/logo/`)
- **Theme colors:** Match the reference repo's MkDocs Material palette
- **mkdocs.yml theme block:** Update to match reference repo's styling
  (primary color, accent, logo path, favicon)
- **Custom CSS:** If the reference repo uses any custom styles, adopt them

## Site Navigation (mkdocs.yml)

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Your First Budget: getting-started/first-budget.md
  - How-To Guides:
    - Migrate from NumPy: how-to/migrate-from-numpy.md
    - Use Einsum: how-to/use-einsum.md
    - Exploit Symmetry: how-to/exploit-symmetry.md
    - Use Linear Algebra: how-to/use-linalg.md
    - Plan Your Budget: how-to/plan-your-budget.md
    - Debug Budget Overruns: how-to/debug-budget-overruns.md
  - Concepts:
    - FLOP Counting Model: concepts/flop-counting-model.md
    - Operation Categories: concepts/operation-categories.md
  - Architecture:
    - Client-Server Model: architecture/client-server.md
    - Running with Docker: architecture/docker.md
  - API Reference:
    - Counted Operations: api/counted-ops.md
    - Free Operations: api/free-ops.md
    - Budget: api/budget.md
    - FLOP Cost Query: api/flops.md
    - Errors: api/errors.md
  - Operation Audit: reference/operation-audit.md
  - Troubleshooting: troubleshooting/common-errors.md
  - Changelog: changelog.md
```

## Page Template

Every documentation page follows this adapted template:

```markdown
# Page Title

## When to use this page
One sentence: who is this for and when do they need it.

## Prerequisites
Links to pages the reader should have completed first.

## [Core content section — title varies by page]
The main instructional content with code examples.

## 🔍 What you'll see
Expected output, behavior, or result.

## ⚠️ Common pitfalls
Symptom → fix pairs in a scannable list.

## 📎 Related pages
- [Page](path) — one-line description of why it's related
```

Sections can be omitted when they don't apply (e.g., concept pages skip
"Prerequisites" and "What you'll see"). The template is a guide, not a
straitjacket.

## Page Specifications

### index.md — Documentation Home

Persona-driven entry points with emoji headers:

- 🚀 **I want to get started** → Installation, Your First Budget
- 🛠 **Something isn't working** → Troubleshooting, Error Reference
- 📈 **I want to optimize my FLOP usage** → How-To guides (einsum, symmetry, budget planning)
- 🧠 **I want to understand how FLOP counting works** → Concepts (FLOP model, operation categories)
- 🏗 **I want to understand the sandboxed architecture** → Client-Server Model, Docker

Includes a 4-line quick example and installation one-liner. Ends with a Full
Taxonomy section listing all pages grouped by category.

### getting-started/installation.md — Install and Verify

- Install as dependency: `uv add git+https://github.com/AIcrowd/mechestim.git`
- Install for development:
  ```bash
  git clone https://github.com/AIcrowd/mechestim.git
  cd mechestim
  uv sync --all-extras
  ```
- Verify: `uv run python -c "import mechestim as me; print(me.__version__)"`
- Expected outcome: version string printed
- Common pitfall: NumPy version mismatch (mechestim requires numpy >=2.1.0,<2.2.0)
- Next step: Your First Budget

### getting-started/first-budget.md — Your First Budget

- Complete 15-line script showing:
  - `import mechestim as me`
  - Create a BudgetContext with a 10M FLOP budget
  - Create arrays (free), run einsum (counted), apply ReLU (counted)
  - Print budget.summary()
- Show the full summary output with annotations explaining each line
- Run with: `uv run python examples/01_basic_usage.py`
- Common pitfall: forgetting the BudgetContext wrapper (NoBudgetContextError)

### how-to/migrate-from-numpy.md — Migrate from NumPy

- Side-by-side comparison: `import numpy as np` → `import mechestim as me`
- What changes:
  - All computation inside `BudgetContext`
  - Some ops unavailable (raises `AttributeError` with guidance)
  - FLOP budget enforced
- What stays the same:
  - All arrays are plain `numpy.ndarray`
  - Same function signatures for supported ops
  - Same broadcasting rules
- Table of common NumPy patterns → mechestim equivalents
- Common pitfall: using `np.linalg.svd` instead of `me.linalg.svd` (bypasses counting)

### how-to/use-einsum.md — Use Einsum

- Einsum as the core computation primitive
- Common subscript strings with cost formulas:
  - Matrix multiply: `'ij,jk->ik'` → m × j × k
  - Batched matmul: `'bij,bjk->bik'` → b × i × j × k
  - Outer product: `'i,j->ij'` → i × j
  - Trace: `'ii->'` → i
  - Bilinear form: `'ai,bi,ab->'` → a × b × i
- `me.dot` / `me.matmul` have equivalent einsum cost
- Common pitfall: unexpected cost from large intermediate dimensions

### how-to/exploit-symmetry.md — Exploit Symmetry Savings

- How same-object detection works: pass the same Python object multiple times
- Code example:
  - `me.einsum('ai,bi,ab->', x, x, A)` — x passed twice, cost halved
  - vs `me.einsum('ai,bi,ab->', x, y, A)` where `y = x.copy()` — no savings
- When symmetry applies (same `id()`) and when it doesn't (copies, slices)
- Symmetry validation: mechestim verifies the arrays are actually equal
- Cost savings calculation: `cost / symmetry_factor`
- Common pitfall: inadvertently breaking symmetry with `.copy()` or slicing

### how-to/use-linalg.md — Use Linear Algebra

Demonstrate all available linalg functions coherently:

- Currently available: `me.linalg.svd(A, k=...)` (truncated SVD)
- API signature and parameters
- Cost formula: m × n × k FLOPs
- Query cost: `me.flops.svd_cost(m, n, k)`
- Code example showing usage within a BudgetContext
- What happens when you call an unsupported linalg function (helpful error)
- Note: This page will grow as more linalg operations are added

### how-to/plan-your-budget.md — Plan Your Budget

- Cost query functions (no BudgetContext needed):
  - `me.flops.einsum_cost(subscripts, shapes)`
  - `me.flops.svd_cost(m, n, k)`
  - `me.flops.pointwise_cost(shape)` / `reduction_cost(shape)`
- Example: budget breakdown table for a multi-step computation
- Common pitfall: not accounting for reduction costs

### how-to/debug-budget-overruns.md — Debug Budget Overruns

- Reading the budget summary output
- Using `budget.op_log` to inspect individual operation costs
- Identifying the most expensive operations
- Focus: how to read the diagnostic output, not algorithmic advice

### concepts/flop-counting-model.md — FLOP Counting Model

- Why FLOPs instead of wall-clock time:
  - Deterministic, hardware-independent
  - Rewards algorithmic efficiency, not engineering optimization
  - Reproducible across machines
- How costs are computed:
  - Analytical formulas based on tensor shapes
  - Computed before execution (budget checked first)
  - No runtime measurement
- Cost formulas by category (summary table)
- FLOP multiplier: scaling costs for experimentation

### concepts/operation-categories.md — Operation Categories

- **Free operations (0 FLOPs):** Tensor creation, reshaping, indexing, random
  - Why free: no arithmetic computation, just memory layout
  - Full list with examples
- **Counted operations:** Unary, binary, reduction, einsum, SVD
  - Cost rule per category
  - Examples with concrete numbers
- **Unsupported operations:** What happens (AttributeError with guidance)

### architecture/client-server.md — Client-Server Model

High-level explanation of the client-server architecture:

- **Why it exists:** Sandboxed execution for competition submissions.
  Participant code runs in an isolated container that can only communicate
  with the mechestim server via a socket. This prevents participants from
  bypassing FLOP counting by importing NumPy directly.
- **How it works (high level):**
  - Server runs the real mechestim library and enforces budgets
  - Client is a drop-in replacement that proxies operations to the server
  - Communication uses msgpack over TCP
  - Arrays stay on the server; client holds lightweight RemoteArray handles
- **Architecture diagram:** ASCII showing client container → TCP → server container
- **API compatibility:** Client exposes the same `import mechestim as me` API.
  Code written for the local library works unchanged with the client.
- **When to use which:**
  - Local library (`src/mechestim/`): development, testing, research
  - Client-server (`mechestim-client/` + `mechestim-server/`): competition
    evaluation, sandboxed environments

### architecture/docker.md — Running with Docker

- How to run the client-server model locally with Docker
- Docker Compose setup (reference existing `docker/` directory)
- Running without Docker: start server manually, connect client
- Environment variables and configuration
- Verifying the connection works
- Common pitfall: port conflicts, container networking

### reference/operation-audit.md — Operation Audit

Structured presentation of the full operation registry:

- **Summary stats:**
  - 209 free operations (0 FLOPs)
  - 73 counted unary operations
  - 45 counted binary operations
  - 37 counted reduction operations
  - 22 counted custom operations (einsum, matmul, SVD, etc.)
  - 96 blacklisted operations (unsupported, with explanations)
- **Tables by category:** For each category, a table listing:
  - Operation name
  - Cost formula
  - NumPy equivalent
- **Blacklisted operations table:** What's not supported and why
- **How to read this page:** Explain that this is auto-generated from
  the operation registry (`_registry.py`) and reflects the current state
- Note: Consider generating this page from the registry at build time
  (script in `scripts/`) so it stays in sync automatically

### troubleshooting/common-errors.md — Common Errors

Symptom → Why → Fix format for each error:

- `BudgetExhaustedError` — operation would exceed budget
- `NoBudgetContextError` — operation called outside BudgetContext
- `AttributeError: module 'mechestim' has no attribute 'X'` — unsupported op
- `ValueError: NumPy version mismatch` — wrong NumPy version
- `ValueError: NaN or Inf in input` — invalid tensor values
- `NestedBudgetError` — BudgetContext inside another BudgetContext

### changelog.md — Keep existing, no changes needed.

## Examples Directory

Five runnable Python scripts in `examples/`. Each includes a header comment
explaining what it demonstrates. All are run with `uv run python examples/NN_name.py`.

### 01_basic_usage.py
- Import mechestim, create BudgetContext
- Create arrays, run basic ops (einsum, exp, sum)
- Print budget summary
- ~20 lines

### 02_einsum_patterns.py
- Demonstrate common einsum subscript patterns
- Show cost for each pattern with concrete shapes
- ~30 lines

### 03_symmetry_savings.py
- Show same-object einsum with cost savings
- Compare with non-symmetric version
- ~25 lines

### 04_svd_approximation.py
- Truncated SVD on a random matrix
- Show cost output
- ~25 lines

### 05_budget_planning.py
- Use cost query functions to plan a computation
- Build a budget breakdown table
- Execute the plan within budget
- ~35 lines

## Changes to Existing Files

### README.md
- Update install command to use `uv`
- Add "Full Documentation" link pointing to docs site
- Keep rest as-is (it's already good)

### mkdocs.yml
- Update `nav` to match new structure (including architecture section)
- Update theme to match reference repo visual identity (logo, colors)
- Add `admonition` and `pymdownx.details` extensions for collapsible sections

## Assets to Copy

- Logo files from reference repo (`/circuit-estimation/logos/`) → `docs/assets/logo/`
- Any custom CSS from reference repo

## What's NOT in scope

- Competition-specific submission guides (audience is general library users)
- Auto-generated API docs restructuring (mkdocstrings approach stays as-is)
- CI/CD for docs deployment
- Algorithmic/mathematical advice (audience knows the math)

## File Count

| Category | New files | Modified files |
|----------|-----------|----------------|
| Getting Started | 2 | 0 |
| How-To | 6 | 0 |
| Concepts | 2 | 0 |
| Architecture | 2 | 0 |
| Reference (audit) | 1 | 0 |
| Troubleshooting | 1 | 0 |
| Index | 0 | 1 (rewrite) |
| Examples | 5 | 0 |
| Config | 0 | 1 (mkdocs.yml) |
| README | 0 | 1 (minor update) |
| Assets (logo) | 1+ | 0 |
| **Total** | **~21** | **3** |
