# whest API Reference Autogeneration Design

## Problem

The current `/docs/api/` page is doing two jobs badly:

- as an index, it hides important information users need at a glance
- as a detail view, it relies on row expansion with low-value "Module Information"

Specific issues:

- `Module` is too literal (`numpy`, `numpy.linalg`, etc.) and not useful as a browsing aid
- `Cost Formula` is shown as plain text instead of rendered math
- `Weight` is buried in the expanded row instead of visible in the table
- the operation name is not the entry point to a real per-function reference page
- the expanded details are too thin to replace actual API docs

At the same time, the codebase supports 500+ operations, so manual page authoring will not scale. The docs need a generator-owned pipeline that produces NumPy-style standalone pages systematically.

## Key Decisions

1. Keep Fumadocs as the docs system and extend the existing Python generator instead of introducing Sphinx.
2. Use live `whest` signatures and docstrings as the primary API-doc content source.
3. Add a thin `whest` overlay for cost model, weight, aliases, normalized area, notes, and `whest`-owned examples.
4. Generate standalone pages only for supported operations.
5. Remove row expansion from `/docs/api/`; the index becomes a flat catalog and every operation name links to its standalone page.
6. Replace the literal module display with normalized area labels: `core`, `linalg`, `fft`, `random`, `stats`.
7. Show both `Area` and `Type` as color-coded metadata in the table.
8. Render cost formulas as LaTeX in both the index and the standalone page.
9. Maintain `whest` examples separately from inherited upstream examples.
10. Track example coverage internally via generated artifacts and CI, not in the public docs.
11. All references to `whest` operations in documentation should resolve through a unified linking mechanism and always render using the `we.<op>` form.

## Rejected Approaches

### Separate Sphinx / numpydoc pipeline

This is closest to how NumPy renders its API reference, but it introduces a second documentation stack, a second routing/story/navigation model, and more build complexity than the current repo needs.

### Single dynamic Next.js route backed only by JSON

This avoids hundreds of generated pages, but it requires building a custom renderer for structured docstring sections, references, math, and examples. That throws away Fumadocs' existing content model and makes the docs stack more bespoke.

## Current Constraints

- The site already uses Fumadocs with MDX under `website/content/docs/`.
- `scripts/generate_api_docs.py` already generates `website/public/ops.json`.
- KaTeX support is already enabled through `remark-math` and `rehype-katex`.
- The current workspace Python can drift from the supported NumPy version; in this workspace, importing `whest` under NumPy `1.24.4` already breaks docs-oriented introspection for some objects.
- The registry source of truth is `src/whest/_registry.py`, and empirical weight data already flows into `ops.json`.

The generation environment must therefore be pinned to the project-managed Python environment and fail hard if the imported package set is wrong.

## Information Architecture

### `/docs/api/`

Purpose: searchable catalog of all operations.

Behavior:

- no row expansion
- operation name is the primary link to the standalone operation page
- filters remain available for search and cost type
- area filter remains available, but uses normalized labels rather than raw module strings

Columns:

- `Operation`
- `Area`
- `Type`
- `Weight`
- `Cost Formula`

Presentation rules:

- `Area` is color coded
- `Type` is color coded
- `Weight` is visible directly in the table
- `Cost Formula` is rendered with LaTeX, not plain monospace text

### `/docs/api/ops/<slug>/`

Purpose: standalone per-operation reference page.

Page order:

1. header with operation name and signature
2. quick info section
3. dense `whest`-specific info section
4. inherited API docs rendered in NumPy style
5. `whest` examples

The standalone page replaces all detail previously shown in expanded rows.

### Blocked operations

Blocked / unsupported operations remain index-only and do not receive standalone pages in the first rollout.

## Canonical Operation Model

The generator should build a single canonical operation record for every supported operation.

Required fields:

- `name`: canonical operation name, e.g. `absolute`, `linalg.svd`
- `slug`: standalone-page slug
- `whest_ref`
- `numpy_ref`
- `area`: normalized area label (`core`, `linalg`, `fft`, `random`, `stats`)
- `type`: supported cost/type classification for display
- `category`: underlying registry category if still needed internally
- `weight`
- `cost_formula`
- `cost_formula_latex`
- `notes`
- `aliases`
- `signature`
- `summary`
- structured inherited doc sections
- `has_whest_examples`
- `example_count`
- `example_sources`

This model is the input to both the index and the standalone-page generator.

## Content Pipeline

The docs generator is split into two layers.

### Layer 1: inherited API extraction

Input:

- live imported `whest` objects from the pinned docs environment

Responsibilities:

- import each supported public callable
- extract the rendered signature
- extract the raw docstring
- parse structured docstring sections in a NumPy-aware way
- preserve sections such as `Parameters`, `Returns`, `Notes`, `See Also`, and inherited upstream `Examples`

Failure policy:

- if structured parsing fails for a function, preserve a raw docstring fallback instead of dropping content
- emit diagnostics for fallback cases

### Layer 2: `whest` overlay

Input:

- registry metadata
- cost metadata
- empirical weight data
- owned `whest` examples

Responsibilities:

- normalize area labels
- attach display type/category
- attach weight and LaTeX formula
- attach aliases and `whest` notes
- attach `whest`-owned examples
- compute example coverage metadata

The overlay remains intentionally thin. The inherited docstrings remain the main documentation body.

## Generated Outputs

The generator should emit four outputs from the same operation model.

### 1. Index manifest

`website/public/ops.json`

Used by `/docs/api/` to render:

- operation link
- normalized area
- display type
- weight
- LaTeX cost formula

### 2. Standalone operation pages

Generated operation pages under the API docs subtree:

- route: `/docs/api/ops/<slug>/`
- source location: generator-owned docs content under `website/content/docs/api/ops/`

These pages are generated before the website build, not hand-authored. The generator owns their structure and content assembly, and authors should treat the directory as generated source.

### 3. Example coverage artifact

Internal JSON artifact:

- `website/.generated/api-example-coverage.json`

Per-operation fields:

- `has_whest_examples`
- `has_inherited_examples`
- `example_count`
- `example_sources`
- `coverage_status`

### 4. CI summary report

Human-readable summary emitted during CI, with:

- supported operation count
- count with `whest` examples
- count without `whest` examples
- coverage percentage
- any regressions relative to the previous committed baseline or current branch state

### 5. Operation reference manifest

Generator-owned manifest used by docs authoring and linting to resolve operation links consistently.

Path:

- `website/.generated/op-refs.json`

Fields:

- canonical operation name
- alias names
- canonical slug
- canonical `we.*` label

## Example Ownership Model

Inherited upstream examples are seed material, not the final `whest` story.

Rules:

- inherited examples remain visible within the inherited API docs section when available
- `whest`-owned examples are maintained separately and rendered in the final section of the page
- `whest`-owned examples must include `whest`-specific context such as `BudgetContext`, FLOP behavior, or other usage details that matter in this project

### Storage strategy

`whest` examples should live as separate small source files rather than as inline blobs inside one giant metadata file.

Benefits:

- clean review diffs
- easy ownership and maintenance
- straightforward coverage detection by presence and validation
- easy future extension to richer MDX example blocks

## Routing And Slug Rules

- every supported canonical operation gets one canonical slug
- aliases do not get their own standalone pages in the first rollout
- alias operations in the index link to the canonical target page
- the standalone page renders aliases explicitly in quick info / dense `whest` info

Examples:

- `absolute` is canonical; `abs` links to the `absolute` page
- `arccos` is canonical; `acos` links to the `arccos` page

## Unified Operation References

All references to `whest` operations in documentation should resolve through a single canonical reference mechanism.

### Authoring format

Use an explicit custom MDX component as the canonical operation reference form:

- `<OpRef name="absolute" />`
- `<OpRef name="linalg.svd" />`

Rendering rules:

- the rendered text always uses the `whest` form, such as `we.absolute` or `we.linalg.svd`
- normal operation references do not support a free-form label override
- the component resolves aliases to canonical operation pages automatically

### Resolver behavior

The resolver should:

- map a canonical name or alias to the canonical operation record
- generate the correct `/docs/api/ops/<slug>/` URL
- provide a single source of truth for:
  - display label
  - canonical destination
  - alias normalization

The same resolver should be used by:

- generated API pages
- hand-written guides and concept pages
- tables, cards, and custom components that mention operations

### Quality gates

CI / docs checks should flag:

- raw unlinked `we.*` mentions in MDX where an `OpRef` should be used
- broken operation references caused by slug or canonical-name drift

## UI Rules

### Index table

- `Area` uses normalized labels, not raw implementation modules
- `Area` is visually de-emphasized compared with the operation name, but still color-coded for scanability
- `Type` is color-coded independently of `Area`
- `Cost Formula` uses rendered math

### Standalone page

Quick info should be dense and scan-friendly. It should surface the `whest`-specific facts users actually need before reading inherited API prose:

- area
- type
- weight
- `whest` reference
- NumPy reference
- aliases

The dense `whest`-specific section should then surface:

- rendered cost formula
- short notes about FLOP charging or other `whest` semantics
- any operation-specific `whest` constraints worth calling out

The inherited API docs then follow in NumPy style, and `whest` examples come last.

## Failure Modes And Handling

### Wrong docs environment

Risk:

- imports succeed against the wrong NumPy version and produce incorrect signatures or broken objects

Handling:

- generator runs only through `uv run`
- generator asserts that imported NumPy matches the supported range in the project
- mismatch is a hard failure

### Docstring parsing drift

Risk:

- unusual docstrings, C-level docstrings, or alias wrappers do not parse cleanly

Handling:

- tolerant parser with raw fallback
- diagnostics report listing fallback cases

### Alias / canonical-name conflicts

Risk:

- duplicate pages or ambiguous links

Handling:

- one canonical slug per operation family
- aliases rendered as metadata, not separate first-class pages

### Stale generation

Risk:

- `ops.json`, generated pages, and coverage artifacts drift from the source registry or examples

Handling:

- CI regenerates artifacts and fails on stale output

### Invalid owned examples

Risk:

- example files exist but are syntactically invalid or use incorrect `whest` patterns

Handling:

- validate example file presence and syntax in CI
- fail on invalid examples

## Testing Strategy

### Unit tests

Cover:

- slug generation
- area normalization
- alias canonicalization
- overlay merge logic
- docstring parser fallback behavior

### Integration tests

Cover:

- generated `ops.json` shape
- presence of new index fields
- generated standalone page structure for representative ops
- supported-op publication count

### Full docs verification

CI should:

- run the generator
- build the website
- fail on stale generated outputs
- emit example coverage summary

## Rollout Plan

### Phase 1: index foundation

- extend the operation model
- normalize area labels
- add `Weight` and LaTeX cost formula to the index manifest
- remove row expansion
- link operation names to future standalone pages

### Phase 2: standalone pages

- generate supported-operation pages
- render the approved page order
- wire routes and navigation under `/docs/api/ops/<slug>/`

### Phase 3: example coverage and CI

- introduce owned `whest` example source files
- emit coverage JSON/report
- add CI checks for stale generation and example validation
- enforce regressions first, not absolute coverage completeness

## Not In Scope

- public example-coverage badges or public coverage dashboards
- standalone pages for blocked operations
- introducing a separate Sphinx docs site
- full replacement of inherited upstream examples in the first rollout
- broad docs IA changes outside the API-reference area

## Final Recommendation

Implement a generator-owned Fumadocs pipeline that:

- upgrades `/docs/api/` into a dense linked catalog
- generates standalone pages for all supported operations
- reuses live `whest` signatures and docstrings for inherited API docs
- layers `whest` metadata and owned examples on top
- tracks `whest` example coverage internally through generated artifacts and CI

This keeps the existing docs stack, matches the scale of a 500+ operation surface, and creates a clear path from inherited API parity to richer `whest`-specific documentation over time.
