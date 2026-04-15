# API Reference NumPy-Parity Design

Date: 2026-04-15
Status: Draft for review
Scope: Per-operation reference pages only

## Goal

Bring `whest` per-operation reference pages to near-feature parity with NumPy reference pages, excluding plot rendering and any execute-in-browser functionality.

For each supported operation page, the user experience should feel like a real technical API reference page rather than a metadata card with placeholders. The page should be docstring-first, render a complete adapted reference body, and preserve `whest`-specific information in a compact overlay near the top.

## In Scope

- Supported per-operation pages under `/docs/api/ops/<slug>/`
- NumPy-style page hierarchy and information density
- Full inherited reference content where available:
  - title
  - signature
  - summary
  - `Parameters`
  - `Returns`
  - `See also`
  - `Notes`
  - `Examples`
- Automatic rewriting of obvious NumPy API references into `whest` references
- Subtle provenance on each page
- Source or show-source links on each page
- Previous and next navigation between operation pages
- One rendered example block per operation page
- Syntax-highlighted code rendering and rendered textual outputs
- Compact `whest` overlay near the top of each page
- Plain-link styling for operation links in the main API table

## Out of Scope

- Plot rendering
- Execute-in-browser features
- Unsupported or blocked operation pages
- Full API index parity with NumPy navigation and layout
- Left-nav or whole-site parity beyond per-op pages
- Pixel-perfect visual imitation of NumPy

## Current Problem

The current generated per-operation pages have the right route structure, but they do not behave like real API reference pages.

Current issues:

- The main title is rendered like inline code inside a pink block, which makes it feel like a code sample rather than a page title.
- The page visually prioritizes `whest` metadata cards over the actual reference content.
- The `API Docs` section is still a placeholder for many pages because live docstring extraction is not fully wired through.
- Example rendering uses raw HTML snippets rather than the site’s normal code block styling.
- The page structure is hard to scan over long reference content.
- Links in the main API table still read like inline-code pills instead of normal navigation links.

## Target User Experience

Each supported operation page should feel closer to `numpy.histogram` or `torch.histogram` than to a metadata dashboard.

The page should:

- open with a plain heading such as `we.histogram`
- show the callable signature directly below in monospace
- show a one-sentence summary adapted from the upstream docs
- include a subtle provenance line linking to the upstream NumPy page
- include source or show-source links
- present a compact `whest` overlay near the top
- render the inherited reference content as the main body of the page
- contain exactly one `Examples` section
- show syntax-highlighted code and textual outputs
- include previous and next navigation between operation pages

The resulting hierarchy should be:

1. title
2. signature
3. summary
4. subtle provenance
5. source or show-source links
6. compact `whest` overlay
7. inherited reference sections
8. one final `Examples` block
9. previous and next page navigation

## Page Contract

Each supported operation page should conform to this structure:

### Header

- Plain title: `we.histogram`
- Monospace signature directly below
- One-line summary adapted from NumPy
- Subtle provenance note such as `Adapted from NumPy docs`
- Source or show-source links where upstream resolution is available

### Compact Whest Overlay

This is a secondary block near the top of the page. It should be dense and easy to scan without overpowering the reference text.

Fields:

- area
- type
- weight
- cost formula
- NumPy ref
- whest-only note when relevant

### Main Reference Body

Rendered in NumPy-style order:

- `Parameters`
- `Returns`
- `See also`
- `Notes`
- `Examples`

If an upstream section is missing, it should simply not render rather than leaving placeholder copy in the final page.

### Examples

There must be exactly one `Examples` section per operation page.

Rules:

- We do not show both inherited examples and separate top-level `Whest Examples`.
- We do not require budget context in every operation page.
- The final rendered example should eventually be generated programmatically using the NumPy example as reference.
- The final example block should render code and textual output, but not plots.

### Footer Navigation

Each operation page should include previous and next navigation within the generated operation set.

Rules:

- navigation should use canonical operation pages
- labels should use `we.*`
- blocked or unsupported operations should not be part of this sequence

## Content Inheritance Strategy

The reference content should be inherited from live upstream NumPy docstrings and then adapted into `whest` form.

### Inherited Content

- signature shape
- summary line
- parameter documentation
- return documentation
- `See also`
- `Notes`
- example intent and structure

### Rewritten Content

The generator should automatically rewrite obvious library-specific references:

- `np.foo` → `we.foo`
- `numpy.foo` → `we.foo`
- `numpy.linalg.svd` → `we.linalg.svd`

This rewriting should also apply to links in:

- `See also`
- parameter prose where safe
- notes where the API reference is explicit

### Conservative Rewriting Policy

Explanatory prose should not be aggressively rewritten unless it clearly refers to the NumPy callable identity. Mathematical or conceptual explanations should remain mostly intact unless a `whest`-specific override exists.

The rule is:

- aggressive rewriting for API references and callable names
- conservative rewriting for explanatory prose

## Provenance Policy

Every adapted operation page should show subtle provenance near the top.

Requirements:

- visible but understated
- link to the upstream NumPy page
- not a large warning banner
- not hidden only in metadata

This preserves attribution and helps users verify the relationship between the `whest` page and the upstream source.

## Rendering And Styling Parity

The parity target is information hierarchy, not visual cloning.

### What Should Match NumPy

- plain title treatment
- signature below title
- summary-first page opening
- document-like reference rhythm
- strong section hierarchy
- technical readability over decorative card layouts
- code blocks that look like real reference examples

### What Should Stay Whest

- overall site branding
- compact whest-specific overlay
- internal canonical links to `we.*` pages
- whest-specific metadata fields such as weight and cost formula

### Styling Direction

- Remove the current code-pill look for the page title.
- Keep the signature in monospace, but visually secondary.
- Render `Parameters` and `Returns` in a structured reference layout, likely definition-list or two-column style.
- Render `See also` as real links.
- Render examples using the site’s proper code block component, with syntax highlighting and separate output blocks.
- Render main table operation names as normal text links instead of inline-code pills.

## Generator Architecture

The current pipeline still leans on opaque HTML fields such as `api_docs_html` and `whest_examples_html`. That is not the right long-term shape for parity.

The generator should move to a structured document model.

### Stage 1: Operation Resolution

For each supported operation:

- resolve canonical `whest` name
- resolve slug
- resolve upstream NumPy target
- resolve aliases
- attach `whest` metadata:
  - area
  - type
  - weight
  - cost formula
  - notes

### Stage 2: Doc Extraction

In the pinned docs environment:

- import the upstream object
- extract signature
- extract raw docstring
- parse the docstring into a structured model using a NumPy-aware parser

Structured output should include:

- summary
- parameters
- returns
- see_also
- notes
- examples
- warnings or references if present

### Stage 3: Whest Adaptation

Transform the structured model:

- rewrite NumPy callable references into `we.*`
- resolve supported operation references into canonical whest links
- attach subtle provenance
- select one final example block for the page

### Stage 4: Frontend Manifest Emission

Emit a richer per-operation manifest containing:

- header fields
- provenance fields
- source and show-source URLs
- `whest` overlay fields
- structured API sections
- one final example block
- previous and next page metadata

The frontend should render from this structured manifest, not from opaque HTML blobs.

## Frontend Rendering Architecture

The React page renderer should consume structured document data and intentionally render each section.

Why:

- section ordering is explicit
- styling is consistent
- link resolution is typed and testable
- code and output rendering can use real components
- edge-case overrides are easier to reason about

The key transition is:

- from: opaque HTML strings injected with `dangerouslySetInnerHTML`
- to: structured page data rendered by purpose-built React components

## Example Strategy

There should be one final example block per operation page.

### Source Layers

1. upstream NumPy example source
2. derived `whest` example
3. final rendered example

### Final Rendering Rule

- exactly one `Examples` section
- syntax-highlighted code
- textual output block when applicable
- no plots for this phase

### Coverage Model

Coverage should evolve from “do we have an owned example file” to “can the generator produce a valid final example for this operation.”

This keeps the public page model simple while still allowing internal coverage tracking and later programmatic generation.

## Link Rewriting Rules

Within adapted operation pages:

- supported operation references should always link to canonical whest operation pages
- displayed callable names should use `we.*`
- unsupported references may remain plain text or external upstream links

This should use the same canonical resolution rules already introduced for `<OpRef />`, but applied inside generated operation pages as well.

## Edge Cases

The design must account for:

- upstream objects with incomplete or oddly structured docstrings
- aliases that should not become independent pages
- `See also` entries that point to unsupported operations
- examples that depend on plots or notebook-only behavior
- examples that rely on NumPy conveniences that should be adapted for `whest`
- special upstream docs stored in non-standard locations

The generator should support targeted per-op overrides, but the default path should remain automatic.

## Error Handling And Fallbacks

The docs generator should fail loudly when the extraction environment is wrong, but degrade gracefully when individual operation pages are imperfect.

### Hard Failures

- wrong NumPy version in the docs environment
- missing required manifests
- broken canonical resolution for supported operations

### Soft Fallbacks

- missing optional sections in upstream docs
- temporarily incomplete example derivation for an operation
- unsupported `See also` targets

Soft fallbacks should never produce broken-looking UI. They should either omit a section or render a minimal fallback representation.

## Testing Strategy

### Generator Tests

- upstream target resolution
- structured doc extraction and parsing
- NumPy-to-whest API rewriting
- canonical link rewriting inside generated sections
- single-example selection behavior

### Frontend Tests

- page section order
- title and signature treatment
- correct rendering of structured sections
- syntax-highlighted example rendering
- output-block rendering
- absence of duplicate example sections

### Route-Level Verification

Representative pages should be verified end to end, such as:

- `absolute`
- `histogram`
- `linalg.svd`
- `fft.fft`

Checks should confirm:

- title is plain text, not code-pill styled
- full reference sections render
- the example section appears once
- supported internal links resolve correctly
- source or show-source links resolve correctly
- previous and next navigation resolve correctly

## First-Slice Delivery

The first implementation slice for parity should include:

- complete per-op page structure parity for supported operations
- structured inherited reference sections
- subtle provenance
- source or show-source links
- previous and next operation navigation
- one rendered example block per page
- syntax-highlighted code and textual outputs
- compact `whest` overlay
- plain-link styling for operation links in the main table

The first slice should explicitly defer:

- plot rendering
- unsupported-op parity pages
- full-site or full-index parity
- execute-in-browser functionality

## Recommendation

Adopt a structured doc-model pipeline that treats NumPy docstrings as the primary content source, rewrites explicit API references into `whest` form, renders one final example block per page, and makes the operation page reference-first rather than metadata-first.

This is the smallest architecture that can actually achieve durable NumPy parity for per-operation pages.
