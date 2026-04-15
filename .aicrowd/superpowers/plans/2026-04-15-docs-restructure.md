# Docs Information Architecture Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure whest docs from textbook-style to audience-focused, progressive-disclosure architecture — splitting long pages, adding missing entry points, unifying reference tables, and replacing the oversized "When to use" H2 with React-style "You will learn" bullets.

**Architecture:** Content-only changes to MDX files + meta.json navigation. No component code changes except extending the ApiReference to include weight data. All changes in `website/content/docs/` and `website/components/api-reference/`.

**Tech Stack:** Fumadocs MDX, meta.json navigation, ops.json data merging

**Spec:** `.aicrowd/superpowers/specs/2026-04-15-docs-restructure-design.md`

---

## File Map

```
CONTENT CHANGES (website/content/docs/):
  meta.json                           ← REWRITE (new sidebar structure)
  getting-started/
    installation.mdx                  ← MODIFY (fix entry pattern)
    first-budget.mdx                  ← REWRITE → quickstart.mdx (trim to essentials)
    competition.mdx                   ← CREATE (new competition guide)
    meta.json                         ← MODIFY (add competition, rename first-budget)
  guides/                             ← CREATE directory (replaces how-to/)
    migrate-from-numpy.mdx            ← MOVE from how-to/, fix entry pattern
    einsum.mdx                        ← MOVE from how-to/use-einsum.mdx, fix entry pattern
    symmetry.mdx                      ← CREATE (trimmed from exploit-symmetry, practical only)
    linalg.mdx                        ← MOVE from how-to/use-linalg.mdx, fix entry pattern
    fft.mdx                           ← MOVE from how-to/use-fft.mdx, fix entry pattern
    budget-planning.mdx               ← CREATE (merged plan-your-budget + debug-budget-overruns)
    meta.json                         ← CREATE
  understanding/                      ← CREATE directory (replaces concepts/ + explanation/)
    how-whest-works.mdx               ← CREATE (new core architecture page)
    flop-counting-model.mdx           ← MOVE from concepts/, trim, de-duplicate weights
    operation-categories.mdx          ← MOVE from concepts/, fix entry pattern
    symmetry-detection.mdx            ← CREATE (merged subgraph-symmetry + explorer)
    calibration.mdx                   ← CREATE (merged calibrate-weights + empirical-weights)
    meta.json                         ← CREATE
  infrastructure/                     ← RENAME from architecture/
    client-server.mdx                 ← MODIFY (scope to competition eval)
    docker.mdx                        ← MOVE, fix entry pattern
    meta.json                         ← MODIFY
  development/
    contributing.mdx                  ← MODIFY (add numpy-compat-testing content)
    meta.json                         ← MODIFY
  api/
    index.mdx                         ← KEEP (interactive ApiReference)
    meta.json                         ← KEEP
  changelog.mdx                       ← KEEP

DELETE:
  how-to/                             ← entire directory (replaced by guides/)
  concepts/                           ← entire directory (replaced by understanding/)
  explanation/                        ← entire directory (merged into understanding/)
  reference/                          ← entire directory (merged into API Reference)
  troubleshooting/                    ← merge common-errors into api/ or understanding/
  architecture/                       ← renamed to infrastructure/

DATA CHANGES:
  scripts/generate_api_docs.py        ← MODIFY (merge weights.json into ops.json)
  website/public/ops.json             ← REGENERATE (with weight field added)

COMPONENT CHANGES:
  website/components/api-reference/
    ApiReference.tsx                   ← MODIFY (add weight column to expandable detail)
    OperationRow.tsx                   ← MODIFY (show weight in detail view)
```

---

## Task 1: Update page entry pattern across all existing pages

This is a mechanical find-and-replace task. For every `.mdx` file, replace the `## When to use this page` H2 section with an italic context line + "You will learn" bullets.

**Files:** All 26 `.mdx` files in `website/content/docs/`

- [ ] **Step 1: Write a migration script**

Create `scripts/fix_entry_pattern.py` that:
1. Reads each `.mdx` file
2. Finds the `## When to use this page` section (H2 + the paragraph after it)
3. Replaces it with an italic line + "You will learn" bullets
4. The "You will learn" bullets should be derived from the H2 headings on the page (the first 3-5 main sections become bullet points)

The script should handle these patterns:
- `## When to use this page\n\nSingle sentence.` → `*Single sentence.*\n\n**You will learn:**\n- Bullet from H2 1\n- Bullet from H2 2\n- Bullet from H2 3`
- Pages without the "When to use" section → add the italic context + bullets from existing content

Run the script, then manually review and adjust the bullets for quality — the auto-generated ones from H2 headings will need human editing to be natural.

- [ ] **Step 2: Run the script**

```bash
uv run python scripts/fix_entry_pattern.py --dry-run
uv run python scripts/fix_entry_pattern.py
```

- [ ] **Step 3: Manually review and polish the "You will learn" bullets**

Go through each file and ensure the bullets are:
- Written as outcomes, not topics ("How to create a SymmetricTensor" not "SymmetricTensor")
- 3-5 bullets per page (not more)
- Natural language (not just heading text copied)

- [ ] **Step 4: Build and verify**

```bash
cd website && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add website/content/docs/ scripts/fix_entry_pattern.py
git commit -m "content: replace 'When to use' H2 with 'You will learn' bullets

React-style entry pattern: italic context line + 3-5 bullet points.
Applied across all 26 doc pages."
```

---

## Task 2: Create new sidebar structure + directories

Set up the new directory structure and meta.json navigation before moving content.

**Files:**
- Create: `website/content/docs/guides/meta.json`
- Create: `website/content/docs/understanding/meta.json`
- Create: `website/content/docs/infrastructure/meta.json`
- Modify: `website/content/docs/meta.json` (root)
- Modify: `website/content/docs/getting-started/meta.json`
- Modify: `website/content/docs/development/meta.json`

- [ ] **Step 1: Create new directories**

```bash
mkdir -p website/content/docs/guides
mkdir -p website/content/docs/understanding
mkdir -p website/content/docs/infrastructure
```

- [ ] **Step 2: Create meta.json for each new directory**

`website/content/docs/guides/meta.json`:
```json
{
  "title": "Guides",
  "pages": [
    "migrate-from-numpy",
    "einsum",
    "symmetry",
    "linalg",
    "fft",
    "budget-planning"
  ]
}
```

`website/content/docs/understanding/meta.json`:
```json
{
  "title": "Understanding whest",
  "pages": [
    "how-whest-works",
    "flop-counting-model",
    "operation-categories",
    "symmetry-detection",
    "calibration"
  ]
}
```

`website/content/docs/infrastructure/meta.json`:
```json
{
  "title": "Infrastructure",
  "pages": [
    "client-server",
    "docker"
  ]
}
```

- [ ] **Step 3: Update root meta.json**

Replace `website/content/docs/meta.json` with:
```json
{
  "title": "Documentation",
  "pages": [
    "getting-started",
    "guides",
    "understanding",
    "api",
    "infrastructure",
    "development",
    "changelog"
  ]
}
```

- [ ] **Step 4: Update getting-started/meta.json**

```json
{
  "title": "Getting Started",
  "pages": [
    "installation",
    "quickstart",
    "competition"
  ]
}
```

- [ ] **Step 5: Update development/meta.json**

```json
{
  "title": "Development",
  "pages": [
    "contributing"
  ]
}
```

- [ ] **Step 6: Commit**

```bash
git add website/content/docs/
git commit -m "chore: create new sidebar structure (guides, understanding, infrastructure)"
```

---

## Task 3: Move and rename existing pages

Move content files from old directories to new ones.

- [ ] **Step 1: Move how-to pages to guides/**

```bash
cp website/content/docs/how-to/migrate-from-numpy.mdx website/content/docs/guides/migrate-from-numpy.mdx
cp website/content/docs/how-to/use-einsum.mdx website/content/docs/guides/einsum.mdx
cp website/content/docs/how-to/use-linalg.mdx website/content/docs/guides/linalg.mdx
cp website/content/docs/how-to/use-fft.mdx website/content/docs/guides/fft.mdx
```

Update the `title` frontmatter in each copied file:
- `einsum.mdx`: title → "Einsum Patterns"
- `linalg.mdx`: title → "Linear Algebra"
- `fft.mdx`: title → "FFT Operations"

- [ ] **Step 2: Move concept pages to understanding/**

```bash
cp website/content/docs/concepts/flop-counting-model.mdx website/content/docs/understanding/flop-counting-model.mdx
cp website/content/docs/concepts/operation-categories.mdx website/content/docs/understanding/operation-categories.mdx
```

- [ ] **Step 3: Move architecture pages to infrastructure/**

```bash
cp website/content/docs/architecture/client-server.mdx website/content/docs/infrastructure/client-server.mdx
cp website/content/docs/architecture/docker.mdx website/content/docs/infrastructure/docker.mdx
```

Add a scoping note to the top of `infrastructure/client-server.mdx` (after frontmatter):

```
*This architecture is used for competition evaluation, where participant code runs in an isolated container that cannot access NumPy directly. For the core architecture of how whest wraps NumPy, see [How whest Works](/docs/understanding/how-whest-works).*
```

- [ ] **Step 4: Rename first-budget to quickstart**

```bash
cp website/content/docs/getting-started/first-budget.mdx website/content/docs/getting-started/quickstart.mdx
```

Update frontmatter title to "Quickstart". Trim the content to just the "Quickest possible start" section — remove BudgetContext decorator form, env var config, and wall-time-limit content (those move to competition.mdx in Task 4).

- [ ] **Step 5: Merge numpy-compat-testing into contributing**

Read `website/content/docs/concepts/numpy-compatibility-testing.mdx` and append its content as a new section at the end of `website/content/docs/development/contributing.mdx` under `## NumPy Compatibility Testing`.

- [ ] **Step 6: Build and verify**

```bash
cd website && npm run build
```

- [ ] **Step 7: Commit**

```bash
git add website/content/docs/
git commit -m "content: move pages to new directory structure (guides, understanding, infrastructure)"
```

---

## Task 4: Create new pages

**Files to create:**
- `website/content/docs/getting-started/competition.mdx`
- `website/content/docs/understanding/how-whest-works.mdx`
- `website/content/docs/guides/symmetry.mdx` (trimmed from exploit-symmetry)
- `website/content/docs/guides/budget-planning.mdx` (merged from 2 pages)
- `website/content/docs/understanding/symmetry-detection.mdx` (merged algorithm + explorer)
- `website/content/docs/understanding/calibration.mdx` (merged from 3 sources)

- [ ] **Step 1: Create competition.mdx**

A new page for competition participants consolidating scattered advice. Content should include:
- BudgetContext with `flop_budget` and `wall_time_limit_s`
- The decorator `@flop_budget` form
- Submission structure tips
- Common competition-specific pitfalls (from existing pages)
- Link to "Debug Budget Overruns" for detailed debugging

Target: ~700 words. Pull content from `first-budget.mdx` (the BudgetContext/decorator/env-var sections) and `for-agents.mdx` (rule #5 about wall_time_limit_s).

- [ ] **Step 2: Create how-whest-works.mdx**

A new page explaining the core architecture:
- "whest wraps NumPy" — `import whest as we` gives you a NumPy-compatible API
- How cost interception works — every operation call goes through `_counting_ops.py` which computes the analytical FLOP cost before delegating to real NumPy
- Budget tracking — how `BudgetContext` accumulates costs and enforces limits
- The registry — `_registry.py` maps every operation to its cost formula
- Free vs counted vs blocked at the code level

Target: ~800 words. Read `src/whest/__init__.py`, `src/whest/_counting_ops.py`, `src/whest/_budget.py`, and `src/whest/_registry.py` to write accurate descriptions. Include a simple flow diagram (Mermaid or text).

- [ ] **Step 3: Create symmetry.mdx (practical guide)**

Extract ONLY the practical content from `exploit-symmetry.mdx`:
- How to create a SymmetricTensor
- How to declare custom symmetries with PermutationGroup
- The automatic cost savings (with a before/after example)
- Common pitfalls

Target: ~800 words. Remove all algorithm details, propagation rules, and deep math — those go to symmetry-detection.mdx.

- [ ] **Step 4: Create budget-planning.mdx (merged)**

Merge `plan-your-budget.mdx` + `debug-budget-overruns.mdx` into one page:
- Part 1: How to estimate costs before running (cost query functions, einsum_path)
- Part 2: How to read budget summaries and diagnose overruns
- Part 3: What to do next (the 5 strategies added earlier)

Target: ~800 words.

- [ ] **Step 5: Create symmetry-detection.mdx (deep dive)**

Merge `subgraph-symmetry.mdx` + `symmetry-explorer.mdx` into one deep-dive page:
- Keep the TL;DR
- Keep the algorithm walkthrough
- Embed the `<SymmetryExplorer />` component
- Mark clearly as "contributor-level detail"

Target: ~2000 words. This is the one page where length is justified.

- [ ] **Step 6: Create calibration.mdx (merged)**

Merge `calibrate-weights.mdx` + relevant parts of `empirical-weights.mdx` + weight sections from `flop-counting-model.mdx`:
- Quick start (run calibration, get weights.json)
- How weights work (the formula, overhead subtraction)
- How to use weights (env var, programmatic loading)
- Methodology summary (no full results tables — those are in the API Reference)

Target: ~1000 words. De-duplicate from flop-counting-model which also explains weights.

- [ ] **Step 7: Build and verify**

```bash
cd website && npm run build
```

- [ ] **Step 8: Commit**

```bash
git add website/content/docs/
git commit -m "content: create 6 new focused pages (competition, architecture, symmetry, budget, calibration)"
```

---

## Task 5: Delete old directories and files

Remove the old directory structure that's been replaced.

- [ ] **Step 1: Delete old directories**

```bash
rm -rf website/content/docs/how-to/
rm -rf website/content/docs/concepts/
rm -rf website/content/docs/explanation/
rm -rf website/content/docs/reference/
rm -rf website/content/docs/troubleshooting/
rm -rf website/content/docs/architecture/
```

- [ ] **Step 2: Merge common-errors content**

Before deleting troubleshooting/, ensure the error content is preserved. Add a "Common Errors" section to `website/content/docs/understanding/how-whest-works.mdx` or create a brief errors page. The simplest approach: the errors are already documented in the API Reference's expandable rows (each operation shows its status/notes). For BudgetExhaustedError etc., add a brief section to the quickstart or competition guide.

- [ ] **Step 3: Fix any cross-references**

Search all remaining `.mdx` files for links to old paths (`/how-to/`, `/concepts/`, `/explanation/`, `/reference/`, `/troubleshooting/`, `/architecture/`) and update to new paths.

```bash
grep -r "how-to/\|concepts/\|explanation/\|reference/\|troubleshooting/\|architecture/" website/content/docs/ --include="*.mdx"
```

Fix each broken link.

- [ ] **Step 4: Build and verify**

```bash
cd website && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove old directory structure, fix cross-references"
```

---

## Task 6: Extend API Reference with weight data

**Files:**
- Modify: `scripts/generate_api_docs.py`
- Modify: `website/public/ops.json`
- Modify: `website/components/api-reference/OperationRow.tsx`

- [ ] **Step 1: Extend generate_api_docs.py to merge weights**

Read the existing `scripts/generate_api_docs.py`. Find where it generates `ops.json`. Add logic to:
1. Read `src/whest/data/weights.json`
2. For each operation in ops.json, add a `weight` field from weights.json (default to 1.0 if not found)

- [ ] **Step 2: Regenerate ops.json**

```bash
uv run python scripts/generate_api_docs.py
cp docs/ops.json website/public/ops.json  # or wherever the script outputs
```

Verify the new field exists:
```bash
python3 -c "import json; d=json.load(open('website/public/ops.json')); print(d['operations'][0])"
```

Expected: each operation now has a `weight` field.

- [ ] **Step 3: Show weight in OperationRow expandable detail**

In `website/components/api-reference/OperationRow.tsx`, add a "Weight" field to the detail grid:

```tsx
<div className={styles.detailItem}>
  <span className={styles.detailLabel}>Weight</span>
  <span>{op.weight !== 1.0 ? `${op.weight}×` : '1.0× (default)'}</span>
</div>
```

- [ ] **Step 4: Build and verify**

```bash
cd website && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ website/public/ops.json website/components/api-reference/
git commit -m "feat: add empirical weight data to API Reference

Merge weights.json into ops.json at build time. Show weight
multiplier in operation detail view."
```

---

## Task 7: Update llmstxt section config

The llmstxt post-build script has hardcoded section/slug mappings. Update them for the new paths.

**Files:**
- Modify: `website/scripts/generate-llmstxt.mjs`

- [ ] **Step 1: Update section config**

Read `website/scripts/generate-llmstxt.mjs` and update the `SECTIONS` object to match the new directory structure:
- `getting-started/installation` stays
- `getting-started/first-budget` → `getting-started/quickstart`
- Add `getting-started/competition`
- `how-to/*` → `guides/*` with new filenames
- `concepts/*` → `understanding/*`
- `explanation/*` → `understanding/symmetry-detection`
- `architecture/*` → `infrastructure/*`
- Remove `reference/*` entries (replaced by interactive API)
- Remove `troubleshooting/*`

- [ ] **Step 2: Build and verify**

```bash
cd website && npm run build
cat website/out/llms.txt | head -30
```

- [ ] **Step 3: Commit**

```bash
git add website/scripts/
git commit -m "fix: update llmstxt section config for new directory structure"
```

---

## Parallelization Guide

```
Task 1 (entry pattern) — FIRST, touches all files
  └─→ Task 2 (sidebar structure) — creates directories
       └─→ Task 3 (move pages) + Task 4 (create new pages) — parallel
            └─→ Task 5 (delete old dirs + fix links) — after both
                 └─→ Task 6 (API Reference weights) — independent
                 └─→ Task 7 (llmstxt) — independent
```

Tasks 6 and 7 can run any time after Task 5.
