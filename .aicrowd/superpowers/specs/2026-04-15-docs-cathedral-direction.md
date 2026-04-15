# whest Docs Cathedral — Direction Spec

## The Problem

The Docusaurus migration is mechanically complete but the docs feel like a lateral move from MkDocs Material. The migration was infrastructure — now we need to build the experience that justifies React.

## The Vision

whest docs should be the best documentation site for a numerical computing library. Not because of volume of content, but because every concept is *interactive*. A reader never has to imagine what a FLOP cost looks like — they can see it, play with it, adjust dimensions and watch costs change in real time.

**The feeling:** "This library takes FLOPs seriously, and these docs take *me* seriously as a learner."

## Design Principles

1. **Interactive over static** — if a concept can be demonstrated interactively, it should be
2. **Docs-first design** — optimized for reading (16px text, generous spacing, 720px content width) not dashboard density
3. **Explorer-inspired** — same coral/gray palette, same fonts (Inter, Montserrat, IBM Plex Mono), but with breathing room
4. **Agent-first** — llms.txt/llms-full.txt for AI consumption, ops.json for programmatic access
5. **Cost formulas are evaluable client-side** — all 49 unique formulas are simple arithmetic; no server needed

## Phase 1: Visual Identity Overhaul

**Goal:** Make the docs feel like they belong to the same product as whestbench-explorer, but designed for reading.

### Design Tokens (explorer-derived, docs-adapted)

```
Colors:
  Primary:     #F0524D (coral)
  Gray-50:     #F8F9F9 (page background)
  Gray-100:    #F1F3F5 (code block background)
  Gray-200:    #D9DCDC (borders)
  Gray-400:    #AAACAD (secondary text, category labels)
  Gray-600:    #5D5F60 (body text secondary)
  Gray-900:    #292C2D (headings, primary text)
  White:       #FFFFFF (cards, panels, content area)
  Success:     #23B761
  Warning:     #FA9E33

Typography:
  Body:        Inter, 16px, line-height 1.75, color gray-900
  Headings:    Montserrat 700 (h1), Inter 600 (h2-h6)
  Code:        IBM Plex Mono, 14px
  Labels:      Inter 600, 11px, uppercase, 0.08em tracking, gray-400

Layout:
  Content width:     720px max
  Sidebar width:     260px
  Border radius:     8px (all components)
  Content padding:   32px
  Card padding:      20px
```

### What needs to change from current state

1. **Content area** — needs more whitespace, max-width 720px feels right for reading
2. **Sidebar** — current uppercase categories match explorer, keep it; but increase spacing between items
3. **Code blocks** — need gray-100 bg, 14px font, more padding (16px), rounded (8px)
4. **Tables** — current explorer-style headers are good but data cells need larger font (14px not 11px)
5. **Landing page** — needs complete redesign (see Phase 2)
6. **Footer** — minimal, just copyright line

## Phase 2: Three Flagship Interactive Components

### 2A. Live FLOP Sandbox (Landing Page)

**Where:** Replaces the current landing page entirely

**What:** A code editor with live FLOP cost visualization

```
┌─────────────────────────────────────────────────────────────┐
│  whest                                        [GitHub] [Docs]│
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NumPy-compatible math primitives with                       │
│  analytical FLOP counting                                    │
│                                                              │
│  ┌────────────────────────────────────────┐                  │
│  │  import whest as we                    │                  │
│  │                                        │                  │
│  │  A = we.random.randn([256], 256)       │   Budget         │
│  │  B = we.random.randn(256, 256)         │   ████████░░     │
│  │  C = we.einsum('ij,jk->ik', A, B)     │   196,608 FLOPs  │
│  │                                        │                  │
│  └────────────────────────────────────────┘   Operations     │
│                                               randn    131K  │
│  Dimensions: n=[256 ←──●──→]                  einsum    65K  │
│                                                              │
│  [Get Started]  [API Reference]  [Migration Guide]           │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  6 navigation cards (same as current)                        │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
- CodeMirror 6 for the editor (lightweight, supports Python syntax)
- Client-side cost engine: parse operation names from code, look up formula in ops.json, evaluate with dimensions from sliders
- Animated budget bar (CSS transition on width)
- Operation breakdown table (same style as Symmetry Explorer's cost view)
- Dimension slider updates costs in real time
- No server needed — purely client-side formula evaluation

**Cost engine approach:**
- Parse code line-by-line with regex to extract `we.operation_name(args)`
- Match operation name to ops.json entry
- Extract array dimensions from code (e.g., `randn(256, 256)` → m=256, n=256)
- Evaluate cost_formula with those dimensions
- Sum all costs, animate the budget bar

### 2B. Animated Einsum Playground

**Where:** `website/docs/explanation/einsum-playground.mdx` (new page in Explanation section)

**What:** Interactive einsum visualizer that shows tensor contractions step by step

**Implementation:**
- Reuse the Symmetry Explorer's `engine/` modules (algorithm.js, pipeline.js) — they already do the analysis
- New visualization layer that shows animated tensor blocks merging
- Step-through controls (play, pause, step forward/back)
- FLOP cost accumulator that ticks up at each step
- Compare optimal vs naive contraction paths side by side
- Connects to the existing Symmetry Explorer for deep-dive symmetry analysis

### 2C. Interactive Migration Guide

**Where:** Replaces or augments `website/docs/how-to/migrate-from-numpy.mdx`

**What:** Side-by-side code transformation with live FLOP annotations

**Implementation:**
- Two CodeMirror panels: NumPy (editable) → whest (auto-generated)
- Simple regex-based transformer: `np.` → `we.`, `numpy` → `whest`
- FLOP annotations injected as inline decorations on the whest side
- Blocked operations flagged with ⚠ and suggested alternatives
- Summary bar at bottom: operations migrated, blocked operations found, total FLOP cost
- Reuses the same cost engine from the Landing Page Sandbox

## Phase 3: Delight Layer

These are small (<30 min each) additions that compound into a polished experience:

1. **Operation cost tooltips** — any `we.function_name` in code blocks gets a hover tooltip showing the FLOP formula. Implemented as a Docusaurus plugin that post-processes code blocks.

2. **"Try in Sandbox" buttons** — code examples in tutorials get a button that opens the landing page sandbox pre-loaded with that code.

3. **Budget progress bar in tutorials** — the "Your First Budget" tutorial shows a visual budget bar that fills up as the reader follows along.

4. **Search** — Algolia DocSearch (free for OSS) or lunr-based local search.

5. **Keyboard navigation** — `/` for search, `n`/`p` for next/prev doc.

## Technical Dependencies

### Shared Cost Engine

The FLOP Sandbox, Einsum Playground, and Migration Guide all need the same cost calculation. Build it once as `website/src/lib/cost-engine.ts`:

```typescript
interface CostResult {
  operation: string;
  formula: string;
  flops: number;
  dimensions: Record<string, number>;
}

function evaluateCost(
  operationName: string,
  dimensions: Record<string, number>,
  opsData: OpsData
): CostResult;

function parseCode(code: string): ParsedOperation[];
function calculateBudget(operations: ParsedOperation[], opsData: OpsData): BudgetSummary;
```

### Shared Visualization Primitives

The budget bar, operation breakdown table, and dimension sliders appear in multiple places. Build them as shared components:

- `<BudgetBar current={n} total={n} />` — animated horizontal bar
- `<OperationBreakdown operations={[]} />` — table of operation costs
- `<DimensionSlider name="n" min={1} max={4096} value={n} onChange={fn} />`

## Execution Order (reprioritized — polish first, interactivity second)

### Sprint 1: "Make it look like someone cared" (PRIORITY)
1. **Visual Identity Overhaul** — complete CSS rewrite, 8px grid, transitions, refined code blocks, table polish, sidebar spacing, heading anchor hover effects (4-6 hours)
2. **Landing Page Redesign** — not just cards, a page that could only be whest's. Hero with personality, tighter layout, maybe the sandbox as a later addition (2-3 hours)
3. **Search** — Algolia DocSearch or lunr-based local search (1 hour)
4. **Micro-interactions** — hover states, smooth transitions, subtle animations (1-2 hours)

### Sprint 2: Interactive Flagships (after Sprint 1 is polished)
5. **Cost Engine** — shared library, tested independently (2-3 hours)
6. **Landing Page Sandbox** — if cost engine works well, embed on landing page (4-6 hours)
7. **Migration Guide** — reuses cost engine + CodeMirror (3-4 hours)

### Sprint 3: Deep Interactivity (lower priority)
8. **Einsum Playground** — complex, can wait (4-6 hours)
9. **Delight Layer** — tooltips, "try it" buttons, budget progress bars (1 hour per item)

## What's NOT in scope

- Dark mode (light-only, matching explorer)
- Versioned docs (not needed yet, single version)
- Blog (not a blog site)
- i18n (English only)
- Server-side execution of whest code (all client-side)
- Mobile-first design (desktop-first, mobile should work but isn't the priority)
