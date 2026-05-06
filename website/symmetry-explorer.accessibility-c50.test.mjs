// V3.1 §C50 — Accessibility Cross-Cutting Sweep (L5.T1.12).
//
// Source-grep tests covering the four audit-cited gaps closed in this
// migration step. The tests intentionally pin the smallest piece of source
// truth that proves each gap is closed, so they survive cosmetic edits.
//
//   Gap 1 — prefers-reduced-motion in OrbitRepMatrix
//   Gap 2 — OrbitDetailCard tap-outside dismiss (TODO removed)
//   Gap 3 — focusable canvas overlay + arrow-key navigation in OrbitRepMatrix
//   Gap 4 — aria-live region in TotalCostView announcing dimension/μ/α/total
//
// Companion plan:
//   .aicrowd/superpowers/plans/2026-05-01-l5-t1-c50-accessibility-cross-cutting.md

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const ORBIT_REP_MATRIX = 'components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx';
const ORBIT_DETAIL_CARD = 'components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx';
const TOTAL_COST_VIEW = 'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx';

// ─── Gap 1 ──────────────────────────────────────────────────────────────────

test('C50 Gap 1 — OrbitRepMatrix uses prefers-reduced-motion via matchMedia', () => {
  const src = read(ORBIT_REP_MATRIX);
  // matchMedia listener for the OS reduced-motion preference.
  assert.match(src, /matchMedia\(['"]\(prefers-reduced-motion: reduce\)['"]\)/);
  // A `useReducedMotion` hook (or local equivalent) is exposed and consumed.
  assert.match(src, /useReducedMotion/);
  // The flag gates at least one CSS transition (suppress under reduced motion).
  assert.match(src, /reducedMotion\s*\?\s*['"]none['"]/);
});

test('C50 Gap 1 — OrbitRepMatrix documents that the canvas paint pipeline does not animate', () => {
  const src = read(ORBIT_REP_MATRIX);
  // Inline comment confirming the canvas itself isn't animated — keeps the
  // reduced-motion responsibility scoped to the DOM overlay, not the canvas
  // draw loop.
  assert.match(src, /canvas[\s\S]{0,160}(does not animate|not animated|no animation|imperatively repaints)/i);
});

// ─── Gap 2 ──────────────────────────────────────────────────────────────────

test('C50 Gap 2 — OrbitDetailCard adds pointerdown listener for tap-outside dismiss', () => {
  const src = read(ORBIT_DETAIL_CARD);
  // pointerdown listener is registered (tap + mouse outside dismiss).
  assert.match(src, /addEventListener\(['"]pointerdown['"]/);
  assert.match(src, /removeEventListener\(['"]pointerdown['"]/);
  // The listener uses the cardRef to decide whether the target is outside.
  assert.match(src, /cardRef\.current/);
  // Calls onDismissRef.current() — the same dismiss surface as Esc.
  assert.match(src, /onDismissRef\.current\(\)/);
});

test('C50 Gap 2 — OrbitDetailCard line-30 mobile tap-dismiss TODO has been removed', () => {
  const src = read(ORBIT_DETAIL_CARD);
  // The original audit-cited TODO comment must not survive into the shipped file.
  assert.doesNotMatch(src, /TODO\(mobile\):\s*floating mode currently has no tap-outside dismiss/);
  // And the dismiss surface section should now mention pointerdown explicitly.
  assert.match(src, /pointerdown/);
});

// ─── Gap 3 ──────────────────────────────────────────────────────────────────

test('C50 Gap 3 — OrbitRepMatrix wraps canvas with role="grid" focusable overlay', () => {
  const src = read(ORBIT_REP_MATRIX);
  // The keyboard overlay is identified by a stable testid + ARIA role/grid contract.
  assert.match(src, /data-testid="orbit-rep-matrix-keyboard-overlay"/);
  assert.match(src, /role="grid"/);
  // Focusable: tabIndex={0}.
  assert.match(src, /tabIndex=\{0\}/);
  // Reflects matrix size to assistive tech.
  assert.match(src, /aria-rowcount=\{numRows\}/);
  assert.match(src, /aria-colcount=\{numCols\}/);
});

test('C50 Gap 3 — OrbitRepMatrix overlay onKeyDown handles all four arrow keys', () => {
  const src = read(ORBIT_REP_MATRIX);
  for (const key of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']) {
    assert.match(src, new RegExp(`['"]${key}['"]`), `arrow key ${key} not handled`);
  }
  // The handler is wired to the overlay div via onKeyDown.
  assert.match(src, /onKeyDown=\{handleKeyDown\}/);
});

test('C50 Gap 3 — OrbitRepMatrix overlay aria-label names the matrix and lists the keyboard shortcuts', () => {
  const src = read(ORBIT_REP_MATRIX);
  // aria-label must describe the matrix axes ("orbit O × representative Q")
  // and list the shortcuts (arrow keys / Enter / Escape) so screen-reader
  // users discover the keyboard model on first focus.
  assert.match(src, /aria-label="Output orbit O × representative Q matrix\..*arrow keys.*Enter.*Escape.*"/);
});

test('C50 Gap 3 — OrbitRepMatrix overlay handles Enter and Escape for pin/clear', () => {
  const src = read(ORBIT_REP_MATRIX);
  // Enter (and Space) pin the floating card to the focused cell via onHoverChange.
  assert.match(src, /case 'Enter'/);
  assert.match(src, /case ' '/);
  // Escape clears the pin (parent-controlled, so we forward null).
  assert.match(src, /case 'Escape'/);
  assert.match(src, /onHoverChange\(null\)/);
});

test('C50 Gap 3 — OrbitRepMatrix has an sr-only aria-live announcer for the focused cell', () => {
  const src = read(ORBIT_REP_MATRIX);
  // Off-screen polite live region surfaces the focused cell's coordinates +
  // incidence count to screen-reader users without taking visual real estate.
  assert.match(src, /data-testid="orbit-rep-matrix-focus-announcer"/);
  assert.match(src, /aria-live="polite"/);
  // Template includes "Row {n} column {n}" + an incidence count phrase.
  assert.match(src, /Row \$\{focusedCell\.row\} column \$\{focusedCell\.col\}/);
  assert.match(src, /incidence/);
});

// ─── Gap 4 ──────────────────────────────────────────────────────────────────

test('C50 Gap 4 — TotalCostView mounts an aria-live="polite" region for count state changes', () => {
  const src = read(TOTAL_COST_VIEW);
  // CountAnnouncer renders an off-screen aria-live region.
  assert.match(src, /data-testid="total-cost-aria-live"/);
  assert.match(src, /aria-live="polite"/);
  // sr-only / off-screen so the region never appears visually.
  assert.match(src, /className="sr-only"/);
  // Wired into TotalCostView with the four count props (dimensionN, mu,
  // alpha, total) so any of the four triggers an announcement update.
  assert.match(src, /<CountAnnouncer dimensionN=\{dimensionN\} mu=\{mu\} alpha=\{alpha\} total=\{totalCost\}/);
});

test('C50 Gap 4 — TotalCostView aria-live announcer template uses the canonical "Dimension updated to n=" prefix', () => {
  const src = read(TOTAL_COST_VIEW);
  // Verbatim template string (per accessibility-checklist.md §3 example).
  assert.match(src, /Dimension updated to n=\$\{t\.dimensionN\}/);
  assert.match(src, /Mu = /);
  assert.match(src, /alpha = /);
  assert.match(src, /total = /);
  // Debounced ~500ms via setTimeout so slider drags don't flood the SR.
  assert.match(src, /setTimeout\([\s\S]+?,\s*500\)/);
});

// ─── Closing — verify accessibility-checklist DoD remains intact ──────────

test('C50 — accessibility-checklist.md remains the canonical DoD reference', () => {
  // The audit-cited gaps reference the cross-cutting checklist, not a
  // per-component spec. This test asserts the checklist file is still
  // present so future edits keep the DoD source-of-truth path stable.
  const path = resolve(__dirname, '../.aicrowd/v3/accessibility-checklist.md');
  const md = readFileSync(path, 'utf-8');
  assert.match(md, /V3\.1 §C50/);
  assert.match(md, /pointerdown|tap-outside|tap-dismiss/i);
  assert.match(md, /aria-live/);
  assert.match(md, /prefers-reduced-motion/);
});
