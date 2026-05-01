import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import {
  tupleKey,
  compactTuple,
  labelledTuple,
  layoutFor,
  cellAtPoint,
  derivePreReps,
  deriveCells,
} from './components/symmetry-aware-einsum-contractions/components/branchingViews/orbitRepMatrixLayout.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('tupleKey is a stable JSON form', () => {
  assert.equal(tupleKey({ i: 0, j: 1 }), '{"i":0,"j":1}');
  assert.equal(tupleKey({ j: 1, i: 0 }), '{"j":1,"i":0}');
  assert.equal(tupleKey(null), '{}');
});

test('compactTuple drops keys, joins with commas', () => {
  assert.equal(compactTuple({ i: 0, j: 0, k: 1 }), '(0, 0, 1)');
  assert.equal(compactTuple(null), '—');
});

test('labelledTuple keeps k=v form', () => {
  assert.equal(labelledTuple({ i: 0, j: 0, k: 1 }), '(i=0, j=0, k=1)');
});

test('layoutFor: fixed canvas dimensions; cell width adapts to numCols, cell height to numRows', () => {
  // 6 × 6 (square): cellWidth = floor(360/6) = 60, cellHeight = floor(360/6) = 60.
  assert.deepEqual(layoutFor({ canvasWidth: 360, canvasHeight: 360, numRows: 6, numCols: 6 }), {
    cellWidth: 60, cellHeight: 60,
    canvasW: 360, canvasH: 360,
    contentWidth: 360, contentHeight: 360,
  });
  // 14 × 10: cellWidth = floor(360/14) = 25, cellHeight = floor(360/10) = 36.
  assert.deepEqual(layoutFor({ canvasWidth: 360, canvasHeight: 360, numRows: 10, numCols: 14 }), {
    cellWidth: 25, cellHeight: 36,
    canvasW: 350, canvasH: 360,
    contentWidth: 350, contentHeight: 360,
  });
  // 165 × 10 (Trilinear, very tall): cellWidth = 36, cellHeight = floor(360/165) = 2.
  // Canvas dimensions are fixed; cells become rectangles (wide-but-short rows).
  const trilinear = layoutFor({ canvasWidth: 360, canvasHeight: 360, numRows: 165, numCols: 10 });
  assert.equal(trilinear.cellWidth, 36);
  assert.equal(trilinear.cellHeight, 2);
  assert.equal(trilinear.canvasW, 360);
  assert.equal(trilinear.canvasH, 330); // 165 × 2
});

test('layoutFor floors at MIN_CELL=1 (no zero-size cells)', () => {
  // 200 cols × 360 width → 1.8 px per cell → floored to 1.
  const tiny = layoutFor({ canvasWidth: 360, canvasHeight: 360, numRows: 50, numCols: 200 });
  assert.equal(tiny.cellWidth, 1);
  assert.equal(tiny.canvasW, 200);
});

test('cellAtPoint maps pixel coords to (row, col)', () => {
  const layout = { cellWidth: 20, cellHeight: 20, canvasW: 200, canvasH: 200 };
  assert.deepEqual(cellAtPoint({ x: 0, y: 0 }, layout), { row: 0, col: 0 });
  assert.deepEqual(cellAtPoint({ x: 25, y: 5 }, layout), { row: 0, col: 1 });
  assert.deepEqual(cellAtPoint({ x: 150, y: 80 }, layout), { row: 4, col: 7 });
  assert.equal(cellAtPoint({ x: -5, y: 0 }, layout), null);
  assert.equal(cellAtPoint({ x: 250, y: 0 }, layout), null);
  assert.equal(cellAtPoint({ x: 0, y: 250 }, layout), null);
});

test('cellAtPoint handles rectangular cells (cellWidth ≠ cellHeight)', () => {
  // Rectangular cells: 36 wide × 2 tall.
  const layout = { cellWidth: 36, cellHeight: 2, canvasW: 360, canvasH: 330 };
  assert.deepEqual(cellAtPoint({ x: 5, y: 1 }, layout), { row: 0, col: 0 });
  assert.deepEqual(cellAtPoint({ x: 38, y: 1 }, layout), { row: 0, col: 1 });
  assert.deepEqual(cellAtPoint({ x: 5, y: 100 }, layout), { row: 50, col: 0 });
});

test('derivePreReps + deriveCells pull stored reps + filled-coeff matrix', () => {
  const orbitRows = [
    { repTuple: { i: 0, j: 0, k: 0 }, outputs: [{ outTuple: { i: 0, j: 0 }, coeff: 1 }], orbitSize: 1 },
    { repTuple: { i: 0, j: 0, k: 1 }, outputs: [
      { outTuple: { i: 0, j: 0 }, coeff: 1 },
      { outTuple: { i: 0, j: 1 }, coeff: 2 },
    ], orbitSize: 3 },
  ];
  const reps = derivePreReps(orbitRows);
  assert.equal(reps.length, 2);
  assert.deepEqual(reps[0].tuple, { i: 0, j: 0 });
  assert.deepEqual(reps[1].tuple, { i: 0, j: 1 });

  const cells = deriveCells(orbitRows, reps);
  assert.deepEqual(cells, [[1, null], [1, 2]]);
});

test('module is JS only — no JSX, no React import', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/orbitRepMatrixLayout.js');
  assert.doesNotMatch(src, /from 'react'/);
  assert.doesNotMatch(src, /<[A-Z]/);
});

test('cellAtPoint rejects coords beyond numRows / numCols', () => {
  const layout = { cellWidth: 20, cellHeight: 20, canvasW: 200, canvasH: 200, numRows: 5, numCols: 4 };
  // valid in-bounds
  assert.deepEqual(cellAtPoint({ x: 5, y: 5 }, layout), { row: 0, col: 0 });
  assert.deepEqual(cellAtPoint({ x: 75, y: 95 }, layout), { row: 4, col: 3 });
  // col >= numCols (canvas extends past content)
  assert.equal(cellAtPoint({ x: 95, y: 5 }, { ...layout, canvasW: 300 }), null);
  // row >= numRows
  assert.equal(cellAtPoint({ x: 5, y: 105 }, { ...layout, canvasH: 300 }), null);
});

test('cellAtPoint short-circuits when cell dimensions are invalid', () => {
  assert.equal(cellAtPoint({ x: 0, y: 0 }, { cellWidth: 0, cellHeight: 0, canvasW: 0, canvasH: 0 }), null);
  assert.equal(cellAtPoint({ x: 0, y: 0 }, { cellWidth: null, cellHeight: 20, canvasW: 100, canvasH: 100 }), null);
  assert.equal(cellAtPoint({ x: 0, y: 0 }, { cellWidth: 20, cellHeight: null, canvasW: 100, canvasH: 100 }), null);
});

test('layoutFor returns a clean empty layout for 0 rows / 0 cols / NaN / negative', () => {
  for (const bad of [{ numRows: 0, numCols: 5 }, { numRows: 5, numCols: 0 }, { numRows: NaN, numCols: 5 }, { numRows: 5, numCols: -3 }]) {
    const out = layoutFor({ canvasWidth: 360, canvasHeight: 360, ...bad });
    assert.equal(out.cellWidth, 0, `cellWidth 0 for ${JSON.stringify(bad)}`);
    assert.equal(out.cellHeight, 0);
    assert.equal(out.canvasW, 0);
    assert.equal(out.canvasH, 0);
    assert.equal(out.contentWidth, 0);
    assert.equal(out.contentHeight, 0);
  }
});

test('OrbitRepMatrix renders into <canvas> using layout helpers', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Imports the layout module (no inline duplication of the math).
  assert.match(src, /from '\.\/orbitRepMatrixLayout\.js'/);
  // Uses <canvas> + getContext('2d') for the grid.
  assert.match(src, /<canvas\b/);
  assert.match(src, /getContext\('2d'\)/);
  // Hit testing through cellAtPoint, not DOM events on per-cell <td>.
  assert.match(src, /cellAtPoint/);
});

test('OrbitRepMatrix uses a ref for hover (no React render per cell), controlled hover prop', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Hover lives in a ref so mousemove paints the canvas without re-rendering React.
  assert.match(src, /hoverRef\s*=\s*useRef/);
  assert.match(src, /hoverRef\.current/);
  // hover is now a controlled prop (not internal state) — parent owns it.
  assert.doesNotMatch(src, /useState[^;]*hover/);
  // No click pin handler.
  assert.doesNotMatch(src, /onClick=\{handleClick\}/);
});

test('OrbitRepMatrix uses onHoverChange + hover contract (no tooltipRef, no click pin)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /onHoverChange/);
  assert.match(src, /hover\s*=\s*null/); // controlled prop default
  // No more click-pin wiring.
  assert.doesNotMatch(src, /onPin/);
  assert.doesNotMatch(src, /onClick=\{handleClick\}/);
  // No more imperative tooltip ref.
  assert.doesNotMatch(src, /tooltipRef/);
});

test('OrbitRepMatrix passes numRows + numCols to cellAtPoint for bounds rejection', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // The layout object passed into cellAtPoint must include numRows/numCols
  // (Task 2's bounds-check guard).
  assert.match(src, /numRows[\s\S]{0,200}cellAtPoint|cellAtPoint[\s\S]{0,200}numRows/);
});

test('OrbitRepMatrix uses no raw hex outside design tokens', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  const allowed = new Set([
    '#F0524D', '#FEF2F1', '#D9DCDC', '#F8F9F9', '#FFFFFF', '#1F2526',
    '#ECEFEF', '#F4F6F6', '#4A7CFF', '#64748B', '#9AA0A0', '#B23E3A',
  ]);
  const hexes = src.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  for (const h of hexes) {
    assert.ok(allowed.has(h.toUpperCase()), `disallowed hex ${h} — use a design token`);
  }
});

test('OrbitRepMatrix renders permanent axis labels Orbit O and Rep Q', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Look for "Orbit" + math O and "Rep" + math Q in axis label slots.
  assert.match(src, /Orbit/);
  assert.match(src, /Rep/);
  assert.match(src, /writingMode:\s*'vertical-rl'/);
  assert.match(src, /import Latex from/);
});

test('OrbitRepMatrix renders the label-legend', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /data-testid="orbit-rep-matrix-legend"/);
  // Uses the W/einstein-summation token as the legend's default text color.
  assert.match(src, /summed:\s*'var\(--ein-w\)'/);
  assert.match(src, /style=\{\{ color: COLOR\.summed \}\}/);
  // Surfaces n.
  assert.match(src, /n\s*=/);
});

test('OrbitRepMatrix uses a faint cell-level hover marker (no row/col wash, no sticky strip)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Hover marker exists.
  assert.match(src, /hoverMarker/);
  // Pinned cell uses the strong coral fill.
  assert.match(src, /cellPinned/);
  // Row/column wash and sticky strip are gone.
  assert.doesNotMatch(src, /rowColWash/);
  assert.doesNotMatch(src, /orbit-rep-matrix-sticky-header/);
  // Y/X tuple bands are gone — tuples now live in the OrbitDetailCard (rendered on click-pin or in the modal).
  assert.doesNotMatch(src, /orbit-rep-matrix-y-band/);
  assert.doesNotMatch(src, /orbit-rep-matrix-x-band/);
});

test('OrbitRepMatrix renders an sr-only mirror table for accessibility', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /<table[^>]*className="sr-only"/);
  assert.match(src, /aria-label="The O → Q matrix"/);
  // Each <td> has aria-label describing (O, Q) + filled/empty.
  assert.match(src, /aria-label=\{[^}]*labelledTuple/);
});

test('OrbitRepMatrix renders Y/X axis tick gutters with computeAxisTicks', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /computeAxisTicks/);
  assert.match(src, /data-testid="orbit-rep-matrix-y-ticks"/);
  assert.match(src, /data-testid="orbit-rep-matrix-x-ticks"/);
});

test('OrbitRepMatrix legend uses hairline-only inline label grouping (no chip borders)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Legend wrapper still exists for screen-reader context.
  assert.match(src, /data-testid="orbit-rep-matrix-legend"/);
  // No more chip-border treatment on V/W labels.
  assert.doesNotMatch(src, /rounded-full border px-2/);
  // No more "LABELS" eyebrow uppercase chrome.
  assert.doesNotMatch(src, />labels<\/span>/i);
  // Visible / summed groupings shown as plain inline text.
  assert.match(src, /visible:/);
  assert.match(src, /summed:/);
});

test('OrbitRepMatrix expand affordance is a quiet gray glyph (no coral border, no small caps)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Trigger element still wired with the same data-action + aria-label.
  assert.match(src, /data-action="open-modal"/);
  assert.match(src, /aria-label="Expand matrix to full screen"/);
  // No more coral border, small caps, or "expand" text label.
  assert.doesNotMatch(src, /borderColor:\s*['"]rgba\(240,82,77,0\.45\)['"]/);
  assert.doesNotMatch(src, />\s*expand\s*<span/);
  // Quiet glyph: SVG with stroke="currentColor", opacity transition.
  assert.match(src, /<svg[^>]*viewBox/);
  assert.match(src, /opacity-50/);
  assert.match(src, /hover:opacity-100/);
});

test('OrbitRepMatrixModal renders modal shell with ESC + backdrop close', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrixModal.jsx');
  assert.match(src, /export default function OrbitRepMatrixModal/);
  assert.match(src, /role="dialog"/);
  assert.match(src, /Escape/);
  assert.match(src, /aria-modal/);
  // Must mount the matrix + OrbitDetailCard (inline mode) inside the modal panel.
  assert.match(src, /import OrbitRepMatrix from/);
  assert.match(src, /import OrbitDetailCard from/);
  assert.match(src, /mode="inline"/);
});

test('OrbitRepMatrix axis ticks render hairline tick marks alongside labels', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Two new tick-mark testids — one per axis.
  assert.match(src, /data-testid="orbit-rep-matrix-y-tick-marks"/);
  assert.match(src, /data-testid="orbit-rep-matrix-x-tick-marks"/);
});

test('OrbitRepMatrix renders an eased CSS overlay over the focused cell (not just a canvas stroke)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // The CSS-overlay div for the focused cell.
  assert.match(src, /data-testid="orbit-rep-matrix-focus-overlay"/);
  // Transition on opacity for the eased reveal. C50 a11y wraps the
  // transition in a reducedMotion ternary, so the string lives inside the
  // ternary rather than at the start of `transition:`.
  assert.match(src, /['"]opacity 120ms/);
});

test('OrbitRepMatrix paints a 1px depth lip on filled cells (subtle Apple-style depth)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // The depth-lip block is gated on cellH > 4 to avoid dominating tiny cells.
  assert.match(src, /showDepthLip\s*=\s*ch\s*>\s*4/);
  // The 1 px top-edge fill uses the darker coral variant.
  assert.match(src, /rgba\(178,\s*62,\s*58,\s*0\.18\)/);
});

test('OrbitRepMatrix axis labels use refined typography (lowercase, lighter weight, lighter color)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Tighter tracking + medium weight (not semibold/uppercase).
  assert.match(src, /font-medium tracking-\[0\.04em\]/);
  // Label text itself is lowercase ("orbit" not "Orbit", "rep" not "Rep").
  assert.match(src, />\s*orbit\s*<Latex/);
  assert.match(src, />\s*rep\s*<Latex/);
});
