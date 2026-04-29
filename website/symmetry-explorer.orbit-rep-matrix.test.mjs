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

test('layoutFor: square cells, canvas height capped at canvas width', () => {
  // 6 × 6: cell = floor(360/6) = 60, ≤ MAX_CELL=64 → cell=60 (fills column)
  // contentW = 6*60 = 360, contentH = 6*60 = 360, both ≤ 360 → no overflow
  assert.deepEqual(layoutFor({ canvasWidth: 360, numRows: 6, numCols: 6 }), {
    cellSize: 60,
    canvasW: 360,
    canvasH: 360,
    overflowY: false,
    overflowX: false,
    contentWidth: 360,
    contentHeight: 360,
  });
  // 14 × 10: cell = floor(360/14) = 25, contentH = 25*10 = 250 ≤ 360 → no overflow
  assert.deepEqual(layoutFor({ canvasWidth: 360, numRows: 10, numCols: 14 }), {
    cellSize: 25,
    canvasW: 350,
    canvasH: 250,
    overflowY: false,
    overflowX: false,
    contentWidth: 350,
    contentHeight: 250,
  });
  // 10 × 165 (Trilinear): cell = floor(360/10) = 36, ≤ MAX_CELL=64 → cell=36
  // contentW = 10*36 = 360, contentH = 165*36 = 5940 → overflowY engages
  const trilinear = layoutFor({ canvasWidth: 360, numRows: 165, numCols: 10 });
  assert.equal(trilinear.cellSize, 36);
  assert.equal(trilinear.canvasW, 360);
  assert.equal(trilinear.canvasH, 360);
  assert.equal(trilinear.overflowY, true);
  assert.equal(trilinear.contentHeight, 36 * 165);
});

test('layoutFor floors cell at MIN_CELL=4', () => {
  // 200 cols × 360 width → 1.8 px per cell → clamped to MIN_CELL=4
  // contentW = 4*200 = 800 > canvasW=360 → overflowX engages
  const tiny = layoutFor({ canvasWidth: 360, numRows: 50, numCols: 200 });
  assert.equal(tiny.cellSize, 4);
  // canvasW capped at canvasWidth (360) since contentW exceeds it
  assert.equal(tiny.canvasW, 360);
  assert.equal(tiny.overflowX, true);
});

test('cellAtPoint maps pixel coords to (row, col)', () => {
  const layout = { cellSize: 20, canvasW: 200, canvasH: 200, scrollTop: 0, scrollLeft: 0 };
  assert.deepEqual(cellAtPoint({ x: 0, y: 0 }, layout), { row: 0, col: 0 });
  assert.deepEqual(cellAtPoint({ x: 25, y: 5 }, layout), { row: 0, col: 1 });
  assert.deepEqual(cellAtPoint({ x: 150, y: 80 }, layout), { row: 4, col: 7 });
  assert.equal(cellAtPoint({ x: -5, y: 0 }, layout), null);
  assert.equal(cellAtPoint({ x: 250, y: 0 }, layout), null);
  assert.equal(cellAtPoint({ x: 0, y: 250 }, layout), null);
});

test('cellAtPoint accounts for scrollTop on tall matrices', () => {
  const layout = { cellSize: 20, canvasW: 200, canvasH: 200, scrollTop: 100, scrollLeft: 0 };
  // y=10 viewport-relative + scrollTop=100 = absolute y=110 → row=5
  assert.deepEqual(cellAtPoint({ x: 5, y: 10 }, layout), { row: 5, col: 0 });
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
  const layout = { cellSize: 20, canvasW: 200, canvasH: 200, scrollTop: 0, scrollLeft: 0, numRows: 5, numCols: 4 };
  // valid in-bounds
  assert.deepEqual(cellAtPoint({ x: 5, y: 5 }, layout), { row: 0, col: 0 });
  assert.deepEqual(cellAtPoint({ x: 75, y: 95 }, layout), { row: 4, col: 3 });
  // col >= numCols
  assert.equal(cellAtPoint({ x: 95, y: 5 }, { ...layout, canvasW: 300 }), null);
  // row >= numRows (with scroll past bottom)
  assert.equal(cellAtPoint({ x: 5, y: 5 }, { ...layout, scrollTop: 200, canvasH: 300 }), null);
});

test('cellAtPoint short-circuits when cellSize is invalid', () => {
  assert.equal(cellAtPoint({ x: 0, y: 0 }, { cellSize: 0, canvasW: 0, canvasH: 0 }), null);
  assert.equal(cellAtPoint({ x: 0, y: 0 }, { cellSize: null, canvasW: 100, canvasH: 100 }), null);
});

test('layoutFor returns a clean empty layout for 0 rows / 0 cols / NaN / negative', () => {
  for (const bad of [{ numRows: 0, numCols: 5 }, { numRows: 5, numCols: 0 }, { numRows: NaN, numCols: 5 }, { numRows: 5, numCols: -3 }]) {
    const out = layoutFor({ canvasWidth: 360, ...bad });
    assert.equal(out.cellSize, 0, `cellSize 0 for ${JSON.stringify(bad)}`);
    assert.equal(out.canvasW, 0);
    assert.equal(out.canvasH, 0);
    assert.equal(out.overflowX, false);
    assert.equal(out.overflowY, false);
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

test('OrbitRepMatrix tracks hover + pin state separately', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Two distinct useState slots.
  assert.match(src, /useState[^;]*hover/);
  assert.match(src, /useState[^;]*pin/);
  // Click handler.
  assert.match(src, /onClick=/);
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
    '#ECEFEF', '#4A7CFF', '#64748B', '#9AA0A0', '#B23E3A',
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

test('OrbitRepMatrix renders the label-legend chip row', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /data-testid="orbit-rep-matrix-legend"/);
  // Uses V chip and W chip styling — V at #4A7CFF, W at #64748B.
  assert.match(src, /#4A7CFF/);
  assert.match(src, /#64748B/);
  // Surfaces n.
  assert.match(src, /n\s*=/);
});

test('OrbitRepMatrix renders hover-driven tuple bands aligned to focused row/col', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /data-testid="orbit-rep-matrix-y-band"/);
  assert.match(src, /data-testid="orbit-rep-matrix-x-band"/);
  // Coral-light tint policy.
  assert.match(src, /rgba\(240,82,77,0\.06\)/);
  // Bands render the labelled tuple form.
  assert.match(src, /labelledTuple/);
});

test('OrbitRepMatrix paints a sticky column header strip when overflowY engages', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /position:\s*'sticky'/);
  assert.match(src, /data-testid="orbit-rep-matrix-sticky-header"/);
  assert.match(src, /layout\.overflowY/);
});

test('OrbitRepMatrix renders an sr-only mirror table for accessibility', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /<table[^>]*className="sr-only"/);
  assert.match(src, /aria-label="The O → Q matrix"/);
  // Each <td> has aria-label describing (O, Q) + filled/empty.
  assert.match(src, /aria-label=\{[^}]*labelledTuple/);
});

test('OrbitRepMatrixModal renders modal shell with ESC + backdrop close', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrixModal.jsx');
  assert.match(src, /export default function OrbitRepMatrixModal/);
  assert.match(src, /role="dialog"/);
  assert.match(src, /Escape/);
  assert.match(src, /aria-modal/);
  // Must mount the matrix + panel inside the modal panel.
  assert.match(src, /import OrbitRepMatrix from/);
  assert.match(src, /import WorkedExamplePanel from/);
});
