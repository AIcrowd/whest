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
  // 6 × 6: cell = floor(360/6) = 60, clamped to MAX_CELL=32 → cell=32
  // contentW = 6*32 = 192, contentH = 6*32 = 192, both ≤ 360 → no overflow
  assert.deepEqual(layoutFor({ canvasWidth: 360, numRows: 6, numCols: 6 }), {
    cellSize: 32,
    canvasW: 192,
    canvasH: 192,
    overflowY: false,
    overflowX: false,
    contentWidth: 192,
    contentHeight: 192,
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
  // 10 × 165 (Trilinear): cell = floor(360/10) = 36, clamped to 32 → cell=32
  // contentW = 10*32 = 320, contentH = 165*32 = 5280 → overflowY engages
  const trilinear = layoutFor({ canvasWidth: 360, numRows: 165, numCols: 10 });
  assert.equal(trilinear.cellSize, 32);
  assert.equal(trilinear.canvasW, 320);
  assert.equal(trilinear.canvasH, 360);
  assert.equal(trilinear.overflowY, true);
  assert.equal(trilinear.contentHeight, 32 * 165);
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
