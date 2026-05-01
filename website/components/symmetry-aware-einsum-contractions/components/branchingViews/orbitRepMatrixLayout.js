// Pure layout / hit-testing math for the O → Q matrix.
// No React, no DOM — easily unit-tested.
//
// Design-system call: the visible matrix viewport is a fixed 360px frame, but
// rows never compress below the 8px spacing unit. Tall matrices render a larger
// canvas inside that viewport and scroll vertically.
//
// Used by:
//   - components/branchingViews/OrbitRepMatrix.jsx (canvas component)
//   - components/BranchingDemo.jsx (parent that owns hover/pin state)

export const SQUARE_FRAME = 360;       // default canvas width on lg
export const FIXED_CANVAS_HEIGHT = 360; // viewport height; tall content scrolls within it
export const MIN_CELL = 1;             // 1-px floor — never invisible
export const MIN_INTERACTIVE_CELL = 8;  // design-system --space-2: smallest readable row

export function tupleKey(tuple) {
  return JSON.stringify(tuple ?? {});
}

export function compactTuple(tuple) {
  if (!tuple || typeof tuple !== 'object') return '—';
  return `(${Object.values(tuple).join(', ')})`;
}

export function labelledTuple(tuple) {
  if (!tuple || typeof tuple !== 'object') return '—';
  return `(${Object.entries(tuple).map(([k, v]) => `${k}=${v}`).join(', ')})`;
}

export function derivePreReps(orbitRows = []) {
  const seen = new Map();
  for (const row of orbitRows) {
    for (const out of row.outputs ?? []) {
      const k = tupleKey(out.outTuple);
      if (!seen.has(k)) seen.set(k, out.outTuple);
    }
  }
  return Array.from(seen.entries()).map(([k, tuple]) => ({ k, tuple }));
}

export function deriveCells(orbitRows = [], reps = []) {
  return orbitRows.map((row) => {
    const coeffByKey = new Map();
    for (const out of row.outputs ?? []) coeffByKey.set(tupleKey(out.outTuple), out.coeff ?? 1);
    return reps.map((rep) => coeffByKey.has(rep.k) ? coeffByKey.get(rep.k) : null);
  });
}

/**
 * Fixed viewport with adaptive rectangular cells.
 *
 * `viewportW = canvasWidth` (responsive to the container)
 * `viewportH = canvasHeight` (fixed — defaults to FIXED_CANVAS_HEIGHT)
 * `cellWidth  = floor(canvasW / numCols)`
 * `cellHeight = max(8, floor(viewportH / numRows))`
 *
 * Compact matrices fill the viewport. Tall matrices keep an 8px row floor,
 * expand the content canvas vertically, and scroll within the viewport.
 */
export function layoutFor({ canvasWidth, canvasHeight, numRows, numCols }) {
  const safeWidth = Math.max(canvasWidth || SQUARE_FRAME, MIN_CELL * 2);
  const safeHeight = Math.max(canvasHeight || FIXED_CANVAS_HEIGHT, MIN_CELL * 2);
  const safeRows = Math.max(Math.floor(numRows) || 0, 0);
  const safeCols = Math.max(Math.floor(numCols) || 0, 0);
  if (safeRows === 0 || safeCols === 0) {
    return {
      cellWidth: 0, cellHeight: 0,
      canvasW: 0, canvasH: 0,
      contentWidth: 0, contentHeight: 0,
      viewportW: 0, viewportH: 0,
      needsVerticalScroll: false,
    };
  }
  const cellWidth = Math.max(MIN_CELL, Math.floor(safeWidth / safeCols));
  const cellHeight = Math.max(MIN_INTERACTIVE_CELL, Math.floor(safeHeight / safeRows));
  const contentWidth = cellWidth * safeCols;
  const contentHeight = cellHeight * safeRows;
  const viewportWidth = Math.min(contentWidth, safeWidth);
  const viewportHeight = Math.min(contentHeight, safeHeight);
  return {
    cellWidth, cellHeight,
    canvasW: contentWidth, canvasH: contentHeight,
    contentWidth, contentHeight,
    viewportW: viewportWidth, viewportH: viewportHeight,
    needsVerticalScroll: contentHeight > safeHeight,
  };
}

/**
 * Compute tick indices for an axis with `n` cells. Always include 0 and n-1
 * (the first and last index), plus interior multiples of a sensible round
 * stride. Limits total ticks to `maxTicks` (≈ 6 by default) to avoid clutter.
 *
 * Examples:
 *   computeAxisTicks(9, 6)   → [0, 2, 4, 6, 8]            // small range, dense
 *   computeAxisTicks(27, 6)  → [0, 5, 10, 15, 20, 26]
 *   computeAxisTicks(165, 6) → [0, 33, 66, 99, 132, 164]  // round-stride spread
 *
 * @param {number} n total number of cells along the axis
 * @param {number} maxTicks soft cap on the number of ticks returned
 * @returns {number[]} sorted, unique tick indices in [0, n-1]
 */
export function computeAxisTicks(n, maxTicks = 6) {
  const safeN = Math.max(0, Math.floor(n));
  if (safeN === 0) return [];
  if (safeN === 1) return [0];
  const safeMax = Math.max(2, Math.floor(maxTicks));
  if (safeN <= safeMax) {
    return Array.from({ length: safeN }, (_, i) => i);
  }
  const stride = Math.max(1, Math.ceil((safeN - 1) / (safeMax - 1)));
  const ticks = new Set();
  ticks.add(0);
  for (let i = stride; i < safeN - 1; i += stride) ticks.add(i);
  ticks.add(safeN - 1);
  return Array.from(ticks).sort((a, b) => a - b);
}

/**
 * Map canvas-relative pixel coords (origin at top-left of the canvas) to a
 * (row, col) cell index. Returns null if outside the rendered cell area.
 *
 * In scroll mode, coords are still content-relative because callers measure
 * against the scrolled canvas node, not the fixed viewport wrapper.
 */
export function cellAtPoint({ x, y }, layout) {
  if (!layout) return null;
  const { cellWidth, cellHeight, canvasW, canvasH, numRows, numCols } = layout;
  if (!cellWidth || !cellHeight || cellWidth <= 0 || cellHeight <= 0) return null;
  if (x < 0 || y < 0 || x >= canvasW || y >= canvasH) return null;
  const col = Math.floor(x / cellWidth);
  const row = Math.floor(y / cellHeight);
  if (col < 0 || row < 0) return null;
  if (Number.isFinite(numRows) && row >= numRows) return null;
  if (Number.isFinite(numCols) && col >= numCols) return null;
  return { row, col };
}
