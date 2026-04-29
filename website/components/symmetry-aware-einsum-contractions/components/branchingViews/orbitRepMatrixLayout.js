// Pure layout / hit-testing math for the O → Q matrix.
// No React, no DOM — easily unit-tested.
//
// Design: the canvas is a FIXED SQUARE frame whose dimensions follow the
// container width (responsive). Cells are RECTANGLES — `cellWidth` =
// floor(canvasW / numCols), `cellHeight` = floor(canvasH / numRows). Every
// matrix fits in-frame at all sizes; there is no internal scroll.
//
// Used by:
//   - components/branchingViews/OrbitRepMatrix.jsx (canvas component)
//   - components/BranchingDemo.jsx (parent that owns hover/pin state)

export const SQUARE_FRAME = 360;       // default canvas width on lg
export const FIXED_CANVAS_HEIGHT = 360; // canvas height — fixed; cell height adapts to numRows
export const MIN_CELL = 1;             // 1-px floor — never invisible

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
 * Fixed-dimensions canvas with adaptive rectangular cells.
 *
 * `canvasW = canvasWidth` (responsive to the container)
 * `canvasH = canvasHeight` (fixed — defaults to FIXED_CANVAS_HEIGHT)
 * `cellWidth  = floor(canvasW / numCols)`
 * `cellHeight = floor(canvasH / numRows)`
 *
 * Cells fill the canvas: column count adjusts cellWidth, row count adjusts
 * cellHeight. This is the user's explicit design — fixed canvas dimensions,
 * row height + column width both adapt to the matrix shape, no scroll.
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
    };
  }
  const cellWidth = Math.max(MIN_CELL, Math.floor(safeWidth / safeCols));
  const cellHeight = Math.max(MIN_CELL, Math.floor(safeHeight / safeRows));
  const contentWidth = cellWidth * safeCols;
  const contentHeight = cellHeight * safeRows;
  return {
    cellWidth, cellHeight,
    canvasW: contentWidth, canvasH: contentHeight,
    contentWidth, contentHeight,
  };
}

/**
 * Map canvas-relative pixel coords (origin at top-left of the canvas) to a
 * (row, col) cell index. Returns null if outside the rendered cell area.
 *
 * No scroll handling — the canvas is fixed-size and fits all content.
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
