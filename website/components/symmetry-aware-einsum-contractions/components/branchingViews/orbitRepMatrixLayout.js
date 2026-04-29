// Pure layout / hit-testing math for the O → Q matrix.
// No React, no DOM — easily unit-tested.
//
// Used by:
//   - components/branchingViews/OrbitRepMatrix.jsx (canvas component)
//   - components/branchingViews/WorkedExamplePanel.jsx (right-side panel)
//   - components/BranchingDemo.jsx (parent that owns hover/pin state)

export const SQUARE_FRAME = 360;       // default canvas width on lg
export const MIN_CELL = 4;             // smallest visible cell side
export const MAX_CELL = 32;            // largest cell side (small presets)

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
 * Compute canvas geometry for a given (rows, cols) at a target frame width.
 * - cellSize is square, clamped to [MIN_CELL, MAX_CELL].
 * - canvasH is capped at canvasWidth (square frame max).
 * - canvasW is min(contentWidth, frame width) — never wider than the frame.
 * - overflowY/overflowX flag when content extends past the visible frame;
 *   the consumer is responsible for rendering a scrollable inner viewport.
 */
export function layoutFor({ canvasWidth, numRows, numCols }) {
  const safeWidth = Math.max(canvasWidth || SQUARE_FRAME, MIN_CELL * 2);
  const safeRows = Math.max(Math.floor(numRows) || 0, 0);
  const safeCols = Math.max(Math.floor(numCols) || 0, 0);
  if (safeRows === 0 || safeCols === 0) {
    return { cellSize: 0, canvasW: 0, canvasH: 0, overflowY: false, overflowX: false, contentWidth: 0, contentHeight: 0 };
  }
  const cellByCols = Math.floor(safeWidth / safeCols);
  const cellSize = Math.max(MIN_CELL, Math.min(MAX_CELL, cellByCols));
  const contentWidth = cellSize * safeCols;
  const contentHeight = cellSize * safeRows;
  const canvasW = Math.min(contentWidth, safeWidth);
  const canvasH = Math.min(contentHeight, safeWidth);
  const overflowY = contentHeight > canvasH;
  const overflowX = contentWidth > canvasW;
  return { cellSize, canvasW, canvasH, overflowY, overflowX, contentWidth, contentHeight };
}

/**
 * Map viewport-relative pixel coords (relative to the canvas frame's origin)
 * to a (row, col) cell index, accounting for any scroll offset of the inner
 * scrollable viewport. Returns null if outside the rendered cell area.
 */
export function cellAtPoint({ x, y }, layout) {
  if (!layout) return null;
  const { cellSize, canvasW, canvasH, scrollTop = 0, scrollLeft = 0, numRows, numCols } = layout;
  if (!cellSize || cellSize <= 0) return null;
  if (x < 0 || y < 0 || x >= canvasW || y >= canvasH) return null;
  const col = Math.floor((x + scrollLeft) / cellSize);
  const row = Math.floor((y + scrollTop) / cellSize);
  if (col < 0 || row < 0) return null;
  if (Number.isFinite(numRows) && row >= numRows) return null;
  if (Number.isFinite(numCols) && col >= numCols) return null;
  return { row, col };
}
