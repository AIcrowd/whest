/**
 * Shared incidence matrix component — used in Section 3 (static),
 * Section 4 (animated σ-loop), and the rejected σ modal.
 *
 * Renders a grid of cells using absolute positioning (card-based),
 * matching the animated matrix style. When `animate` is true, rows
 * and columns transition smoothly to new positions.
 */
import { buildUVertexLabels } from '../engine/uVertexLabel.js';

// Layout constants — single source of truth
export const CELL_W = 70;
export const CELL_H = 38;
export const HEADER_H = 32;
export const LABEL_W = 90;

export default function IncidenceMatrix({
  matrix,          // number[][] — the matrix data to display
  colLabels,       // string[] — column header labels
  uVertices,       // from graph — for building row labels
  example,         // for building row labels via buildUVertexLabels
  freeLabels,      // Set<string> — V labels (colored blue)
  rowPerm,         // number[] | null — row ordering (null = identity)
  colPerm,         // number[] | null — column reordering (null = identity)
  movedRows,       // Set<number> | null — highlighted rows
  movedCols,       // Set<number> | null — highlighted columns
  animate,         // boolean — enable CSS transitions
  label,           // string | null — title above the matrix (e.g. "M", "σ(M)")
  compact,         // boolean — slightly smaller for modal use
}) {
  const numRows = matrix.length;
  const numCols = colLabels.length;
  const uLabels = buildUVertexLabels(uVertices, example);

  const cellW = compact ? 56 : CELL_W;
  const cellH = compact ? 32 : CELL_H;
  const headerH = compact ? 28 : HEADER_H;
  const labelW = compact ? 70 : LABEL_W;

  const effectiveRowPerm = rowPerm || Array.from({ length: numRows }, (_, i) => i);
  const effectiveColPerm = colPerm || Array.from({ length: numCols }, (_, i) => i);

  const containerH = headerH + numRows * cellH + 4;
  const containerW = labelW + numCols * cellW;

  // Which rows/cols are moved
  const movedRowSet = movedRows || new Set();
  const movedColSet = movedCols || new Set();

  // Inverse row perm: positionOf[uIdx] = visual row
  const positionOf = {};
  effectiveRowPerm.forEach((uIdx, row) => { positionOf[uIdx] = row; });

  const identity = Array.from({ length: numRows }, (_, i) => i);
  const movedUIndices = new Set();
  effectiveRowPerm.forEach((uIdx, k) => { if (uIdx !== k) movedUIndices.add(uIdx); });

  const transitionStyle = animate ? { transition: 'transform 0.5s ease' } : {};

  return (
    <div className="inc-matrix-outer">
      {label && <div className="inc-matrix-label">{label}</div>}
      <div className="inc-matrix" style={{ height: containerH, width: containerW }}>
        {/* Column headers */}
        {colLabels.map((lbl, ci) => {
          const isMoved = movedColSet.has(ci);
          const isV = freeLabels?.has(lbl);
          return (
            <div key={`h-${lbl}`}
              className={`inc-col-header ${isMoved ? 'inc-moved' : ''} ${isV ? 'inc-col-v' : 'inc-col-w'}`}
              style={{
                ...transitionStyle,
                transform: `translateX(${labelW + ci * cellW}px)`,
                width: cellW, height: headerH,
              }}>
              {lbl}
            </div>
          );
        })}

        {/* Row cards */}
        {identity.map(uIdx => {
          const visualRow = positionOf[uIdx];
          const isMoved = movedUIndices.has(uIdx);
          const rowData = matrix[visualRow];

          return (
            <div key={uIdx}
              className={`inc-row ${isMoved ? 'inc-row-moved' : ''}`}
              style={{
                ...transitionStyle,
                transform: `translateY(${headerH + visualRow * cellH}px)`,
                height: cellH,
              }}>
              {/* Row label */}
              <div className="inc-row-label" style={{ width: labelW }}>
                {uLabels[uIdx]}
              </div>
              {/* Cells */}
              {rowData.map((val, ci) => {
                const xPos = effectiveColPerm[ci];
                const colMoved = movedColSet.has(ci);
                return (
                  <div key={ci}
                    className={`inc-cell ${val > 0 ? 'inc-cell-active' : ''} ${colMoved ? 'inc-cell-col-moved' : ''}`}
                    style={{
                      ...transitionStyle,
                      transform: `translateX(${labelW + xPos * cellW}px)`,
                      width: cellW, height: cellH,
                    }}>
                    {val}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
