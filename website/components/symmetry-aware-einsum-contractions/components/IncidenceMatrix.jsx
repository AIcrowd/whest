/**
 * Shared incidence matrix component — used in Section 3 (static),
 * Section 4 (animated σ-loop), and the rejected σ modal.
 *
 * Renders a grid of cells using absolute positioning (card-based),
 * matching the animated matrix style. When `animate` is true, rows
 * and columns transition smoothly to new positions.
 */
import Latex from './Latex.jsx';
import { buildUVertexLabels } from '../engine/uVertexLabel.js';
import { notationLatex } from '../lib/notationSystem.js';

// Layout constants — single source of truth
export const CELL_W = 70;
export const CELL_H = 38;
export const HEADER_H = 32;
export const LABEL_W = 90;
const DEFAULT_LABEL_COLOR = '#9ca3af';

export default function IncidenceMatrix({
  matrix,          // number[][] — the matrix data to display
  colLabels,       // string[] — column header labels
  uVertices,       // from graph — for building row labels
  example,         // for building row labels via buildUVertexLabels
  freeLabels,      // Set<string> — V labels (colored blue)
  variableColors,  // { [name]: { color } } — per-variable colors for row labels
  rowPerm,         // number[] | null — row ordering (null = identity)
  colPerm,         // number[] | null — column reordering (null = identity)
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
  const movedColSet = movedCols || new Set();

  // Inverse row perm: positionOf[uIdx] = visual row
  const positionOf = {};
  effectiveRowPerm.forEach((uIdx, row) => { positionOf[uIdx] = row; });

  const identity = Array.from({ length: numRows }, (_, i) => i);
  const movedUIndices = new Set();
  effectiveRowPerm.forEach((uIdx, k) => { if (uIdx !== k) movedUIndices.add(uIdx); });

  const transitionStyle = animate ? { transition: 'transform 0.5s ease' } : {};

  // Compute column fingerprints from the displayed matrix
  const fingerprints = {};
  const fpToLabels = {};
  for (let ci = 0; ci < numCols; ci++) {
    const lbl = colLabels[ci];
    const fp = matrix.map(row => row[ci]).join(', ');
    fingerprints[lbl] = fp;
    (fpToLabels[fp] ??= new Set()).add(lbl);
  }
  // Color equivalent fingerprints
  const fpColorPalette = ['#4a7cff', '#3ddc84', '#ffb74d', '#bb86fc', '#ff5252'];
  const fpColors = {};
  let fpColorIdx = 0;
  for (const [fp, lblSet] of Object.entries(fpToLabels)) {
    if (lblSet.size >= 2) {
      fpColors[fp] = fpColorPalette[fpColorIdx++ % fpColorPalette.length];
    }
  }

  return (
    <div className="inc-matrix-outer">
      {label && (
        <div className="inc-matrix-label">
          <Latex math={label} />
        </div>
      )}
      <div className="inc-matrix-legend">
        <span className="inc-matrix-legend-item">
          <span className="inc-matrix-legend-kicker">rows:</span>
          <Latex math={notationLatex('u_axis_classes')} />
          <span className="inc-matrix-legend-text">axis classes</span>
        </span>
        <span className="inc-matrix-legend-item">
          <span className="inc-matrix-legend-kicker">columns:</span>
          <Latex math={`${notationLatex('l_labels')} = ${notationLatex('v_free')} \\sqcup ${notationLatex('w_summed')}`} />
        </span>
      </div>
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
              <Latex math={lbl} />
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
              {/* Row label — colored by variable */}
              <div className="inc-row-label" style={{
                width: labelW,
                color: (() => {
                  const opName = example?.operandNames?.[uVertices[uIdx]?.opIdx];
                  return variableColors?.[opName]?.color || DEFAULT_LABEL_COLOR;
                })(),
              }}>
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
      {/* Column fingerprints */}
      <div className="inc-fingerprints-header">Column Fingerprints</div>
      <div className="inc-fingerprints">
        {colLabels.map(lbl => {
          const fp = fingerprints[lbl];
          const eqColor = fpColors[fp];
          return (
            <div key={lbl} className="inc-fp-item" style={eqColor ? { borderColor: eqColor } : {}}>
              <span className={`inc-fp-label ${freeLabels?.has(lbl) ? 'inc-col-v' : 'inc-col-w'}`}>
                <Latex math={lbl} />
              </span>
              <code className="inc-fp-value">({fp})</code>
              {eqColor && <span className="inc-fp-eq" style={{ background: eqColor }}>≡</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
