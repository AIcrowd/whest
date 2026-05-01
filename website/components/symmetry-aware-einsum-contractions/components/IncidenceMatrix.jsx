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
import {
  explorerThemeColor,
  getExplorerThemeFingerprintPalette,
} from '../lib/explorerTheme.js';
import {
  getActiveExplorerThemeId,
  notationColor,
  notationLatex,
} from '../lib/notationSystem.js';

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
  variableColors,  // { [name]: { color } } — per-variable colors for row labels
  rowPerm,         // number[] | null — row ordering (null = identity)
  colPerm,         // number[] | null — column reordering (null = identity)
  movedCols,       // Set<number> | null — highlighted columns
  animate,         // boolean — enable CSS transitions
  label,           // string | null — title above the matrix (e.g. "M", "σ(M)")
  compact,         // boolean — slightly smaller for modal use
  rejected,        // boolean — render with red turn-red styling for failed π search (C22)
  mismatchedLabels,// Set<string> | null — column labels with no matching fingerprint, marked with red border/× overlay (C22)
  compactMode,     // boolean — collapse the per-row matrix into column signatures only (C22 "Show fingerprints" toggle)
}) {
  const explorerThemeId = getActiveExplorerThemeId();
  const numRows = matrix.length;
  const numCols = colLabels.length;
  const uLabels = buildUVertexLabels(uVertices, example);
  const defaultLabelColor = explorerThemeColor(explorerThemeId, 'muted');

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
  const fpColorPalette = getExplorerThemeFingerprintPalette(explorerThemeId);
  const fpColors = {};
  let fpColorIdx = 0;
  for (const [fp, lblSet] of Object.entries(fpToLabels)) {
    if (lblSet.size >= 2) {
      fpColors[fp] = fpColorPalette[fpColorIdx++ % fpColorPalette.length];
    }
  }

  // C22: classify row/col label semantic role for hover-help tooltips and
  // domain-mismatch marking. Mismatched labels (those without a fingerprint
  // partner during π search) are marked with a red border + × overlay so the
  // user can see exactly which columns broke the σ → π recovery.
  const mismatchedSet = mismatchedLabels || new Set();
  const outerClassName = [
    'inc-matrix-outer',
    rejected ? 'inc-matrix-rejected' : '',
    compactMode ? 'inc-matrix-compact-mode' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={outerClassName} data-rejected={rejected ? 'true' : undefined}>
      {label && (
        <div className="inc-matrix-label">
          <Latex math={label} />
          {rejected && (
            <span className="inc-matrix-reject-badge" aria-label="no compatible π — fingerprint mismatch">
              ✗ no compatible π
            </span>
          )}
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
      {/* Matrix grid — hidden when compactMode (Show fingerprints) is on */}
      {!compactMode && (
        <div className="inc-matrix" style={{ height: containerH, width: containerW }}>
          {/* Column headers */}
          {colLabels.map((lbl, ci) => {
            const isMoved = movedColSet.has(ci);
            const isV = freeLabels?.has(lbl);
            const isMismatch = mismatchedSet.has(lbl);
            const headerClass = [
              'inc-col-header',
              isMoved ? 'inc-moved' : '',
              isV ? 'inc-col-v' : 'inc-col-w',
              isMismatch ? 'inc-col-mismatch' : '',
              'cursor-help',
            ].filter(Boolean).join(' ');
            const headerTitle = isV
              ? `Free (visible) label ${lbl} — survives the contraction.`
              : `Summed label ${lbl} — internal axis the contraction sums over.`;
            return (
              <div key={`h-${lbl}`}
                className={headerClass}
                title={isMismatch ? `${headerTitle} No matching column fingerprint — π cannot place this label.` : headerTitle}
                tabIndex={0}
                style={{
                  ...transitionStyle,
                  transform: `translateX(${labelW + ci * cellW}px)`,
                  width: cellW, height: headerH,
                }}>
                <Latex math={lbl} />
                {isMismatch && (
                  <span className="inc-col-mismatch-mark" aria-label={`column ${lbl} has no matching fingerprint`}>×</span>
                )}
              </div>
            );
          })}

          {/* Row cards */}
          {identity.map(uIdx => {
            const visualRow = positionOf[uIdx];
            const isMoved = movedUIndices.has(uIdx);
            const rowData = matrix[visualRow];
            const opName = example?.operandNames?.[uVertices[uIdx]?.opIdx];
            const rowTitle = `Axis class ${uLabels[uIdx]} — row of the M incidence matrix; one row per (operand, axis) class${opName ? ` (operand ${opName})` : ''}.`;

            return (
              <div key={uIdx}
                className={`inc-row ${isMoved ? 'inc-row-moved' : ''}`}
                style={{
                  ...transitionStyle,
                  transform: `translateY(${headerH + visualRow * cellH}px)`,
                  height: cellH,
                }}>
                {/* Row label — colored by variable */}
                <div className="inc-row-label cursor-help"
                  title={rowTitle}
                  tabIndex={0}
                  style={{
                    width: labelW,
                    color: variableColors?.[opName]?.color || defaultLabelColor,
                  }}>
                  {uLabels[uIdx]}
                </div>
                {/* Cells */}
                {rowData.map((val, ci) => {
                  const xPos = effectiveColPerm[ci];
                  const colMoved = movedColSet.has(ci);
                  const colMismatch = mismatchedSet.has(colLabels[ci]);
                  return (
                    <div key={ci}
                      className={`inc-cell ${val > 0 ? 'inc-cell-active' : ''} ${colMoved ? 'inc-cell-col-moved' : ''} ${colMismatch ? 'inc-cell-mismatch' : ''}`}
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
      )}
      {/* Column fingerprints — always rendered (this is the summary the
          compactMode toggle exposes). When compactMode is on it is the only
          visible representation of the matrix; otherwise it sits below the
          grid as a per-column equivalence summary. */}
      <div className="inc-fingerprints-header">Column Fingerprints</div>
      <div className="inc-fingerprints">
        {colLabels.map(lbl => {
          const fp = fingerprints[lbl];
          const eqColor = fpColors[fp];
          const isMismatch = mismatchedSet.has(lbl);
          const itemStyle = {
            ...(eqColor ? { borderColor: eqColor } : {}),
            ...(isMismatch ? { borderColor: 'var(--coral)', borderWidth: '2px' } : {}),
          };
          return (
            <div key={lbl} className={`inc-fp-item ${isMismatch ? 'inc-fp-mismatch' : ''}`} style={itemStyle}>
              <span className={`inc-fp-label ${freeLabels?.has(lbl) ? 'inc-col-v' : 'inc-col-w'}`}>
                <Latex math={lbl} />
              </span>
              <code className="inc-fp-value">({fp})</code>
              {eqColor && (
                <span className="inc-fp-eq" style={{ background: eqColor, color: notationColor('m_incidence') }}>
                  ≡
                </span>
              )}
              {isMismatch && (
                <span className="inc-fp-mismatch-mark" aria-label={`no fingerprint partner for ${lbl}`}>×</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
