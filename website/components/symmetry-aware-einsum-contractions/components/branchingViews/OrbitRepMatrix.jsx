import { useMemo, useState } from 'react';
import Latex from '../Latex.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../../lib/explorerTheme.js';
import { notationColor } from '../../lib/notationSystem.js';

// OrbitRepMatrix — bipartite (orbit × stored output rep) matrix.
//
// Pedagogically: counting filled cells = α (the unified accumulation count
// α = #{(O, Q) ∈ X/G × Y/H : π_V(O) ∩ Q ≠ ∅}).
//
// Visual decisions are anchored in the Flopscope design system at
// /design-system. Refined-minimal aesthetic: coral hero on neutral gray,
// no shadow, 1px borders only where they encode structure, content IS
// the visual.
//
// Axis treatment: per the user's pedagogical request, individual tuple
// labels are NOT shown on the row/column headers. Instead, the matrix
// has two single axis labels — "Orbit" on the Y-axis, "Rep" on the
// X-axis — and the rich (k=v) tuple values surface inside the hover
// tooltip. This keeps the matrix readable across all preset sizes (6×6
// up to 56×56) without crowding.
//
// Sizing: the matrix is rendered as a square. Cell size = min(square /
// cols, square / rows), clamped to [MIN_CELL, MAX_CELL] so dense matrices
// don't shrink below readability and sparse ones don't blow out. The
// whole grid stays inside a fixed 320×320 px square; the wrapper has
// `overflow-auto` so very dense matrices scroll natively if needed.

function compactTupleValues(tuple) {
  if (!tuple || typeof tuple !== 'object') return '—';
  return `(${Object.values(tuple).join(', ')})`;
}

function labelledTuple(tuple) {
  if (!tuple || typeof tuple !== 'object') return '—';
  return `(${Object.entries(tuple).map(([k, v]) => `${k}=${v}`).join(', ')})`;
}

// Compact LaTeX rendering of a tuple as `(k_1=v_1,\,k_2=v_2,\,\ldots)`.
function tupleLatex(tuple) {
  if (!tuple || typeof tuple !== 'object') return '\\,—\\,';
  const parts = Object.entries(tuple).map(([k, v]) => `${k}{=}${v}`);
  return `(${parts.join(',\\,')})`;
}

function tupleKey(tuple) {
  return JSON.stringify(tuple ?? {});
}

const SQUARE_SIZE = 320;
const MIN_CELL = 4;
const MAX_CELL = 32;

export default function OrbitRepMatrix({
  orbitRows = [],
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,
}) {
  const themeId = getActiveExplorerThemeId();
  const [hover, setHover] = useState(null); // { i, j, x, y } when over a cell
  const [hoverAxis, setHoverAxis] = useState(null); // { kind: 'row'|'col', idx }

  const reps = useMemo(() => {
    const seen = new Map();
    orbitRows.forEach((row) => {
      (row.outputs ?? []).forEach((out) => {
        const k = tupleKey(out.outTuple);
        if (!seen.has(k)) seen.set(k, out.outTuple);
      });
    });
    return Array.from(seen.entries()).map(([k, tuple]) => ({ k, tuple }));
  }, [orbitRows]);

  // For each (orbit, rep), the cell stores either null (no projection) or
  // the matching output's coeff (= number of orbit members that project
  // to this stored rep, after H-canonicalization). The boolean filled-ness
  // is `cells[i][j] !== null`.
  const cells = useMemo(
    () => orbitRows.map((row) => {
      const repCoeffByKey = new Map();
      (row.outputs ?? []).forEach((out) => {
        repCoeffByKey.set(tupleKey(out.outTuple), out.coeff ?? 1);
      });
      return reps.map((rep) =>
        repCoeffByKey.has(rep.k) ? repCoeffByKey.get(rep.k) : null,
      );
    }),
    [orbitRows, reps],
  );

  const alphaTotal = cells.reduce(
    (acc, row) => acc + row.filter((c) => c !== null).length,
    0,
  );

  // Axis labels — read off the first available tuple in each axis.
  const rowLabels = orbitRows.length > 0 ? Object.keys(orbitRows[0].repTuple ?? {}) : [];
  const colLabels = reps.length > 0 ? Object.keys(reps[0].tuple ?? {}) : [];

  // Square geometry. Cell size adapts to the largest of (rows, cols) so
  // both axes still fit the 320 px envelope.
  const numRows = orbitRows.length;
  const numCols = reps.length;
  const denom = Math.max(numRows, numCols, 1);
  const cellSize = Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor(SQUARE_SIZE / denom)));

  // Theme tokens. All bound to design-system roles.
  const muted = explorerThemeColor(themeId, 'muted');
  const body = explorerThemeColor(themeId, 'body');
  const border = explorerThemeColor(themeId, 'border');
  const surface = explorerThemeColor(themeId, 'surface');
  const surfaceInset = explorerThemeColor(themeId, 'surfaceInset');
  const hero = explorerThemeColor(themeId, 'hero');
  const heroMuted = explorerThemeColor(themeId, 'heroMuted');
  const heroLight = explorerThemeTint(themeId, 'hero', 0.08);
  const filledCellTint = explorerThemeTint(themeId, 'hero', 0.22);
  const hoverShade = explorerThemeTint(themeId, 'hero', 0.06);
  const alphaAccent = notationColor('alpha_total');

  // Empty state.
  if (numRows === 0) {
    return (
      <div
        data-testid="orbit-rep-matrix-empty"
        className="flex h-[260px] w-full items-center justify-center text-[12px]"
        style={{ color: muted }}
      >
        no orbit data for this preset
      </div>
    );
  }

  function handleCellEnter(i, j, event) {
    const rect = event.currentTarget.getBoundingClientRect();
    setHover({ i, j, x: rect.left + rect.width / 2, y: rect.top });
    if (onHover) {
      const labels = Object.keys(orbitRows[i]?.repTuple ?? {});
      onHover({ labels, leafKeys: [] });
    }
  }
  function handleCellLeave() {
    setHover(null);
  }
  function handleMatrixLeave() {
    setHover(null);
    setHoverAxis(null);
    if (onHover) onHover(null);
  }

  const matrixWidth = cellSize * numCols;
  const matrixHeight = cellSize * numRows;

  return (
    <div data-testid="orbit-rep-matrix" className="w-full">
      {/* Reading guide — collapsed by default. */}
      <details className="mb-3" data-testid="orbit-rep-matrix-reading-guide">
        <summary
          className="cursor-help text-[10px] font-semibold uppercase tracking-[0.16em]"
          style={{ color: muted }}
        >
          How to read this matrix
        </summary>
        <ul
          className="mt-2 list-disc space-y-1 pl-5 font-serif text-[13px] leading-6"
          style={{ color: body }}
        >
          <li>
            <strong className="font-semibold">Y-axis (Orbit)</strong> — product
            orbits (M total). Each row is one orbit; rows are labelled by the
            orbit&apos;s representative tuple of label values
            ({rowLabels.length > 0 ? `over ${compactTupleValues(Object.fromEntries(rowLabels.map((l) => [l, l])))}` : ''}).
          </li>
          <li>
            <strong className="font-semibold">X-axis (Rep)</strong> — stored
            output representatives (|Y/H| total). Each column is one rep,
            labelled by its visible-side tuple, canonicalized under
            H = Stab<sub>G</sub>(V)|<sub>V</sub>
            ({colLabels.length > 0 ? `over ${compactTupleValues(Object.fromEntries(colLabels.map((l) => [l, l])))}` : ''}).
          </li>
          <li>
            <strong className="font-semibold">Filled cell</strong> — that
            orbit projects to that rep at least once. <em>Counting filled
            cells = α</em> (the accumulation count).
          </li>
          <li>
            <strong className="font-semibold">Hover any cell</strong> for the
            tuples + how the orbit&apos;s members map to output bins, with
            the LaTeX contribution.
          </li>
        </ul>
      </details>

      {/* α total readout. */}
      <div
        className="mb-2 flex items-baseline gap-2 text-[11px]"
        style={{ color: muted }}
      >
        <span className="font-semibold uppercase tracking-[0.16em]">
          {numRows} orbits × {numCols} reps
        </span>
        <span className="ml-auto font-mono">
          α ={' '}
          <strong style={{ color: alphaAccent }}>{alphaTotal}</strong>
          <span className="ml-1 text-[10px]" style={{ color: muted }}>
            (count of filled cells)
          </span>
        </span>
      </div>

      {/* Matrix itself. Wrapped in a square frame; PanZoomCanvas not used
          for the matrix because the axes need the labelled axes outside
          the transform. Native scroll handles overflow when the cell
          size hits the lower bound (large presets). */}
      <div
        className="relative flex"
        style={{ height: SQUARE_SIZE + 56, color: body }}
        onMouseLeave={handleMatrixLeave}
      >
        {/* Y-axis label, vertical-rl */}
        <div
          className="flex flex-col items-center justify-center pr-2 text-[10px] font-semibold uppercase tracking-[0.16em]"
          style={{
            color: muted,
            writingMode: 'vertical-rl',
            transform: 'rotate(180deg)',
            width: 28,
          }}
        >
          Orbit
        </div>

        {/* Matrix area with X-axis label below */}
        <div className="flex flex-1 flex-col">
          <div
            className="overflow-auto rounded"
            style={{
              border: `1px solid ${border}`,
              background: surface,
              maxHeight: SQUARE_SIZE,
              maxWidth: SQUARE_SIZE,
              width: 'fit-content',
            }}
          >
            <table
              data-testid="orbit-rep-matrix-grid"
              style={{ borderCollapse: 'collapse' }}
            >
              <tbody>
                {orbitRows.map((row, i) => {
                  const isSelected = i === selectedOrbitIdx;
                  return (
                    <tr key={i}>
                      {cells[i].map((coeff, j) => {
                        const filled = coeff !== null;
                        const isHovered =
                          hover?.i === i && hover?.j === j;
                        const cellBg = filled
                          ? isSelected
                            ? hero
                            : filledCellTint
                          : isHovered
                          ? hoverShade
                          : 'transparent';
                        return (
                          <td
                            key={j}
                            onMouseEnter={(e) => handleCellEnter(i, j, e)}
                            onMouseLeave={handleCellLeave}
                            onClick={() => onSelectOrbit(i)}
                            style={{
                              background: cellBg,
                              borderTop: `1px solid ${border}`,
                              borderLeft: `1px solid ${border}`,
                              width: cellSize,
                              height: cellSize,
                              minWidth: cellSize,
                              cursor: 'pointer',
                              transition:
                                'background 180ms cubic-bezier(0.4, 0, 0.2, 1)',
                              outline: isHovered
                                ? `1px solid ${hero}`
                                : 'none',
                              outlineOffset: -1,
                            }}
                          />
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* X-axis label */}
          <div
            className="mt-2 text-center text-[10px] font-semibold uppercase tracking-[0.16em]"
            style={{
              color: muted,
              maxWidth: matrixWidth + 2,
            }}
          >
            Rep
          </div>
        </div>
      </div>

      {/* Floating tooltip — fixed-position, appears next to the hovered
          cell. Editorial callout vocabulary: surface bg, 1px gray-200
          border, no shadow. */}
      {hover && (() => {
        const i = hover.i;
        const j = hover.j;
        const row = orbitRows[i];
        const rep = reps[j];
        if (!row || !rep) return null;

        const coeff = cells[i][j];
        const filled = coeff !== null;
        const orbitSize = row.orbitSize ?? '?';
        const reachCount = (row.outputs ?? []).length;

        return (
          <div
            data-testid="orbit-rep-matrix-tooltip"
            role="tooltip"
            className="pointer-events-none fixed z-50 max-w-[440px] rounded text-[12px] leading-5"
            style={{
              top: hover.y - 12,
              left: hover.x,
              transform: 'translate(-50%, -100%)',
              background: surface,
              border: `1px solid ${border}`,
              padding: '12px 14px',
              color: body,
              boxShadow: 'none',
            }}
          >
            <div
              className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.16em]"
              style={{ color: filled ? heroMuted : muted }}
            >
              {filled ? 'Orbit · Rep contribution' : 'No projection at this cell'}
            </div>

            {/* Labelled tuples in mono */}
            <div className="font-mono text-[11px] leading-6">
              <div>
                <span style={{ color: muted }}>orbit&nbsp;rep:&nbsp;</span>
                <strong>{labelledTuple(row.repTuple)}</strong>
                <span className="ml-1.5" style={{ color: muted }}>
                  · size {orbitSize}
                  {reachCount > 1 ? ` · branches to ${reachCount} reps` : ''}
                </span>
              </div>
              <div>
                <span style={{ color: muted }}>stored&nbsp;rep:&nbsp;</span>
                <strong>{labelledTuple(rep.tuple)}</strong>
              </div>
            </div>

            {/* LaTeX contribution + English explanation */}
            <div className="mt-2 border-t pt-2" style={{ borderColor: border }}>
              {filled ? (
                <>
                  <div className="overflow-x-auto">
                    <Latex
                      math={String.raw`\#\{m \in \mathrm{orbit}\,${tupleLatex(row.repTuple)} \,:\, \pi_V(m) \in \mathrm{rep}\,${tupleLatex(rep.tuple)}\} = ${coeff}`}
                      display
                    />
                  </div>
                  <p
                    className="mt-1 font-serif text-[12.5px] leading-6"
                    style={{ color: body }}
                  >
                    {coeff === 1 ? 'Exactly one' : `${coeff} of ${orbitSize}`}{' '}
                    member{coeff === 1 ? '' : 's'} of this orbit{' '}
                    {coeff === 1 ? 'projects' : 'project'} (under π<sub>V</sub>,
                    canonicalized by H) to this stored rep — so the orbit{' '}
                    <em>writes to this output bin</em>. Each filled cell adds
                    1 to α regardless of the member count.
                    {reachCount > 1 && (
                      <>
                        {' '}This orbit is <strong>branching</strong>: its{' '}
                        {orbitSize} members map to {reachCount} distinct
                        stored reps (different output bins) under H.
                      </>
                    )}
                  </p>
                </>
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <Latex
                      math={String.raw`\pi_V(\mathrm{orbit}\,${tupleLatex(row.repTuple)}) \,\cap\, \mathrm{rep}\,${tupleLatex(rep.tuple)} = \varnothing`}
                      display
                    />
                  </div>
                  <p
                    className="mt-1 font-serif text-[12.5px] leading-6"
                    style={{ color: body }}
                  >
                    No member of this orbit projects (after H-canonicalization)
                    to this stored rep. The cell stays empty and contributes
                    nothing to α.
                  </p>
                </>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
