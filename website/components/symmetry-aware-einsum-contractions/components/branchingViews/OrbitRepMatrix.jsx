import { useMemo, useState } from 'react';
import Latex from '../Latex.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../../lib/explorerTheme.js';
import { notationColor } from '../../lib/notationSystem.js';
import {
  canonicalTupleUnderGroup,
  visibleTupleFromFullTuple,
} from '../../engine/outputOrbit.js';

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

// Canonical-form LaTeX for the einsum equation (the same form shown in the
// Section 1 "tensor operation written as one equation" block). Falls back
// to a simple `R[V] = sum_W ...` placeholder if expressionInfo is missing.
function canonicalEquationLatex(expressionInfo) {
  if (!expressionInfo) return null;
  const { subscripts = [], output = '', operandNames = [] } = expressionInfo;
  if (subscripts.length === 0 || output.length === 0) return null;

  // Summed labels = labels in operands but not in output.
  const inOperands = new Set();
  subscripts.forEach((sub) => [...sub].forEach((c) => inOperands.add(c)));
  const outChars = [...output];
  const outSet = new Set(outChars);
  const summedChars = [...inOperands].filter((c) => !outSet.has(c)).sort();

  const opTerms = subscripts.map((sub, i) => {
    const name = operandNames[i] ?? operandNames[0] ?? 'X';
    return `${name}[${[...sub].join(',')}]`;
  });
  const sumPrefix = summedChars.length > 0 ? `\\sum_{${summedChars.join(',')}}\\,` : '';
  return `R[${outChars.join(',')}] \\;=\\; ${sumPrefix}${opTerms.join(' \\cdot ')}`;
}

// LaTeX for one concrete contribution line:
//   R[<output indices for member>] \mathrel{+}=\;<operand_1>[<member indices>] \cdot <operand_2>[...] ...
function memberContributionLatex(member, expressionInfo) {
  if (!expressionInfo || !member) return '';
  const { subscripts = [], output = '', operandNames = [] } = expressionInfo;
  const outIdx = [...output].map((c) => member[c]).join(',');
  const opTerms = subscripts.map((sub, i) => {
    const name = operandNames[i] ?? operandNames[0] ?? 'X';
    const idx = [...sub].map((c) => member[c]).join(',');
    return `${name}[${idx}]`;
  });
  return `R[${outIdx}] \\mathrel{+}{=} ${opTerms.join(' \\cdot ')}`;
}

// Find which orbit members project (under H) to this cell's stored rep.
// Returns the array of member tuples (objects keyed by label name).
function membersProjectingTo(orbit, repTuple, componentInfo) {
  if (!orbit?.orbitTuples || !componentInfo) return [];
  const { labels, vLabels, hElements } = componentInfo;
  if (!labels || !vLabels || !hElements) return [];

  // Target canonical key for this cell.
  const targetVisibleArray = vLabels.map((l) => repTuple?.[l]);
  const targetKey = canonicalTupleUnderGroup(targetVisibleArray, hElements);

  return orbit.orbitTuples.filter((member) => {
    const fullArray = labels.map((l) => member[l]);
    const visiblePositions = vLabels.map((l) => labels.indexOf(l));
    const memberVisible = visibleTupleFromFullTuple(fullArray, visiblePositions);
    const memberKey = canonicalTupleUnderGroup(memberVisible, hElements);
    return memberKey === targetKey;
  });
}

export default function OrbitRepMatrix({
  orbitRows = [],
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,
  expressionInfo = null,
  componentInfo = null,
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
              {filled ? (() => {
                // Canonical einsum form (e.g. R[a,b,c] = sum_i X[i,a]·X[i,b]·X[i,c])
                const canonical = canonicalEquationLatex(expressionInfo);
                // Concrete members of this orbit that project to this stored
                // rep under H. With branching, multiple members can land on
                // the same canonical bin.
                const contributing = membersProjectingTo(row, rep.tuple, componentInfo);
                const allOrbitOutputs = (row.outputs ?? []);
                return (
                  <>
                    {canonical && (
                      <div className="mb-2 overflow-x-auto">
                        <div
                          className="mb-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
                          style={{ color: muted }}
                        >
                          einsum equation
                        </div>
                        <Latex math={canonical} display />
                      </div>
                    )}

                    {/* Concrete expanded contribution. For branching cells
                        with multiple members landing here, render each line
                        separately so the reader sees how the orbit's members
                        collect into this output bin. */}
                    {contributing.length > 0 && expressionInfo ? (
                      <div className="overflow-x-auto">
                        <div
                          className="mb-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
                          style={{ color: muted }}
                        >
                          this orbit&apos;s contribution to this output bin
                        </div>
                        {contributing.map((m, mi) => (
                          <Latex
                            key={mi}
                            math={memberContributionLatex(m, expressionInfo)}
                            display
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="overflow-x-auto">
                        <Latex
                          math={String.raw`\#\{m \in \mathrm{orbit}\,${tupleLatex(row.repTuple)} \,:\, \pi_V(m) \in \mathrm{rep}\,${tupleLatex(rep.tuple)}\} = ${coeff}`}
                          display
                        />
                      </div>
                    )}

                    {/* English summary. For branching cells, additionally
                        show the FULL set of bins this orbit reaches, so
                        readers see the multi-bin collection at a glance. */}
                    <p
                      className="mt-2 font-serif text-[12.5px] leading-6"
                      style={{ color: body }}
                    >
                      {coeff === 1 ? 'Exactly one' : `${coeff} of ${orbitSize}`}{' '}
                      member{coeff === 1 ? '' : 's'} of this orbit{' '}
                      {coeff === 1 ? 'writes' : 'write'} to{' '}
                      <strong>R[{[...(expressionInfo?.output ?? '')].map((c) => rep.tuple[c]).join(', ')}]</strong>
                      {coeff > 1 && expressionInfo && (
                        <>
                          {' '}— different orbit members
                          {' '}<em>collect into the same output bin</em>{' '}
                          because their visible-side projections agree under
                          H = Stab<sub>G</sub>(V)|<sub>V</sub>.
                        </>
                      )}
                      {' '}Each filled cell adds 1 to α regardless of member
                      count.
                      {reachCount > 1 && expressionInfo && (
                        <>
                          {' '}This orbit also reaches{' '}
                          {allOrbitOutputs
                            .filter((o) => JSON.stringify(o.outTuple) !== JSON.stringify(rep.tuple))
                            .map((o, i, arr) => {
                              const bin = [...(expressionInfo.output ?? '')]
                                .map((c) => o.outTuple[c])
                                .join(', ');
                              return (
                                <span key={i}>
                                  <code className="rounded px-1 font-mono text-[11px]" style={{ background: surfaceInset }}>
                                    R[{bin}]
                                  </code>
                                  {i < arr.length - 1 ? ', ' : ''}
                                </span>
                              );
                            })}
                          {' '}— branching means the orbit&apos;s members
                          split across <strong>{reachCount} output bins</strong>.
                        </>
                      )}
                    </p>
                  </>
                );
              })() : (
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
