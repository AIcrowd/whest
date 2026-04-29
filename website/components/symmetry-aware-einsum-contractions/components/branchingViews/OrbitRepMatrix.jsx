import { useMemo, useState } from 'react';
import PanZoomCanvas from '../PanZoomCanvas.jsx';
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
// /design-system. Specifically:
//
//   - .chip family rule (preview/components.html line 38) — row-header pill
//     uses chip-pill geometry (border-radius 999px, 1px gray-200 border).
//     Selected row uses .chip--coral recipe (coral-light bg, coral-tinted
//     border, coral-hover text) PLUS the design-system "single active-state
//     recipe" 3px coral left-rail stripe.
//   - Eyebrow voice — Inter 10px 600 letterspaced 0.16em (.lbl rule).
//   - Body prose register — Source Serif 4 13px (.callout p register).
//   - Tuple labels — JetBrains Mono 11px (var(--font-mono)).
//   - Borders only where they encode structure (cell grid + sticky-header
//     line). No outer card chrome, no shadow.
//   - Motion — 180 ms hover (var(--dur-fast)). No entrance animations.

function compactTuple(tuple) {
  if (!tuple || typeof tuple !== 'object') return '—';
  return `(${Object.values(tuple).join(',')})`;
}

function labelledTuple(tuple) {
  if (!tuple || typeof tuple !== 'object') return '—';
  return `(${Object.entries(tuple).map(([k, v]) => `${k}=${v}`).join(', ')})`;
}

function tupleKey(tuple) {
  return JSON.stringify(tuple ?? {});
}

const ROTATE_COL_THRESHOLD = 16;

export default function OrbitRepMatrix({
  orbitRows = [],
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,
}) {
  const themeId = getActiveExplorerThemeId();
  const [hover, setHover] = useState(null);

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

  const cells = useMemo(
    () => orbitRows.map((row) => {
      const orbitRepKeys = new Set(
        (row.outputs ?? []).map((o) => tupleKey(o.outTuple)),
      );
      return reps.map((rep) => orbitRepKeys.has(rep.k));
    }),
    [orbitRows, reps],
  );

  const alphaTotal = cells.reduce(
    (acc, row) => acc + row.filter(Boolean).length,
    0,
  );

  const rotateColLabels = reps.length >= ROTATE_COL_THRESHOLD;

  // Theme tokens. Bound to the design-system palette.
  const headerBg = explorerThemeColor(themeId, 'surfaceInset');
  const headerText = explorerThemeColor(themeId, 'muted');
  const muted = explorerThemeColor(themeId, 'muted');
  const body = explorerThemeColor(themeId, 'body');
  const border = explorerThemeColor(themeId, 'border');
  const surface = explorerThemeColor(themeId, 'surface');
  const heroLight = explorerThemeTint(themeId, 'hero', 0.08);
  const heroLightStrong = explorerThemeTint(themeId, 'hero', 0.18);
  const heroBorder = explorerThemeTint(themeId, 'hero', 0.25);
  const hero = explorerThemeColor(themeId, 'hero');
  const heroMuted = explorerThemeColor(themeId, 'heroMuted');
  const hoverShade = explorerThemeTint(themeId, 'hero', 0.04);
  const filledCell = explorerThemeTint(themeId, 'hero', 0.22);
  const filledCellSelected = hero;
  const alphaAccent = notationColor('alpha_total');

  // Empty state — no orbit data yet (e.g. preset is still loading).
  if (orbitRows.length === 0) {
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

  function rowHover(i) {
    setHover({ kind: 'row', i });
    if (onHover) {
      const labels = Object.keys(orbitRows[i]?.repTuple ?? {});
      onHover({ labels, leafKeys: [] });
    }
  }
  function colHover(j) {
    setHover({ kind: 'col', j });
  }
  function clearHover() {
    setHover(null);
    if (onHover) onHover(null);
  }

  return (
    <div data-testid="orbit-rep-matrix" className="w-full">
      {/* Reading guide — collapsed by default. Eyebrow voice on the summary
          (Inter 10px 600 letterspaced 0.16em); body in Source Serif 4 13px. */}
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
            <strong className="font-semibold">Rows</strong> — product orbits
            (M total). Each row label is the orbit&apos;s representative tuple
            of label values.
          </li>
          <li>
            <strong className="font-semibold">Columns</strong> — stored output
            representatives (|Y/H| total). Each column label is a visible-side
            tuple, canonicalized under H = Stab<sub>G</sub>(V)|<sub>V</sub>.
          </li>
          <li>
            <strong className="font-semibold">Filled cell</strong> — that
            orbit projects to that rep at least once. <em>Counting filled
            cells</em> = α (the accumulation count).
          </li>
          <li>
            <strong className="font-semibold">Multi-cell rows</strong> —
            branching: one orbit reaches multiple stored reps.
            Functional-projection presets render one filled cell per row
            (α = M).
          </li>
        </ul>
      </details>

      {/* α total readout. Sanity-checks against BranchingDemo's liveAlpha. */}
      <div
        className="mb-2 flex items-center gap-2 text-[11px]"
        style={{ color: muted }}
      >
        <span className="font-semibold uppercase tracking-[0.16em]">
          {orbitRows.length} orbits × {reps.length} reps
        </span>
        <span className="ml-auto font-mono">
          α ={' '}
          <strong style={{ color: alphaAccent }}>{alphaTotal}</strong>
          <span className="ml-1 text-[10px]" style={{ color: muted }}>
            (count of filled cells)
          </span>
        </span>
      </div>

      {/* PanZoomCanvas wrapper — same gesture contract as
          LabelInteractionGraph: ⌘-scroll zoom, drag pan, double-click reset,
          corner buttons. */}
      <div
        className="rounded"
        style={{ border: `1px solid ${border}`, background: surface }}
      >
        <PanZoomCanvas
          className="h-[360px]"
          ariaLabel="Orbit-rep matrix (zoomable, drag to pan)"
        >
          <table
            className="font-mono text-[11px]"
            style={{ borderCollapse: 'collapse', background: surface }}
            onMouseLeave={clearHover}
          >
            <thead>
              <tr>
                <th
                  className="text-[10px] font-semibold uppercase tracking-[0.16em]"
                  style={{
                    background: headerBg,
                    color: headerText,
                    position: 'sticky',
                    top: 0,
                    left: 0,
                    zIndex: 3,
                    padding: '6px 10px',
                    borderBottom: `1px solid ${border}`,
                    borderRight: `1px solid ${border}`,
                    minWidth: 110,
                    textAlign: 'left',
                  }}
                >
                  orbit \ rep
                </th>
                {reps.map((rep, j) => {
                  const isHovered =
                    hover?.kind === 'col' && hover?.j === j;
                  return (
                    <th
                      key={rep.k}
                      onMouseEnter={() => colHover(j)}
                      title={`stored output rep: ${labelledTuple(rep.tuple)}`}
                      style={{
                        background: isHovered ? hoverShade : headerBg,
                        color: headerText,
                        position: 'sticky',
                        top: 0,
                        zIndex: 1,
                        padding: rotateColLabels ? '8px 2px' : '6px 8px',
                        borderBottom: `1px solid ${border}`,
                        borderLeft: `1px solid ${border}`,
                        height: rotateColLabels ? 76 : 32,
                        minWidth: rotateColLabels ? 18 : 56,
                        whiteSpace: 'nowrap',
                        textAlign: 'center',
                        fontWeight: 500,
                        transition: 'background 180ms cubic-bezier(0.4, 0, 0.2, 1)',
                        ...(rotateColLabels
                          ? {
                              writingMode: 'vertical-rl',
                              transform: 'rotate(180deg)',
                            }
                          : {}),
                      }}
                    >
                      {compactTuple(rep.tuple)}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {orbitRows.map((row, i) => {
                const isSelected = i === selectedOrbitIdx;
                const isRowHovered = hover?.kind === 'row' && hover?.i === i;
                const rowBg = isSelected
                  ? heroLight
                  : isRowHovered
                  ? hoverShade
                  : 'transparent';
                return (
                  <tr key={i} style={{ background: rowBg }}>
                    {/* Row header — chip-pill geometry. Selected row uses
                        the .chip--coral recipe + 3px coral left-rail stripe
                        per the design-system "single active-state recipe." */}
                    <th
                      onClick={() => onSelectOrbit(i)}
                      onMouseEnter={() => rowHover(i)}
                      title={`orbit rep: ${labelledTuple(row.repTuple)}`}
                      style={{
                        background: isSelected ? heroLight : headerBg,
                        color: isSelected ? heroMuted : headerText,
                        position: 'sticky',
                        left: 0,
                        zIndex: 2,
                        padding: '6px 10px 6px 14px',
                        borderTop: `1px solid ${border}`,
                        borderRight: `1px solid ${border}`,
                        boxShadow: isSelected
                          ? `inset 3px 0 0 0 ${hero}`
                          : 'none',
                        textAlign: 'left',
                        fontWeight: isSelected ? 600 : 500,
                        whiteSpace: 'nowrap',
                        cursor: 'pointer',
                        transition:
                          'background 180ms cubic-bezier(0.4, 0, 0.2, 1)',
                      }}
                    >
                      {compactTuple(row.repTuple)}
                    </th>
                    {cells[i].map((filled, j) => {
                      const isColHovered =
                        hover?.kind === 'col' && hover?.j === j;
                      const cellBg = filled
                        ? isSelected
                          ? filledCellSelected
                          : filledCell
                        : isColHovered || isRowHovered
                        ? hoverShade
                        : 'transparent';
                      return (
                        <td
                          key={j}
                          onMouseEnter={() => {
                            setHover({ kind: 'cell', i, j });
                          }}
                          onClick={() => onSelectOrbit(i)}
                          title={
                            filled
                              ? `orbit ${labelledTuple(row.repTuple)} → rep ${labelledTuple(reps[j].tuple)}`
                              : `orbit ${labelledTuple(row.repTuple)} does not project to rep ${labelledTuple(reps[j].tuple)}`
                          }
                          style={{
                            background: cellBg,
                            borderTop: `1px solid ${border}`,
                            borderLeft: `1px solid ${border}`,
                            width: 18,
                            height: 18,
                            minWidth: 18,
                            cursor: 'pointer',
                            transition:
                              'background 180ms cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        />
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </PanZoomCanvas>
      </div>
    </div>
  );
}
