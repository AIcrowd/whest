import { memo, useEffect, useMemo, useRef, useState } from 'react';
import Latex from '../Latex.jsx';
import {
  derivePreReps,
  deriveCells,
  layoutFor,
  cellAtPoint,
  labelledTuple,
  computeAxisTicks,
  SQUARE_FRAME,
  FIXED_CANVAS_HEIGHT,
} from './orbitRepMatrixLayout.js';

// Token palette anchored to design-system colors_and_type.css.
// `cellGrid` is near-invisible by design — the grid is a structural hint, not chrome.
// `cellFilled` is now bright enough to read at small cell sizes; `cellPinned` is the
// stronger "this cell is the reading anchor" treatment when hover-focused.
const COLOR = {
  bg: '#FFFFFF',
  cellGrid: '#F4F6F6',
  cellFilled: 'rgba(240, 82, 77, 0.55)',
  cellPinned: '#F0524D',
  hoverMarker: '#F0524D',
  branchOutline: 'rgba(240, 82, 77, 0.55)',
  border: '#D9DCDC',
};

function OrbitRepMatrix({
  orbitRows = [],
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,
  expressionInfo = null,
  componentInfo = null,
  /** ({row, col, clickX, clickY}|null) => void — fires when hover enters a new cell.
   *  clickX/clickY are pointer clientX/clientY (kept for naming continuity with the
   *  floating-card flip math, even though no click is involved). */
  onHoverChange = null,
  /** ({row, col, clickX, clickY}|null) — current hover (controlled). Drives the
   *  strong-coral fill on the canvas + the position of the floating card. */
  hover = null,
  /** () => void — opens the matrix in a viewport-sized modal. The trigger
   *  button is rendered as a prominent overlay in the canvas's top-right. */
  onExpand = null,
}) {
  const canvasRef = useRef(null);
  const offscreenRef = useRef(null); // cached base layer (grid + filled cells)
  const containerRef = useRef(null);
  const rafRef = useRef(null);
  // Hover lives in a ref, not React state. Mousemove updates the ref and
  // schedules a manual rAF paint that reads from the ref. No React re-render
  // per hover — that's what makes hover feel instant on big matrices.
  const hoverRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(SQUARE_FRAME);

  const reps = useMemo(() => derivePreReps(orbitRows), [orbitRows]);
  const cells = useMemo(() => deriveCells(orbitRows, reps), [orbitRows, reps]);
  const layout = useMemo(
    () => layoutFor({
      canvasWidth: containerWidth,
      canvasHeight: FIXED_CANVAS_HEIGHT,
      numRows: orbitRows.length,
      numCols: reps.length,
    }),
    [containerWidth, orbitRows.length, reps.length],
  );

  // Observe container width — keep cell sizing responsive.
  // Dep `orbitRows.length === 0` is a boolean: re-runs only when the
  // empty/populated branch flips, so the observer re-attaches to the
  // newly-mounted wrapper instead of the discarded empty-state node.
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setContainerWidth(Math.floor(e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [orbitRows.length === 0]);

  // Cancel any in-flight rAF on unmount.
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ─── Three-stage draw pipeline ────────────────────────────────────────────
  //
  // Drawing 18k+ cells on every mousemove is too expensive — the user feels it
  // as sluggishness. We split into three stages so per-hover cost is tiny:
  //
  //   1. SIZING (deps: [layout])      — set canvas + offscreen dimensions, DPR.
  //   2. BASE   (deps: [layout, cells, hover]) — paint grid + filled cells into
  //                                              an offscreen canvas. Hover is in
  //                                              the base because hovered cells
  //                                              use the stronger coral fill.
  //   3. PAINT  (deps: [layout, hover, cells]) — drawImage(base) + small
  //                                              branching outlines +
  //                                              hover marker. <1 ms even
  //                                              on 18k-cell matrices.
  //
  // Per mousemove only PAINT runs; BASE only runs when the data or hover
  // changes, which is rare relative to mousemove frequency.

  // Stage 1: SIZING — runs only when layout changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout.cellWidth || !layout.cellHeight) return;
    const dpr = window.devicePixelRatio || 1;
    const w = layout.contentWidth;
    const h = layout.contentHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Mirror the offscreen canvas's dimensions so drawImage maps 1:1.
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }
    const off = offscreenRef.current;
    off.width = canvas.width;
    off.height = canvas.height;
    const offCtx = off.getContext('2d');
    offCtx.setTransform(1, 0, 0, 1, 0, 0);
    offCtx.scale(dpr, dpr);
  }, [layout]);

  // Stage 2: BASE — paint grid + filled cells into the offscreen canvas.
  // Re-runs only when the data or hover changes (rare).
  useEffect(() => {
    const off = offscreenRef.current;
    if (!off || !layout.cellWidth || !layout.cellHeight) return;
    const ctx = off.getContext('2d');
    const cw = layout.cellWidth;
    const ch = layout.cellHeight;

    // Clear (logical coords; DPR scale already applied in stage 1).
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, layout.contentWidth, layout.contentHeight);

    // Filled cells. Focused (hovered) cell gets the stronger coral.
    for (let r = 0; r < orbitRows.length; r += 1) {
      for (let c = 0; c < reps.length; c += 1) {
        const coeff = cells[r][c];
        if (coeff === null) continue;
        const isFocus = hover && hover.row === r && hover.col === c;
        ctx.fillStyle = isFocus ? COLOR.cellPinned : COLOR.cellFilled;
        ctx.fillRect(c * cw, r * ch, cw, ch);
      }
    }

    // Grid lines — single pass, each line drawn once (symmetric). Skip when
    // cells are too small to leave room for a 1-px grid line (≤ 2 px cells:
    // grid would dominate the cell).
    if (cw > 2 && ch > 2) {
      ctx.strokeStyle = COLOR.cellGrid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let r = 0; r <= orbitRows.length; r += 1) {
        const y = r * ch + 0.5;
        ctx.moveTo(0, y);
        ctx.lineTo(layout.contentWidth, y);
      }
      for (let c = 0; c <= reps.length; c += 1) {
        const x = c * cw + 0.5;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, layout.contentHeight);
      }
      ctx.stroke();
    }
  }, [layout, orbitRows, reps, cells, hover]);

  // Stage 3: PAINT — composite the cached base + dynamic overlays.
  // Imperative, called from event handlers via rAF. Reads hover from a ref
  // so React never re-renders for hover. <1 ms per call even on big matrices.
  function paintOverlay() {
    const canvas = canvasRef.current;
    const off = offscreenRef.current;
    if (!canvas || !off || !layout.cellWidth || !layout.cellHeight) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const cw = layout.cellWidth;
    const ch = layout.cellHeight;
    const hoverCell = hoverRef.current;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(off, 0, 0);
    ctx.scale(dpr, dpr);

    const focus = hoverCell;
    if (focus && cw > 2 && ch > 2) {
      ctx.strokeStyle = COLOR.branchOutline;
      ctx.lineWidth = 1;
      for (let c = 0; c < reps.length; c += 1) {
        if (c === focus.col) continue;
        if (cells[focus.row][c] === null) continue;
        ctx.strokeRect(c * cw + 1.5, focus.row * ch + 1.5, cw - 3, ch - 3);
      }
    }

    if (hoverCell) {
      ctx.strokeStyle = COLOR.hoverMarker;
      ctx.lineWidth = 2;
      ctx.strokeRect(hoverCell.col * cw + 1, hoverCell.row * ch + 1, Math.max(cw - 2, 0), Math.max(ch - 2, 0));
    }
  }

  // After layout/data/hover changes, repaint once so the static base + any
  // existing hover overlay re-renders. (Hover refreshes happen via mousemove.)
  useEffect(() => {
    paintOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, orbitRows, reps, cells, hover]);

  // Pointer event helpers. The canvas is fixed-size with no internal scroll,
  // so coords are simply mouse-relative-to-canvas.
  function pointerCoords(e) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }
  function pointToCell(pt) {
    if (!pt) return null;
    return cellAtPoint(pt, {
      cellWidth: layout.cellWidth,
      cellHeight: layout.cellHeight,
      canvasW: layout.canvasW,
      canvasH: layout.canvasH,
      numRows: orbitRows.length,
      numCols: reps.length,
    });
  }
  function handleMouseMove(e) {
    // Capture pointer immediately (SyntheticEvent gets reused).
    const pt = pointerCoords(e);
    const clientX = e.clientX;
    const clientY = e.clientY;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const cell = pointToCell(pt);
      const prev = hoverRef.current;
      // Skip if cell hasn't changed.
      if (prev === cell) return;
      if (prev && cell && prev.row === cell.row && prev.col === cell.col) return;
      hoverRef.current = cell;
      paintOverlay();

      // Surface the hover-cell change to the parent so the floating card
      // updates content + position.
      if (onHoverChange) {
        if (cell) {
          onHoverChange({ row: cell.row, col: cell.col, clickX: clientX, clickY: clientY });
          if (onSelectOrbit) onSelectOrbit(cell.row);
        } else {
          onHoverChange(null);
        }
      }
    });
  }
  function handleMouseLeave() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (hoverRef.current !== null) {
      hoverRef.current = null;
      paintOverlay();
    }
    if (onHoverChange) onHoverChange(null);
  }

  if (orbitRows.length === 0) {
    return (
      <div
        data-testid="orbit-rep-matrix-empty"
        ref={containerRef}
        className="flex h-[260px] w-full items-center justify-center text-[12px] text-gray-400"
      >
        no orbit data for this preset
      </div>
    );
  }

  // Derive role-coded labels from componentInfo for the legend chip row.
  // Tuple values themselves don't appear on the axes anymore — they live in
  // the OrbitDetailCard (rendered on hover or in the modal).
  const allLabels = orbitRows.length > 0 ? Object.keys(orbitRows[0].repTuple ?? {}) : [];
  const vLabelSet = new Set(componentInfo?.vLabels ?? []);
  const dimensionN = componentInfo?.dimensionN ?? null;

  return (
    <div
      ref={containerRef}
      data-testid="orbit-rep-matrix"
      className="w-full"
    >
      {/* Label legend chip row */}
      <div
        data-testid="orbit-rep-matrix-legend"
        className="mb-3 flex flex-wrap items-baseline gap-2 text-[11px] text-gray-600"
      >
        <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400">labels</span>
        {allLabels.map((l) => {
          const isV = vLabelSet.has(l);
          return (
            <span
              key={l}
              className="inline-flex items-center gap-1 rounded-full border px-2 py-[2px] font-mono"
              style={{
                background: isV ? 'rgba(74,124,255,0.08)' : 'rgba(100,116,139,0.08)',
                color: isV ? '#4A7CFF' : '#64748B',
                borderColor: isV ? 'rgba(74,124,255,0.30)' : 'rgba(100,116,139,0.30)',
              }}
            >
              {l}
              <span className="ml-1 text-[9px] tracking-[0.12em] text-gray-400">{isV ? 'V' : 'W'}</span>
            </span>
          );
        })}
        {dimensionN !== null && (
          <span className="ml-auto font-mono text-[10px] text-gray-600">
            n = <strong className="text-gray-900">{dimensionN}</strong>
            <span className="ml-1 text-gray-400">{`· values {0..${dimensionN - 1}}`}</span>
          </span>
        )}
      </div>

      {/* Canvas frame with permanent axis labels + tick gutters.
          Grid: 3 columns (axis-label | tick-gutter | canvas),
                3 rows  (canvas | tick-row | axis-label-row).
          Tuple values appear in the OrbitDetailCard (on hover or in the modal). */}
      {(() => {
        const yTicks = computeAxisTicks(orbitRows.length, 6);
        const xTicks = computeAxisTicks(reps.length, 6);
        return (
          <div
            className="grid"
            style={{
              gridTemplateColumns: '20px 28px minmax(0, 1fr)',
              gridTemplateRows: 'auto 18px 18px',
            }}
          >
            {/* Y axis label — col 1, row 1 */}
            <div
              style={{
                gridColumn: 1, gridRow: 1,
                writingMode: 'vertical-rl',
                transform: 'rotate(180deg)',
                color: '#1F2526',
              }}
              className="flex items-center justify-center text-[10px] font-semibold uppercase tracking-[0.16em] font-sans"
            >
              Orbit <Latex math="O" />
            </div>

            {/* Y tick gutter — col 2, row 1 */}
            <div
              style={{ gridColumn: 2, gridRow: 1, position: 'relative' }}
              aria-hidden="true"
              data-testid="orbit-rep-matrix-y-ticks"
            >
              {yTicks.map((rowIdx) => {
                const y = (rowIdx + 0.5) * layout.cellHeight;
                return (
                  <span
                    key={rowIdx}
                    className="font-mono"
                    style={{
                      position: 'absolute',
                      right: 6,
                      top: y,
                      transform: 'translateY(-50%)',
                      fontSize: 9,
                      color: '#9AA0A0',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {rowIdx}
                  </span>
                );
              })}
            </div>

            {/* Canvas — col 3, row 1. Fixed size, no scroll wrapper.
                Chart-style axes: left border (Y-axis line) + bottom border
                (X-axis line) only. box-sizing: content-box so the canvas
                fits inside the wrapper at exact `canvasW × canvasH`. */}
            <div
              style={{
                gridColumn: 3, gridRow: 1,
                position: 'relative',
                boxSizing: 'content-box',
                width: layout.canvasW, height: layout.canvasH,
                background: COLOR.bg,
                borderLeft: `1px solid ${COLOR.border}`,
                borderBottom: `1px solid ${COLOR.border}`,
                cursor: 'default',
              }}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              <canvas ref={canvasRef} />
              {onExpand && (
                <button
                  type="button"
                  data-action="open-modal"
                  onClick={(e) => { e.stopPropagation(); onExpand(); }}
                  className="absolute top-2 right-2 inline-flex items-center gap-1.5 rounded-md border bg-white px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] transition-colors hover:bg-gray-50"
                  style={{
                    color: '#B23E3A',
                    borderColor: 'rgba(240,82,77,0.45)',
                    cursor: 'pointer',
                  }}
                  aria-label="Expand matrix to full screen"
                >
                  expand <span aria-hidden="true">↗</span>
                </button>
              )}
            </div>

            {/* X tick gutter — col 3, row 2 */}
            <div
              style={{ gridColumn: 3, gridRow: 2, position: 'relative' }}
              aria-hidden="true"
              data-testid="orbit-rep-matrix-x-ticks"
            >
              {xTicks.map((colIdx) => {
                const x = (colIdx + 0.5) * layout.cellWidth;
                return (
                  <span
                    key={colIdx}
                    className="font-mono"
                    style={{
                      position: 'absolute',
                      top: 4,
                      left: x,
                      transform: 'translateX(-50%)',
                      fontSize: 9,
                      color: '#9AA0A0',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {colIdx}
                  </span>
                );
              })}
            </div>

            {/* X axis label — col 3, row 3 */}
            <div
              style={{ gridColumn: 3, gridRow: 3, color: '#1F2526' }}
              className="text-center text-[10px] font-semibold uppercase tracking-[0.16em] font-sans pt-1"
            >
              Rep <Latex math="Q" />
            </div>
          </div>
        );
      })()}

      {/* Off-screen mirror table for screen-reader / keyboard a11y. */}
      <table className="sr-only" aria-label="The O → Q matrix">
        <thead>
          <tr>
            <th scope="col">Orbit</th>
            {reps.map((rep, c) => (
              <th key={c} scope="col">{labelledTuple(rep.tuple)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {orbitRows.map((row, r) => (
            <tr key={r}>
              <th scope="row">{labelledTuple(row.repTuple)}</th>
              {reps.map((rep, c) => {
                const coeff = cells[r][c];
                const filled = coeff !== null;
                return (
                  <td
                    key={c}
                    aria-label={`${labelledTuple(row.repTuple)} ${filled ? 'reaches' : 'does not reach'} ${labelledTuple(rep.tuple)}${filled ? `, contributes ${coeff} update${coeff === 1 ? '' : 's'} to alpha` : ''}`}
                  >
                    {filled ? '●' : ''}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Memoize: BranchingDemo re-renders when hover state changes, but
// OrbitRepMatrix's props are stable objects (orbitRows, componentInfo, etc.).
// React.memo with default shallow-prop equality skips the re-render when the
// parent re-renders for an unrelated reason.
export default memo(OrbitRepMatrix);
