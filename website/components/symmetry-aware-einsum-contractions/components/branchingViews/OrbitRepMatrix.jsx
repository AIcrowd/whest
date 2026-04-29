import { memo, useEffect, useMemo, useRef, useState } from 'react';
import Latex from '../Latex.jsx';
import {
  derivePreReps,
  deriveCells,
  layoutFor,
  cellAtPoint,
  labelledTuple,
  SQUARE_FRAME,
} from './orbitRepMatrixLayout.js';

// Token palette anchored to design-system colors_and_type.css.
// `cellGrid` is near-invisible by design — the grid is a structural hint, not chrome.
// `cellFilled` is now bright enough to read at small cell sizes; `cellPinned` is the
// stronger "this cell is the reading anchor" treatment when click-pinned.
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
  // selectedOrbitIdx, expressionInfo, componentInfo are read by Tasks 4-9
  // (axis labels, label legend chip, worked-example panel) — stubbed here
  // so the prop API stays stable across the multi-task redesign.
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,
  expressionInfo = null,
  componentInfo = null,
  /** ({hover, pin}) => void — surfaces both slots so BranchingDemo's
   *  WorkedExamplePanel can render the right-side worked example. */
  onStateChange = null,
}) {
  const canvasRef = useRef(null);
  const offscreenRef = useRef(null); // cached base layer (grid + filled cells)
  const containerRef = useRef(null);
  const rafRef = useRef(null);
  // Hover lives in a ref, not React state. Mousemove updates the ref and
  // schedules a manual rAF paint that reads from the ref. No React re-render
  // per hover — that's what makes hover feel instant on big matrices.
  const hoverRef = useRef(null);
  const [pin, setPin] = useState(/* pin: { row, col } | null */ null);
  const [containerWidth, setContainerWidth] = useState(SQUARE_FRAME);

  const reps = useMemo(() => derivePreReps(orbitRows), [orbitRows]);
  const cells = useMemo(() => deriveCells(orbitRows, reps), [orbitRows, reps]);
  const layout = useMemo(
    () => layoutFor({
      canvasWidth: containerWidth,
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

  // Surface PIN changes to the parent (panel reads pin only — hover is
  // canvas-only, lives in a ref, and never triggers a React render).
  useEffect(() => {
    if (onStateChange) onStateChange({ hover: null, pin });
  }, [pin, onStateChange]);

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
  //   2. BASE   (deps: [layout, cells, pin]) — paint grid + filled cells into
  //                                            an offscreen canvas. Pin is in
  //                                            the base because pinned cells
  //                                            use the stronger coral fill.
  //   3. PAINT  (deps: [layout, hover, pin, cells]) — drawImage(base) + small
  //                                                   branching outlines +
  //                                                   hover marker. <1 ms even
  //                                                   on 18k-cell matrices.
  //
  // Per mousemove only PAINT runs; BASE only runs when the data or pin
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
  // Re-runs only when the data or pin changes (rare).
  useEffect(() => {
    const off = offscreenRef.current;
    if (!off || !layout.cellWidth || !layout.cellHeight) return;
    const ctx = off.getContext('2d');
    const cw = layout.cellWidth;
    const ch = layout.cellHeight;

    // Clear (logical coords; DPR scale already applied in stage 1).
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, layout.contentWidth, layout.contentHeight);

    // Filled cells. Pinned cell gets the stronger coral.
    for (let r = 0; r < orbitRows.length; r += 1) {
      for (let c = 0; c < reps.length; c += 1) {
        const coeff = cells[r][c];
        if (coeff === null) continue;
        const isPin = pin && pin.row === r && pin.col === c;
        ctx.fillStyle = isPin ? COLOR.cellPinned : COLOR.cellFilled;
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
  }, [layout, orbitRows, reps, cells, pin]);

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
    const hover = hoverRef.current;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(off, 0, 0);
    ctx.scale(dpr, dpr);

    const focus = pin || hover;
    if (focus && cw > 2 && ch > 2) {
      ctx.strokeStyle = COLOR.branchOutline;
      ctx.lineWidth = 1;
      for (let c = 0; c < reps.length; c += 1) {
        if (c === focus.col) continue;
        if (cells[focus.row][c] === null) continue;
        ctx.strokeRect(c * cw + 1.5, focus.row * ch + 1.5, cw - 3, ch - 3);
      }
    }

    if (hover && !pin) {
      ctx.strokeStyle = COLOR.hoverMarker;
      ctx.lineWidth = 2;
      ctx.strokeRect(hover.col * cw + 1, hover.row * ch + 1, Math.max(cw - 2, 0), Math.max(ch - 2, 0));
    }
  }

  // After layout/data/pin changes, repaint once so the static base + any
  // existing pin overlay re-renders. (Hover refreshes happen via mousemove.)
  useEffect(() => {
    paintOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, orbitRows, reps, cells, pin]);

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
  }
  function handleClick(e) {
    const cell = pointToCell(pointerCoords(e));
    if (!cell) return;
    if (pin && pin.row === cell.row && pin.col === cell.col) {
      setPin(null);
    } else {
      setPin(cell);
      onSelectOrbit(cell.row);
    }
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
  // the always-visible WorkedExamplePanel on the right of BranchingDemo.
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

      {/* Canvas frame with permanent axis labels — tuple values appear in
          the WorkedExamplePanel on the right of BranchingDemo. Fixed-size
          square canvas with rectangular cells when numRows ≠ numCols; no
          internal scroll. */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: '22px minmax(0, 1fr)',
          gridTemplateRows: 'auto 22px',
        }}
      >
        {/* Y axis label */}
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

        {/* Canvas — fixed size, no scroll wrapper. */}
        <div
          style={{
            gridColumn: 2, gridRow: 1,
            width: layout.canvasW, height: layout.canvasH,
            background: COLOR.bg,
            border: `1px solid ${COLOR.border}`,
            borderRadius: 4,
            cursor: 'pointer',
          }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
        >
          <canvas ref={canvasRef} />
        </div>

        {/* X axis label */}
        <div
          style={{ gridColumn: 2, gridRow: 2, color: '#1F2526' }}
          className="text-center text-[10px] font-semibold uppercase tracking-[0.16em] font-sans"
        >
          Rep <Latex math="Q" />
        </div>
      </div>

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

// Memoize: BranchingDemo re-renders on every hover/pin update, but
// OrbitRepMatrix's props don't depend on hover/pin (those live in the matrix's
// own internal state). React.memo with default shallow-prop equality skips
// the re-render when the parent re-renders for an unrelated reason.
export default memo(OrbitRepMatrix);
