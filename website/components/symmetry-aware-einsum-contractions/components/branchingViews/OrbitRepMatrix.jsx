import { useEffect, useMemo, useRef, useState } from 'react';
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

export default function OrbitRepMatrix({
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
  const containerRef = useRef(null);
  const scrollRef = useRef(null);
  const rafRef = useRef(null);
  const [hover, setHover] = useState(/* hover: { row, col } | null */ null);
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

  // Surface state changes to the parent.
  useEffect(() => {
    if (onStateChange) onStateChange({ hover, pin });
  }, [hover, pin, onStateChange]);

  // Cancel any in-flight rAF on unmount so we don't fire setHover after teardown.
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Imperative canvas sizing — runs only when layout changes.
  // Split out from drawing because setting canvas.width clears canvas state,
  // which we want to avoid on every hover update.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || layout.cellSize === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const w = layout.contentWidth;
    const h = layout.contentHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    // Reset transform then re-apply DPR scale.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }, [layout]);

  // Imperative canvas draw — runs on layout, data, and focus changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || layout.cellSize === 0) return;
    const ctx = canvas.getContext('2d');
    // Clear in logical coordinates (the DPR scale was already applied in the sizing effect).
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, layout.contentWidth, layout.contentHeight);

    // Cells: filled fills. Pinned cell gets the stronger coral; everything else
    // uses the regular filled tint. Hover gets a 2 px border (drawn after the
    // grid lines below) so it sits on top without obscuring the cell's content.
    for (let r = 0; r < orbitRows.length; r += 1) {
      for (let c = 0; c < reps.length; c += 1) {
        const coeff = cells[r][c];
        const filled = coeff !== null;
        if (!filled) continue;
        const x = c * layout.cellSize;
        const y = r * layout.cellSize;
        const isPin = pin && pin.row === r && pin.col === c;
        ctx.fillStyle = isPin ? COLOR.cellPinned : COLOR.cellFilled;
        ctx.fillRect(x, y, layout.cellSize, layout.cellSize);
      }
    }

    // Grid lines — drawn once each so internal cells aren't double-stroked.
    ctx.strokeStyle = COLOR.cellGrid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let r = 0; r <= orbitRows.length; r += 1) {
      const y = r * layout.cellSize + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(layout.contentWidth, y);
    }
    for (let c = 0; c <= reps.length; c += 1) {
      const x = c * layout.cellSize + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, layout.contentHeight);
    }
    ctx.stroke();

    // Branching outlines for the orbit's other reached cells in the focused row.
    // Visible while either pinning or hovering — they're the "where else does
    // this orbit go" hint.
    const focus = pin || hover;
    if (focus) {
      ctx.strokeStyle = COLOR.branchOutline;
      ctx.lineWidth = 1;
      for (let c = 0; c < reps.length; c += 1) {
        if (c === focus.col) continue;
        if (cells[focus.row][c] === null) continue;
        const x = c * layout.cellSize;
        const y = focus.row * layout.cellSize;
        ctx.strokeRect(x + 1.5, y + 1.5, layout.cellSize - 3, layout.cellSize - 3);
      }
    }

    // Faint cell-level hover marker (only when NOT pinned — pin already has its
    // own stronger fill). 2 px coral border anchors the eye on the cell the
    // mouse is over without re-fillng the cell.
    if (hover && !pin) {
      const x = hover.col * layout.cellSize;
      const y = hover.row * layout.cellSize;
      ctx.strokeStyle = COLOR.hoverMarker;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, layout.cellSize - 2, layout.cellSize - 2);
    }
  }, [layout, orbitRows, reps, cells, hover, pin]);

  // Pointer event helpers.
  function pointerCoords(e) {
    const canvas = canvasRef.current;
    const scrollEl = scrollRef.current;
    if (!canvas || !scrollEl) return null;
    const rect = scrollEl.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }
  function pointToCell(pt) {
    if (!pt) return null;
    const scrollEl = scrollRef.current;
    return cellAtPoint(pt, {
      cellSize: layout.cellSize,
      canvasW: layout.canvasW,
      canvasH: layout.canvasH,
      scrollTop: scrollEl?.scrollTop ?? 0,
      scrollLeft: scrollEl?.scrollLeft ?? 0,
      numRows: orbitRows.length,
      numCols: reps.length,
    });
  }
  function handleMouseMove(e) {
    // Capture the pointer position immediately (the SyntheticEvent is reused).
    const pt = pointerCoords(e);
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const cell = pointToCell(pt);
      setHover((prev) => {
        // Skip the setState if the cell hasn't changed — avoid useless re-renders.
        if (prev === cell) return prev;
        if (prev && cell && prev.row === cell.row && prev.col === cell.col) return prev;
        return cell;
      });
      if (onHover) onHover(cell);
    });
  }
  function handleMouseLeave() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setHover(null);
    if (onHover) onHover(null);
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
          the WorkedExamplePanel on the right of BranchingDemo. */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: '22px minmax(0, 1fr)',
          gridTemplateRows: `${layout.canvasH}px 22px`,
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

        {/* Canvas */}
        <div
          ref={scrollRef}
          style={{
            gridColumn: 2, gridRow: 1,
            width: layout.canvasW, height: layout.canvasH,
            background: COLOR.bg,
            border: `1px solid ${COLOR.border}`,
            borderRadius: 4,
            overflowY: layout.overflowY ? 'auto' : 'hidden',
            overflowX: layout.overflowX ? 'auto' : 'hidden',
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
