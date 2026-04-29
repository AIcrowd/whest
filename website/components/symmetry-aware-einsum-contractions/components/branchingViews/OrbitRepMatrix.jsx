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
const COLOR = {
  bg: '#FFFFFF',
  cellGrid: '#ECEFEF',
  cellFilled: 'rgba(240,82,77,0.22)',
  cellHovered: '#F0524D',
  branchOutline: 'rgba(240,82,77,0.45)',
  rowColWash: 'rgba(240,82,77,0.06)',
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

  // Imperative canvas draw.
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
    ctx.scale(dpr, dpr);

    // 1. Clear.
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, w, h);

    // 2. Row + col wash for focused cell (pin or hover).
    const focus = pin || hover;
    if (focus) {
      ctx.fillStyle = COLOR.rowColWash;
      ctx.fillRect(0, focus.row * layout.cellSize, w, layout.cellSize);
      ctx.fillRect(focus.col * layout.cellSize, 0, layout.cellSize, h);
    }

    // 3. Cells: filled fills, then grid lines.
    for (let r = 0; r < orbitRows.length; r += 1) {
      for (let c = 0; c < reps.length; c += 1) {
        const coeff = cells[r][c];
        const filled = coeff !== null;
        const x = c * layout.cellSize;
        const y = r * layout.cellSize;
        const isFocus = focus && focus.row === r && focus.col === c;
        if (filled) {
          ctx.fillStyle = isFocus ? COLOR.cellHovered : COLOR.cellFilled;
          ctx.fillRect(x, y, layout.cellSize, layout.cellSize);
        }
        ctx.strokeStyle = COLOR.cellGrid;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, layout.cellSize - 1, layout.cellSize - 1);
      }
    }

    // 4. Branching outlines for orbit's other reached cells in focused row.
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
    const cell = pointToCell(pointerCoords(e));
    setHover(cell);
    if (onHover) onHover(cell);
  }
  function handleMouseLeave() {
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

  // Derive role-coded labels from componentInfo (stubbed in Task 3, used here).
  const focused = pin || hover;
  const yTuple = focused ? labelledTuple(orbitRows[focused.row]?.repTuple) : null;
  const xTuple = focused ? labelledTuple(reps[focused.col]?.tuple) : null;
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

      {/* Canvas frame with axis labels + tuple bands */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: '22px 90px minmax(0, 1fr)',
          gridTemplateRows: `${layout.canvasH}px 26px 22px`,
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

        {/* Y tuple band — visible only when a cell is focused */}
        <div
          data-testid="orbit-rep-matrix-y-band"
          style={{
            gridColumn: 2, gridRow: 1,
            background: yTuple ? 'rgba(240,82,77,0.06)' : 'transparent',
            borderRight: yTuple ? '1px solid rgba(240,82,77,0.25)' : 'none',
            writingMode: 'vertical-rl',
            transform: 'rotate(180deg)',
            color: '#B23E3A',
          }}
          className="flex items-center justify-center font-mono text-[11px] font-semibold"
        >
          {yTuple || ''}
        </div>

        {/* Canvas — keep the existing scroll wrapper here */}
        <div
          ref={scrollRef}
          style={{
            gridColumn: 3, gridRow: 1,
            width: layout.canvasW, height: layout.canvasH,
            background: COLOR.bg,
            border: `1px solid ${COLOR.border}`,
            borderRadius: 4,
            overflowY: layout.overflowY ? 'auto' : 'hidden',
            overflowX: layout.overflowX ? 'auto' : 'hidden',
            cursor: 'pointer',
            transition: 'background 180ms cubic-bezier(0.4, 0, 0.2, 1)',
          }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
        >
          <canvas ref={canvasRef} />
        </div>

        {/* X tuple band — visible only when a cell is focused */}
        <div
          data-testid="orbit-rep-matrix-x-band"
          style={{
            gridColumn: 3, gridRow: 2,
            background: xTuple ? 'rgba(240,82,77,0.06)' : 'transparent',
            borderTop: xTuple ? '1px solid rgba(240,82,77,0.25)' : 'none',
            color: '#B23E3A',
          }}
          className="flex items-center px-1 font-mono text-[11px] font-semibold"
        >
          {xTuple ? (
            <span style={{ marginLeft: focused ? `${focused.col * layout.cellSize - 14}px` : 0 }}>
              {xTuple}
            </span>
          ) : ''}
        </div>

        {/* X axis label */}
        <div
          style={{ gridColumn: 3, gridRow: 3, color: '#1F2526' }}
          className="text-center text-[10px] font-semibold uppercase tracking-[0.16em] font-sans"
        >
          Rep <Latex math="Q" />
        </div>
      </div>
    </div>
  );
}
