import { useEffect, useMemo, useRef, useState } from 'react';
import {
  derivePreReps,
  deriveCells,
  layoutFor,
  cellAtPoint,
  SQUARE_FRAME,
} from './orbitRepMatrixLayout.js';

// Token palette anchored to design-system colors_and_type.css.
const COLOR = {
  bg: '#FFFFFF',
  cellGrid: '#ECEFEF',
  cellEmpty: '#FFFFFF',
  cellFilled: 'rgba(240,82,77,0.22)',
  cellHovered: '#F0524D',
  branchOutline: 'rgba(240,82,77,0.45)',
  rowColWash: 'rgba(240,82,77,0.06)',
  border: '#D9DCDC',
};

export default function OrbitRepMatrix({
  orbitRows = [],
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
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setContainerWidth(Math.floor(e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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

  return (
    <div
      ref={containerRef}
      data-testid="orbit-rep-matrix"
      className="w-full"
    >
      <div
        ref={scrollRef}
        className="relative rounded"
        style={{
          border: `1px solid ${COLOR.border}`,
          width: layout.canvasW,
          height: layout.canvasH,
          background: COLOR.bg,
          cursor: 'pointer',
          overflowY: layout.overflowY ? 'auto' : 'hidden',
          overflowX: layout.overflowX ? 'auto' : 'hidden',
          transition: 'background 180ms cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
