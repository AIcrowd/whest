import { useCallback, useMemo, useRef, useState } from 'react';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import Latex from './Latex.jsx';
import OrbitRepMatrix from './branchingViews/OrbitRepMatrix.jsx';
import OrbitDetailCard from './branchingViews/OrbitDetailCard.jsx';
import OrbitRepMatrixModal from './branchingViews/OrbitRepMatrixModal.jsx';
import {
  derivePreReps,
  deriveCells,
} from './branchingViews/orbitRepMatrixLayout.js';
import { restrictStabilizerToPositions } from '../engine/outputOrbit.js';

export default function BranchingDemo({
  componentData,
  costModel,
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,           // legacy; no longer wired
  expressionInfo = null,
  dimensionN = null,
  hoveredLabels = null,
}) {
  // Hover is the only React state on this surface besides `modalOpen`.
  // OrbitRepMatrix's own `hoverRef` still drives the canvas paint without React;
  // BranchingDemo's `hover` mirrors the cell-level transitions only, so the
  // floating card body re-renders once per cell change (via memo).
  const [hover, setHover] = useState(/* { row, col, clickX, clickY } | null */ null);
  const [modalOpen, setModalOpen] = useState(false);
  const matrixRef = useRef(null);

  const liveOrbitRows = costModel?.orbitRows ?? [];
  const reps = useMemo(() => derivePreReps(liveOrbitRows), [liveOrbitRows]);
  const cells = useMemo(() => deriveCells(liveOrbitRows, reps), [liveOrbitRows, reps]);
  const rowFillCounts = useMemo(
    () => cells.map((row) => row.filter((c) => c !== null).length),
    [cells],
  );
  const prefixAlpha = useMemo(() => {
    const prefix = [0];
    for (const count of rowFillCounts) prefix.push(prefix[prefix.length - 1] + count);
    return prefix;
  }, [rowFillCounts]);
  const liveAlpha = prefixAlpha[prefixAlpha.length - 1] ?? 0;

  // V3.1 §15 — α-counter overlay state.
  //   cumulativeMode=false (Total): always show M = total rows, α = total filled cells.
  //   cumulativeMode=true  (Cumulative): show "rows counted so far" / "filled cells
  //     counted so far" — driven by the user's top-to-bottom scan. The hovered row
  //     index acts as the cursor; rows 0..hoveredRow contribute to the running totals.
  const [cumulativeMode, setCumulativeMode] = useState(false);
  const hoveredRow = hover ? hover.row : -1;
  const rowFillCount = (rowIdx) =>
    rowIdx >= 0 && rowIdx < rowFillCounts.length
      ? rowFillCounts[rowIdx]
      : 0;
  // In cumulative mode the visible counters are the running prefix totals as the
  // user moves the cursor down the matrix; absent any hover, they read 0 / 0.
  const cumulativeRows = hoveredRow >= 0 ? hoveredRow + 1 : 0;
  const cumulativeAlpha = prefixAlpha[cumulativeRows] ?? 0;

  const liveComponentInfo = useMemo(() => {
    const c = componentData?.components?.[0];
    if (!c) return null;
    const visiblePositions = c.va.map((l) => c.labels.indexOf(l));
    return {
      labels: c.labels,
      vLabels: c.va,
      visiblePositions,
      elements: c.elements,
      hElements: restrictStabilizerToPositions(c.elements ?? [], visiblePositions),
      dimensionN,
    };
  }, [componentData, dimensionN]);

  // Clear hover on preset switch (costModel ref change). We use a render-time
  // guard so a stale hover row/col never references rows that no longer exist.
  const lastCostModelRef = useRef(costModel);
  if (lastCostModelRef.current !== costModel) {
    lastCostModelRef.current = costModel;
    if (hover !== null) setHover(null);
  }

  const handleHoverChange = useCallback((nextHover) => {
    setHover(nextHover);
    // Note: we deliberately do NOT call onSelectOrbit on hover — it propagates
    // to ComponentCostView's selectedOrbitIdx, which cascades into cost cards,
    // classification tree, and summary table re-rendering. The original
    // click-pin model only fired this on click; the new hover-driven model
    // would fire it 60+ times/sec on dense presets without this guard.
  }, []);

  const handleDismiss = useCallback(() => setHover(null), []);
  const handleOpenModal = useCallback(() => setModalOpen(true), []);

  if (!componentData || !costModel) return null;

  return (
    <div id="orbit-rep-matrix" className="bg-white p-4 scroll-mt-sticky" ref={matrixRef}>
      <ExplorerSubsectionHeader anchorId="orbit-rep-matrix" labelText="Branching">
        The O <Latex math="\to" /> Q matrix
      </ExplorerSubsectionHeader>

      <p className="explorer-support-prose mt-2">
        Each row is one product orbit <Latex math="O" /> · each column is one stored output representative <Latex math="Q" /> · a filled cell means orbit <Latex math="O" />'s members project onto <Latex math="Q" /> via <Latex math="\pi_V" />. Counting filled cells gives <Latex math="\alpha" />.
      </p>

      {/* Matrix as a full-width figure (no 2-col grid). */}
      <div className="mt-4">
        <OrbitRepMatrix
          orbitRows={liveOrbitRows}
          selectedOrbitIdx={selectedOrbitIdx}
          onSelectOrbit={onSelectOrbit}
          onHover={null}
          expressionInfo={expressionInfo}
          componentInfo={liveComponentInfo}
          onHoverChange={handleHoverChange}
          hover={hover}
          onExpand={handleOpenModal}
          hoveredLabels={hoveredLabels}
        />
        {/* V3.1 §15 — α-counter overlay strip. Renders M (rows) and α (filled
            cells) in either Total or Cumulative mode. Toggle is keyboard
            focusable; hovering a row appends a "selected row contributes …
            to alpha" hint. */}
        <div
          data-testid="branching-alpha-counter-strip"
          className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md border px-3 py-2 font-mono text-[12px]"
          style={{ borderColor: 'var(--grid-faint)', color: '#1F2526', background: '#F8F9F9' }}
        >
          {cumulativeMode ? (
            <>
              <span data-testid="branching-alpha-counter-m">
                <span className="text-gray-400">rows counted so far </span>
                <strong className="font-semibold">{cumulativeRows}</strong>
                <span className="text-gray-400"> / {liveOrbitRows.length}</span>
              </span>
              <span data-testid="branching-alpha-counter-alpha">
                <span className="text-gray-400">filled cells counted so far </span>
                <strong className="font-semibold">{cumulativeAlpha}</strong>
                <span className="text-gray-400"> / {liveAlpha}</span>
              </span>
            </>
          ) : (
            <>
              <span data-testid="branching-alpha-counter-m">
                <span className="text-gray-400">M = number of rows = </span>
                <strong className="font-semibold">{liveOrbitRows.length}</strong>
              </span>
              <span data-testid="branching-alpha-counter-alpha">
                <span className="text-gray-400">alpha = number of filled cells = </span>
                <strong className="font-semibold">{liveAlpha}</strong>
              </span>
            </>
          )}
          {hoveredRow >= 0 && (
            <span
              data-testid="branching-alpha-counter-hover-hint"
              className="text-gray-500"
            >
              <span className="text-gray-400">— </span>
              selected row contributes <strong className="font-semibold text-gray-700">{rowFillCount(hoveredRow)}</strong> to alpha
            </span>
          )}
          <button
            type="button"
            data-testid="branching-alpha-counter-toggle"
            aria-label="Toggle cumulative versus total counter mode"
            aria-pressed={cumulativeMode}
            onClick={() => setCumulativeMode((v) => !v)}
            className="ml-auto inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-sans font-medium tracking-tight transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-300"
            style={{
              borderColor: '#D9DCDC',
              background: cumulativeMode ? 'var(--coral-light)' : '#FFFFFF',
              color: cumulativeMode ? 'var(--coral)' : '#1F2526',
              cursor: 'pointer',
            }}
          >
            {cumulativeMode ? 'Cumulative' : 'Total'}
            <span aria-hidden="true" className="mx-1 text-gray-400">↔</span>
            <span aria-hidden="true" className="text-gray-400">{cumulativeMode ? 'Total' : 'Cumulative'}</span>
          </button>
        </div>
        {/* Hero α answer — the figure's takeaway. Big numeric + small-caps caption.
            Faint top-divider gives the answer breathing room from the canvas. */}
        <div
          className="mt-6 pt-4 border-t"
          data-testid="branching-alpha-total"
          style={{ borderColor: 'var(--grid-faint)' }}
        >
          <div className="text-[28px] font-semibold tracking-tight text-gray-900 leading-none font-sans">
            α = {liveAlpha}
          </div>
          <div className="mt-1.5 text-[10px] uppercase tracking-[0.18em] text-gray-400 font-sans">
            total updates · across {liveOrbitRows.length} orbit{liveOrbitRows.length === 1 ? '' : 's'}
          </div>
        </div>
      </div>

      {/* Hover-driven floating card. Auto-dismisses on mouse-leave-matrix
          (handled by OrbitRepMatrix calling onHoverChange(null)) or Esc. */}
      {!modalOpen && (
        <OrbitDetailCard
          hover={hover}
          orbitRows={liveOrbitRows}
          reps={reps}
          cells={cells}
          expressionInfo={expressionInfo}
          componentInfo={liveComponentInfo}
          onDismiss={handleDismiss}
          matrixRef={matrixRef}
          mode="floating"
        />
      )}

      <OrbitRepMatrixModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        orbitRows={liveOrbitRows}
        reps={reps}
        cells={cells}
        hover={hover}
        onHoverChange={handleHoverChange}
        expressionInfo={expressionInfo}
        componentInfo={liveComponentInfo}
        selectedOrbitIdx={selectedOrbitIdx}
        onSelectOrbit={onSelectOrbit}
      />
    </div>
  );
}
