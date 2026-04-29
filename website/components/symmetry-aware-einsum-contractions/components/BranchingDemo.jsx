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
  const liveAlpha = cells.reduce(
    (acc, row) => acc + row.filter((c) => c !== null).length,
    0,
  );

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
    <div id="orbit-rep-matrix" className="bg-white p-4 scroll-mt-24" ref={matrixRef}>
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
        />
        <div
          className="mt-3 font-mono text-[11px] text-gray-600"
          data-testid="branching-alpha-total"
        >
          across all {liveOrbitRows.length} orbits: α = <strong className="text-gray-900">{liveAlpha}</strong>
        </div>
      </div>

      {/* Hover-driven floating card. Auto-dismisses on mouse-leave-matrix
          (handled by OrbitRepMatrix calling onHoverChange(null)) or Esc. */}
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
