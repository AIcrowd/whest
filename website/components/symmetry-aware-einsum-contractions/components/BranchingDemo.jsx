import { useCallback, useMemo, useRef, useState } from 'react';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import Latex from './Latex.jsx';
import OrbitRepMatrix from './branchingViews/OrbitRepMatrix.jsx';
import MatrixHoverTooltip from './branchingViews/MatrixHoverTooltip.jsx';
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
  onHover = null,           // ignored — we no longer propagate hover up
  expressionInfo = null,
  dimensionN = null,
}) {
  // Pin is the only React state on this surface. Hover lives in OrbitRepMatrix's
  // ref + a directly-mutated tooltip DOM node, both bypassing React entirely.
  const [pin, setPin] = useState(/* { row, col, clickX, clickY } | null */ null);
  const [modalOpen, setModalOpen] = useState(false);
  const tooltipRef = useRef(null);
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

  // Clear pin on preset switch (costModel ref change). We use a render-time
  // guard so a stale pin row/col never references rows that no longer exist.
  const lastCostModelRef = useRef(costModel);
  if (lastCostModelRef.current !== costModel) {
    lastCostModelRef.current = costModel;
    if (pin !== null) setPin(null);
  }

  const handlePin = useCallback((nextPin) => {
    setPin(nextPin);
    if (nextPin && onSelectOrbit) onSelectOrbit(nextPin.row);
  }, [onSelectOrbit]);

  const handleDismiss = useCallback(() => setPin(null), []);
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
          tooltipRef={tooltipRef}
          onPin={handlePin}
          pin={pin}
          onExpand={handleOpenModal}
        />
        <div
          className="mt-3 font-mono text-[11px] text-gray-600"
          data-testid="branching-alpha-total"
        >
          across all {liveOrbitRows.length} orbits: α = <strong className="text-gray-900">{liveAlpha}</strong>
        </div>
      </div>

      {/* Imperative hover tooltip — written into via tooltipRef.current.update. */}
      <MatrixHoverTooltip ref={tooltipRef} />

      {/* Floating pin-detail card — visible when pin is set; auto-dismisses
          on Esc or when the matrix scrolls offscreen. */}
      <OrbitDetailCard
        pin={pin}
        orbitRows={liveOrbitRows}
        reps={reps}
        cells={cells}
        expressionInfo={expressionInfo}
        componentInfo={liveComponentInfo}
        onDismiss={handleDismiss}
        matrixRef={matrixRef}
        mode="floating"
      />

      {/* Modal — hosts the same OrbitDetailCard at viewport size. Task 6
          updates the modal's signature; for now we pass the new shape. */}
      <OrbitRepMatrixModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        orbitRows={liveOrbitRows}
        reps={reps}
        cells={cells}
        pin={pin}
        onPin={handlePin}
        expressionInfo={expressionInfo}
        componentInfo={liveComponentInfo}
        selectedOrbitIdx={selectedOrbitIdx}
        onSelectOrbit={onSelectOrbit}
      />
    </div>
  );
}
