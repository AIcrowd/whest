import { useCallback, useMemo, useState } from 'react';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import Latex from './Latex.jsx';
import OrbitRepMatrix from './branchingViews/OrbitRepMatrix.jsx';
import WorkedExamplePanel from './branchingViews/WorkedExamplePanel.jsx';
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
  onHover = null,
  expressionInfo = null,
  dimensionN = null,
}) {
  const [hover, setHover] = useState(/* hover: { row, col } | null */ null);
  const [pin, setPin] = useState(/* pin: { row, col } | null */ null);
  const [modalOpen, setModalOpen] = useState(false);

  const liveOrbitRows = costModel?.orbitRows ?? [];
  const reps = useMemo(() => derivePreReps(liveOrbitRows), [liveOrbitRows]);
  const cells = useMemo(() => deriveCells(liveOrbitRows, reps), [liveOrbitRows, reps]);
  const liveAlpha = cells.reduce(
    (acc, row) => acc + row.filter((c) => c !== null).length,
    0,
  );

  // Derive H = Stab_G(V)|_V for the active component, and pull V labels.
  // Used by WorkedExamplePanel + OrbitRepMatrix legend chip to render the
  // role-coded labels and the per-member projection ledger.
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

  // Stable callback identities so OrbitRepMatrix's onStateChange effect
  // doesn't re-fire on every parent render (Task 3 review caveat).
  const handleStateChange = useCallback(({ hover: h, pin: p }) => {
    setHover(h);
    setPin(p);
    if (onHover && h) {
      const labels = Object.keys(liveOrbitRows[h.row]?.repTuple ?? {});
      onHover({ labels, leafKeys: [] });
    } else if (onHover && !h) {
      onHover(null);
    }
    if (p && onSelectOrbit) onSelectOrbit(p.row);
  }, [liveOrbitRows, onHover, onSelectOrbit]);

  const handleClearPin = useCallback(() => {
    setPin(null);
  }, []);

  if (!componentData || !costModel) return null;

  return (
    <div id="orbit-rep-matrix" className="bg-white p-4 scroll-mt-24">
      {/* Header row: subsection title + expand trigger */}
      <div className="flex items-baseline justify-between gap-3">
        <ExplorerSubsectionHeader anchorId="orbit-rep-matrix" labelText="Branching">
          The O <Latex math="\to" /> Q matrix
        </ExplorerSubsectionHeader>
        <button
          type="button"
          data-action="open-modal"
          onClick={() => setModalOpen(true)}
          className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-600 transition-colors"
          style={{ cursor: 'pointer' }}
        >
          expand ↗
        </button>
      </div>

      {/* Deck */}
      <p className="explorer-support-prose mt-2">
        Each row is one product orbit <Latex math="O" /> · each column is one stored output representative <Latex math="Q" /> · a filled cell means orbit <Latex math="O" />'s members project onto <Latex math="Q" /> via <Latex math="\pi_V" />. Counting filled cells gives <Latex math="\alpha" />.
      </p>

      {/* 2-column body: matrix on left, worked-example panel on right */}
      <div className="mt-4 grid gap-6 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <OrbitRepMatrix
          orbitRows={liveOrbitRows}
          selectedOrbitIdx={selectedOrbitIdx}
          onSelectOrbit={onSelectOrbit}
          onHover={null}
          expressionInfo={expressionInfo}
          componentInfo={liveComponentInfo}
          onStateChange={handleStateChange}
        />
        <WorkedExamplePanel
          hover={hover}
          pin={pin}
          orbitRows={liveOrbitRows}
          reps={reps}
          cells={cells}
          expressionInfo={expressionInfo}
          componentInfo={liveComponentInfo}
          onClearPin={handleClearPin}
        />
      </div>

      {/* Live α footer */}
      <div
        className="mt-3 font-mono text-[11px] text-gray-600"
        data-testid="branching-alpha-total"
      >
        across all {liveOrbitRows.length} orbits: α = <strong className="text-gray-900">{liveAlpha}</strong>
      </div>

      {/* Modal */}
      <OrbitRepMatrixModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        orbitRows={liveOrbitRows}
        reps={reps}
        cells={cells}
        hover={hover}
        pin={pin}
        onStateChange={handleStateChange}
        onClearPin={handleClearPin}
        expressionInfo={expressionInfo}
        componentInfo={liveComponentInfo}
        selectedOrbitIdx={selectedOrbitIdx}
        onSelectOrbit={onSelectOrbit}
      />
    </div>
  );
}
