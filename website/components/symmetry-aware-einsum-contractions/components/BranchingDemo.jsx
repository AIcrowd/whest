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
  // Live hover for the WorkedExamplePanel. The matrix owns hover in a ref
  // for instant canvas paint; it surfaces the latest cell here ~80 ms after
  // the mouse settles via `onHoverDeferred`. So sweeps don't trigger panel
  // re-renders, but pauses do — visually live without the 60 Hz render storm.
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
  // doesn't re-fire on every parent render.
  //
  // Critical perf note: we DO NOT propagate hover to the App-level `onHover`
  // (cross-spotlight). That handler used to fire `setGraphHover` on every
  // cell-to-cell hover, which forced the entire App to re-render (cost cards,
  // classification tree, summary table, partition counter, every section).
  // For matrices with many cells, mousemove was 500–1000 ms per cell. The
  // cross-spotlight payload is the orbit's label set, which is identical for
  // every orbit in a given preset — so per-hover propagation was pure waste.
  // Pin still propagates `onSelectOrbit` (rare; click-only).
  const handleStateChange = useCallback(({ pin: p }) => {
    setPin(p);
    if (p && onSelectOrbit) onSelectOrbit(p.row);
  }, [onSelectOrbit]);

  const handleDeferredHover = useCallback((cell) => {
    setHover((prev) => {
      if (prev === cell) return prev;
      if (prev && cell && prev.row === cell.row && prev.col === cell.col) return prev;
      return cell;
    });
  }, []);

  const handleClearPin = useCallback(() => {
    setPin(null);
  }, []);

  if (!componentData || !costModel) return null;

  const handleOpenModal = useCallback(() => setModalOpen(true), []);

  return (
    <div id="orbit-rep-matrix" className="bg-white p-4 scroll-mt-24">
      {/* Header row: subsection title only — expand button now lives next
          to the canvas where it's discoverable. */}
      <ExplorerSubsectionHeader anchorId="orbit-rep-matrix" labelText="Branching">
        The O <Latex math="\to" /> Q matrix
      </ExplorerSubsectionHeader>

      {/* Deck */}
      <p className="explorer-support-prose mt-2">
        Each row is one product orbit <Latex math="O" /> · each column is one stored output representative <Latex math="Q" /> · a filled cell means orbit <Latex math="O" />'s members project onto <Latex math="Q" /> via <Latex math="\pi_V" />. Counting filled cells gives <Latex math="\alpha" />.
      </p>

      {/* 2-column body: matrix on the left at a FIXED width so its cell
          dimensions don't drift when the panel content changes. The panel
          takes whatever's left. `items-start` keeps each column at its
          natural height — without it, when the panel grows on hover the
          grid row stretches to match and leaves dead whitespace below
          the matrix. */}
      <div className="mt-4 grid gap-6 grid-cols-1 items-start lg:grid-cols-[400px_minmax(0,1fr)]">
        {/* Matrix column — wraps the matrix + alpha footer together so the
            footer sits directly beneath the canvas instead of below the
            full-grid (which would put it under the much-taller panel and
            leave dead space between matrix and footer). */}
        <div>
          <OrbitRepMatrix
            orbitRows={liveOrbitRows}
            selectedOrbitIdx={selectedOrbitIdx}
            onSelectOrbit={onSelectOrbit}
            onHover={null}
            expressionInfo={expressionInfo}
            componentInfo={liveComponentInfo}
            onStateChange={handleStateChange}
            onHoverDeferred={handleDeferredHover}
            onExpand={handleOpenModal}
          />
          <div
            className="mt-3 font-mono text-[11px] text-gray-600"
            data-testid="branching-alpha-total"
          >
            across all {liveOrbitRows.length} orbits: α = <strong className="text-gray-900">{liveAlpha}</strong>
          </div>
        </div>
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
        onHoverDeferred={handleDeferredHover}
      />
    </div>
  );
}
