import { useState } from 'react';
import { burnsideCount } from '../engine/permutation.js';
import CaseBadge from './CaseBadge.jsx';
import Latex from './Latex.jsx';
import OrbitInspector from './OrbitInspector.jsx';
import RoleBadge from './RoleBadge.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { LabelInteractionGraph } from './ComponentView.jsx';
import DecisionLadder from './DecisionLadder.jsx';
import PanZoomCanvas from './PanZoomCanvas.jsx';
import { getCasePresentation, getRegimePresentation } from './regimePresentation.js';

function isTrivial(comp) {
  return comp.caseType === 'trivial';
}

// Direct product count: ρ = ∏ n_ℓ over all labels in the component.
// Dedicated trivial path — no Burnside call, no orbit enumeration.
function directCount(comp, dimensionN) {
  const k = comp.labels?.length ?? 0;
  return Math.pow(dimensionN, k);
}

function computeMultiplicationOrbits(comp, dimensionN) {
  if (isTrivial(comp)) return directCount(comp, dimensionN);
  try {
    const sizes = (comp.labels ?? []).map(() => dimensionN);
    if (!comp.elements?.length) return 0;
    return burnsideCount(comp.elements, sizes).uniqueCount;
  } catch {
    return 0;
  }
}

function computeAccumulationCost(comp, dimensionN, fallbackReductionCost) {
  if (isTrivial(comp)) return directCount(comp, dimensionN);
  switch (comp.caseType) {
    case 'A':
      return Math.pow(dimensionN, comp.va?.length ?? 0);
    case 'B':
      return computeMultiplicationOrbits(comp, dimensionN);
    case 'D':
      try {
        const sizes = (comp.labels ?? []).map(() => dimensionN);
        if (!comp.haElements?.length) return 0;
        return burnsideCount(comp.haElements, sizes).uniqueCount;
      } catch {
        return 0;
      }
    default:
      return fallbackReductionCost;
  }
}

function accumulationCount(comp) {
  return comp.accumulation?.count ?? null;
}

function accumulationFormula(comp) {
  return comp.accumulation?.latex ?? null;
}

function multiplicationOrbits(comp) {
  return comp.accumulation?.count != null
    ? (comp.multiplication?.count ?? null)
    : (comp.multiplication?.count ?? null);
}

function methodLabel(comp) {
  return getCasePresentation(comp.caseType)?.methodLabel ?? 'Orbit enumeration';
}

function methodHumanName(comp) {
  return getCasePresentation(comp.caseType)?.humanName ?? null;
}

function supportsOrbitEnumeration(comp) {
  return !isTrivial(comp) && (comp.caseType === 'C' || comp.caseType === 'E');
}

function methodFormula(comp) {
  return getCasePresentation(comp.caseType)?.tooltip?.latex ?? null;
}

function LabelsCell({ comp }) {
  const orderedLabels = comp.labels ?? [];

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {orderedLabels.map((label) => {
        const role = (comp.va ?? []).includes(label) ? 'v' : 'w';
        return (
          <RoleBadge key={`${comp.caseType}-${label}`} role={role}>
            {label}
          </RoleBadge>
        );
      })}
    </div>
  );
}

function ComponentSummaryTable({
  components,
  dimensionN,
  fallbackReductionCost,
  orbitRows,
  onOpenOrbitModal,
}) {
  return (
    <div className="max-w-full overflow-x-auto rounded-xl border border-border bg-white shadow-sm">
      <Table className="text-sm">
        <TableHeader className="bg-surface-raised">
          <TableRow className="border-border hover:bg-surface-raised">
            <TableHead className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Case</TableHead>
            <TableHead className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Labels</TableHead>
            <TableHead className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Symmetry</TableHead>
            <TableHead className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Method</TableHead>
            <TableHead className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Multiplication Cost</TableHead>
            <TableHead className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Accumulation Cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody className="divide-y divide-border">
          {components.map((comp, idx) => {
            const multiplicationOrbits = computeMultiplicationOrbits(comp, dimensionN);
            const accumulationCost = computeAccumulationCost(comp, dimensionN, fallbackReductionCost);
            const canOpenOrbits = supportsOrbitEnumeration(comp) && (orbitRows?.length ?? 0) > 0;

            return (
              <TableRow key={`comp-row-${idx}`} className="border-0 bg-surface hover:bg-surface-raised">
                <TableCell className="px-3 py-2">
                  <CaseBadge caseType={comp.caseType} size="sm" />
                </TableCell>
                <TableCell className="px-3 py-2">
                  <LabelsCell comp={comp} />
                </TableCell>
                <TableCell className="px-3 py-2">
                  <SymmetryBadge value={comp.groupName || 'trivial'} />
                </TableCell>
                <TableCell className="px-3 py-2">
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      {canOpenOrbits ? (
                        <button
                          type="button"
                          className="rounded-full border border-coral bg-white px-2.5 py-1 text-xs font-semibold text-coral transition-colors hover:bg-coral-light"
                          onClick={() => onOpenOrbitModal?.(comp)}
                        >
                          Orbit Enumeration
                        </button>
                      ) : (
                        <span className="text-xs font-medium text-foreground">{methodLabel(comp)}</span>
                      )}
                    </div>
                    {methodHumanName(comp) ? (
                      <div className="text-[11px] italic leading-snug text-muted-foreground">
                        {methodHumanName(comp)}
                      </div>
                    ) : null}
                    {methodFormula(comp) ? (
                      <div className="text-xs text-muted-foreground">
                        <Latex math={methodFormula(comp)} />
                      </div>
                    ) : null}
                  </div>
                </TableCell>
                <TableCell className="px-3 py-2">
                  <code className="font-mono text-xs text-foreground">{multiplicationOrbits.toLocaleString()}</code>
                </TableCell>
                <TableCell className="px-3 py-2">
                  {accumulationCount(comp) !== null
                    ? <code className="font-mono text-xs text-foreground">{accumulationCount(comp).toLocaleString()}</code>
                    : <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800">Unavailable</span>}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export default function ComponentCostView({
  componentData,
  costModel,
  dimensionN,
  allLabels,
  vLabels,
  selectedOrbitIdx,
  onSelectOrbit,
}) {
  if (!componentData || !costModel) return null;

  const [showOrbitModal, setShowOrbitModal] = useState(false);
  const [orbitModalComponent, setOrbitModalComponent] = useState(null);
  const components = componentData.components ?? [];
  const fallbackReductionCost = costModel.reductionCost ?? 0;
  const orbitRows = costModel.orbitRows ?? [];

  return (
    <div className="min-w-0 space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Interaction Graph</div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            These independent components come from the label interaction graph induced by the detected group generators.
          </p>
          <PanZoomCanvas
            className="mt-4 h-[620px]"
            ariaLabel="Interaction graph (zoomable)"
          >
            <LabelInteractionGraph
              allLabels={allLabels}
              vLabels={vLabels}
              interactionGraph={componentData.interactionGraph}
            />
          </PanZoomCanvas>
        </div>

        <DecisionLadder activeRegimeId={components[0]?.accumulation?.regimeId ?? null} />
      </div>

      <ComponentSummaryTable
        components={components}
        dimensionN={dimensionN}
        fallbackReductionCost={fallbackReductionCost}
        orbitRows={orbitRows}
        onOpenOrbitModal={(comp) => {
          setOrbitModalComponent(comp);
          setShowOrbitModal(true);
        }}
      />

      {showOrbitModal && (
        <div
          className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={() => {
            setShowOrbitModal(false);
            setOrbitModalComponent(null);
          }}
        >
          <div
            className="max-h-[85vh] w-[min(960px,92vw)] overflow-y-auto rounded-xl bg-white shadow-2xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="orbit-inspector-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-4 border-b border-border/70 px-5 py-4">
              <h2 id="orbit-inspector-modal-title" className="text-sm font-medium text-foreground">
                Orbit Enumeration
              </h2>
              <button
                type="button"
                className="rounded-md px-2 py-1 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                aria-label="Close orbit inspector"
                onClick={() => {
                  setShowOrbitModal(false);
                  setOrbitModalComponent(null);
                }}
              >
                Close
              </button>
            </div>
            <div className="px-5 pb-5 pt-5">
              <OrbitInspector
                orbitRows={orbitRows}
                selectedOrbitIdx={selectedOrbitIdx}
                onSelectOrbit={onSelectOrbit}
                showHeader={false}
                formulaMath={ORBIT_ENUMERATION_FORMULA}
                dimensionN={dimensionN}
                componentContext={orbitModalComponent}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
