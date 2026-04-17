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
import ExplorerModal from './ExplorerModal.jsx';
import MultiplicationCostCard from './MultiplicationCostCard.jsx';
import AccumulationHardCard from './AccumulationHardCard.jsx';

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
      <Table className="w-full table-fixed text-sm">
        <colgroup>
          <col className="w-[10%]" />
          <col className="w-[10%]" />
          <col className="w-[12%]" />
          <col className="w-[30%]" />
          <col className="w-[11%]" />
          <col className="w-[11%]" />
          <col className="w-[16%]" />
        </colgroup>
        <TableHeader className="bg-surface-raised">
          <TableRow className="border-border hover:bg-surface-raised">
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Case</TableHead>
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Labels</TableHead>
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Symmetry</TableHead>
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Method</TableHead>
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">MUL Cost</TableHead>
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Acc Cost</TableHead>
            <TableHead className="whitespace-normal px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Savings</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody className="divide-y divide-border">
          {components.map((comp, idx) => {
            const multiplicationOrbits = computeMultiplicationOrbits(comp, dimensionN);
            const accumulationCost = computeAccumulationCost(comp, dimensionN, fallbackReductionCost);
            const canOpenOrbits = supportsOrbitEnumeration(comp) && (orbitRows?.length ?? 0) > 0;
            // Dense baseline for this component: every tuple does one product
            // and one write. Actual cost: mult orbits + distinct output bins.
            // Savings % = 1 - actual / dense, floored at 0.
            const labelCount = comp.labels?.length ?? 0;
            const denseCell = dimensionN ** labelCount; // |X| for this component
            const denseWork = 2 * denseCell;            // one product + one write per cell
            const actualAcc = accumulationCount(comp);
            const actualWork = multiplicationOrbits + (actualAcc ?? 0);
            const pct = (actual, dense) =>
              dense > 0 ? Math.max(0, Math.round((1 - actual / dense) * 100)) : null;
            const multSavingsPct = pct(multiplicationOrbits, denseCell);
            const accSavingsPct = actualAcc !== null ? pct(actualAcc, denseCell) : null;
            const totalSavingsPct =
              actualAcc !== null ? pct(actualWork, denseWork) : null;

            return (
              <TableRow key={`comp-row-${idx}`} className="border-0 bg-surface hover:bg-surface-raised">
                <TableCell className="px-3 py-2">
                  <CaseBadge regimeId={comp.accumulation?.regimeId ?? comp.shape ?? comp.caseType} caseType={comp.caseType} size="sm" />
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
                <TableCell className="px-3 py-2">
                  {totalSavingsPct !== null ? (
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                          Total
                        </span>
                        <span
                          className={`rounded-full px-2 py-0.5 font-mono text-xs font-semibold ${
                            totalSavingsPct >= 50
                              ? 'bg-emerald-100 text-emerald-800'
                              : totalSavingsPct > 0
                                ? 'bg-amber-50 text-amber-800'
                                : 'bg-stone-100 text-stone-600'
                          }`}
                          title={`dense ${denseWork.toLocaleString()} → actual ${actualWork.toLocaleString()}`}
                        >
                          {totalSavingsPct}%
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-x-1.5 font-mono text-[10px] leading-tight">
                        <span className="font-semibold text-primary">Mult {multSavingsPct}%</span>
                        <span className="text-stone-300" aria-hidden="true">·</span>
                        <span className="font-semibold text-amber-700">Acc {accSavingsPct}%</span>
                      </div>
                    </div>
                  ) : (
                    <span className="text-[11px] text-muted-foreground">—</span>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

// Compact stat read next to the card's title. Three numbers that answer
// "how big / connected is this graph" at a glance, in the same visual
// register as the rest of Act 4 (uppercase eyebrow + mono numeral).
function InteractionGraphMetricStrip({ labelCount, edgeCount, componentCount }) {
  const cells = [
    { label: 'labels', value: labelCount },
    { label: 'edges', value: edgeCount },
    { label: 'components', value: componentCount },
  ];
  return (
    <div className="flex shrink-0 items-stretch gap-1.5">
      {cells.map((cell) => (
        <div
          key={cell.label}
          className="flex flex-col items-center rounded-md border border-border/70 bg-surface-raised px-2 py-1"
        >
          <span className="font-mono text-sm font-semibold leading-tight text-foreground">
            {cell.value}
          </span>
          <span className="mt-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            {cell.label}
          </span>
        </div>
      ))}
    </div>
  );
}

// Inline legend for the graph's visual vocabulary. The V/W dot colors here
// must match LabelInteractionGraph's COLOR_V / COLOR_W so the legend stays
// truthful — see ComponentView.jsx.
function InteractionGraphLegend() {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2.5 rounded-full" style={{ backgroundColor: '#4A7CFF' }} />
        free label (<Latex math="V" />)
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2.5 rounded-full" style={{ backgroundColor: '#94A3B8' }} />
        summed label (<Latex math="W" />)
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-px w-5" style={{ backgroundColor: '#6B7280' }} />
        edge: co-permuted by some <Latex math="\sigma \in G" />
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span
          className="inline-block size-2.5 rounded-sm border"
          style={{ borderStyle: 'dashed', borderColor: '#94A3B8' }}
        />
        hull: one independent component (Case A–E)
      </span>
    </div>
  );
}

export default function ComponentCostView({
  componentData,
  costModel,
  dimensionN,
  allLabels,
  vLabels,
  fullGenerators,
  selectedOrbitIdx,
  onSelectOrbit,
  onGraphHover,
  spotlightLeafIds,
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
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Interaction Graph</div>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Nodes are <strong className="font-semibold text-foreground">labels</strong>; an edge marks labels that a generator of&nbsp;
                <Latex math="G" />&nbsp;moves together. Disjoint components factor the cost into independent sub-problems — each one lands on a case in the decision tree below.
              </p>
            </div>
            <InteractionGraphMetricStrip
              labelCount={allLabels.length}
              edgeCount={componentData.interactionGraph?.edges?.length ?? 0}
              componentCount={components.length}
            />
          </div>
          <PanZoomCanvas
            className="mt-4 h-[620px]"
            ariaLabel="Interaction graph (zoomable)"
          >
            <LabelInteractionGraph
              allLabels={allLabels}
              vLabels={vLabels}
              interactionGraph={componentData.interactionGraph}
              components={components}
              fullGenerators={fullGenerators}
              onHover={onGraphHover}
            />
          </PanZoomCanvas>
          <InteractionGraphLegend />
        </div>

        <div className="flex flex-col gap-6">
          <MultiplicationCostCard
            components={components.map((comp) => ({
              ...comp,
              multiplicationCount: computeMultiplicationOrbits(comp, dimensionN),
            }))}
          />
          <AccumulationHardCard />
        </div>
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">
          Classification Tree
        </div>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          Each component is routed through a yes/no spine that dispatches to the
          cheapest applicable closed form, or to brute-force orbit projection
          when nothing else fits. The highlighted leaf on the left is where the
          current example lands.
        </p>
        <div className="mt-4">
          <DecisionLadder
            activeLeafIds={components
              .flatMap((c) => [c.accumulation?.regimeId, c.shape])
              .filter(Boolean)}
            spotlightLeafIds={spotlightLeafIds}
          />
        </div>
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

      <ExplorerModal
        title="Orbit Enumeration"
        titleId="orbit-inspector-modal-title"
        open={showOrbitModal}
        onClose={() => {
          setShowOrbitModal(false);
          setOrbitModalComponent(null);
        }}
      >
        <OrbitInspector
          orbitRows={orbitRows}
          selectedOrbitIdx={selectedOrbitIdx}
          onSelectOrbit={onSelectOrbit}
          showHeader={false}
          formulaMath={null}
          dimensionN={dimensionN}
          componentContext={orbitModalComponent}
        />
      </ExplorerModal>
    </div>
  );
}
