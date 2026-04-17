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
import ExplorerModal from './ExplorerModal.jsx';
import MultiplicationCostCard from './MultiplicationCostCard.jsx';
import AccumulationHardCard from './AccumulationHardCard.jsx';
import { getRegimePresentation } from './regimePresentation.js';

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

function supportsOrbitEnumeration(comp) {
  return !isTrivial(comp) && (comp.caseType === 'C' || comp.caseType === 'E');
}

/**
 * Method description: split "Technique — reason" on the em-dash and render
 * the technique bold + the reason in muted body. Plain-text throughout
 * (colour-coding of individual math symbols is a separate follow-up).
 */
function MethodDescription({ text }) {
  if (typeof text !== 'string' || !text.includes('—')) {
    return <p className="text-[12.5px] leading-snug text-foreground">{text}</p>;
  }
  const [head, ...rest] = text.split('—');
  const technique = head.trim();
  const reason = rest.join('—').trim();
  return (
    <p className="text-[12.5px] leading-snug text-foreground">
      <span className="font-semibold">{technique}</span>
      <span className="text-muted-foreground"> — </span>
      <span className="text-stone-700">{reason}</span>
    </p>
  );
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
  // Shared column template so every component card's middle-row lines up
  // column-wise with the global header at the top.
  const MIDDLE_COLS = 'grid-cols-[1.2fr_2.5fr_0.9fr_0.9fr_1.4fr]';

  return (
    <div className="max-w-full overflow-x-auto rounded-xl border border-border bg-white shadow-sm">
      {/* Global column header — only labels the 5 middle-row columns. */}
      <div
        className={`grid ${MIDDLE_COLS} items-center gap-x-4 bg-surface-raised px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground`}
      >
        <span>Labels</span>
        <span>Method</span>
        <span>MUL Cost</span>
        <span>Acc Cost</span>
        <span>Savings</span>
      </div>

      {components.map((comp, idx) => {
        const multiplicationOrbits = computeMultiplicationOrbits(comp, dimensionN);
        const accumulationCost = computeAccumulationCost(comp, dimensionN, fallbackReductionCost);
        const canOpenOrbits = supportsOrbitEnumeration(comp) && (orbitRows?.length ?? 0) > 0;
        // Dense baseline for this component: every tuple does one product
        // and one write. Actual cost: mult orbits + distinct output bins.
        // Savings % = 1 - actual / dense, floored at 0.
        const labelCount = comp.labels?.length ?? 0;
        const denseCell = dimensionN ** labelCount;
        const denseWork = 2 * denseCell;
        const actualAcc = accumulationCount(comp);
        const actualWork = multiplicationOrbits + (actualAcc ?? 0);
        const pct = (actual, dense) =>
          dense > 0 ? Math.max(0, Math.round((1 - actual / dense) * 100)) : null;
        const multSavingsPct = pct(multiplicationOrbits, denseCell);
        const accSavingsPct = actualAcc !== null ? pct(actualAcc, denseCell) : null;
        const totalSavingsPct =
          actualAcc !== null ? pct(actualWork, denseWork) : null;

        const leafId = comp.accumulation?.regimeId ?? comp.shape ?? comp.caseType;
        const presentation = getRegimePresentation(leafId);
        const methodDescription = presentation?.tooltip?.body;
        const methodLatex = presentation?.tooltip?.latex;

        return (
          <div
            key={`comp-${idx}`}
            className="border-t-2 border-border/70 px-5"
          >
            {/* Band 1 — Case (full-width header band) */}
            <div className="flex items-center gap-2 py-3">
              <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                Case
              </span>
              <CaseBadge regimeId={leafId} caseType={comp.caseType} size="sm" />
            </div>

            <div className="border-t border-border/40" aria-hidden="true" />

            {/* Band 2 — the 5-column middle row */}
            <div className={`grid ${MIDDLE_COLS} items-start gap-x-4 py-3`}>
              {/* Labels */}
              <div>
                <LabelsCell comp={comp} />
              </div>

              {/* Method: description + α formula, wrapped in a CaseBadge
                  passthrough so hovering any part opens the full tooltip
                  (glossary and all). */}
              <div className="space-y-2">
                <CaseBadge regimeId={leafId} caseType={comp.caseType}>
                  <div className="space-y-2">
                    {methodDescription ? (
                      <MethodDescription text={methodDescription} />
                    ) : null}
                    {methodLatex ? (
                      <div className="overflow-x-auto pl-2 text-[13px] text-foreground">
                        <Latex math={methodLatex} />
                      </div>
                    ) : null}
                  </div>
                </CaseBadge>
                {canOpenOrbits ? (
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-[11px] font-medium text-primary underline decoration-primary/40 decoration-dotted underline-offset-[3px] transition-colors hover:decoration-primary"
                    onClick={() => onOpenOrbitModal?.(comp)}
                  >
                    Enumerate orbits →
                  </button>
                ) : null}
              </div>

              {/* MUL Cost (with greyed-out dense reference) */}
              <div className="flex items-baseline gap-1">
                <code className="font-mono text-sm font-semibold text-foreground">
                  {multiplicationOrbits.toLocaleString()}
                </code>
                <span
                  className="font-mono text-[11px] text-muted-foreground/60"
                  title={`Dense baseline: every tuple does one product (${denseCell.toLocaleString()} = n^${labelCount})`}
                >
                  / {denseCell.toLocaleString()}
                </span>
              </div>

              {/* Acc Cost (with greyed-out dense reference) */}
              <div className="flex items-baseline gap-1">
                {actualAcc !== null ? (
                  <>
                    <code className="font-mono text-sm font-semibold text-foreground">
                      {actualAcc.toLocaleString()}
                    </code>
                    <span
                      className="font-mono text-[11px] text-muted-foreground/60"
                      title={`Dense baseline: one write per tuple (${denseCell.toLocaleString()} = n^${labelCount})`}
                    >
                      / {denseCell.toLocaleString()}
                    </span>
                  </>
                ) : (
                  <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800">
                    Unavailable
                  </span>
                )}
              </div>

              {/* Savings */}
              <div>
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
              </div>
            </div>

            <div className="border-t border-border/40" aria-hidden="true" />

            {/* Band 3 — Symmetry (full-width footer band) */}
            <div className="flex items-center gap-2 py-3">
              <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                Symmetry
              </span>
              <SymmetryBadge value={comp.groupName || 'trivial'} />
            </div>
          </div>
        );
      })}
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
  numTerms = 2,
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

        {/* justify-center spreads any slack vertical space (this column is
            shorter than the interaction-graph column on the left) equally
            between the top and bottom, so the two cost cards sit centred
            rather than huddled at the top with dead space beneath. */}
        <div className="flex flex-col justify-center gap-6">
          <MultiplicationCostCard
            components={components.map((comp) => ({
              ...comp,
              multiplicationCount: computeMultiplicationOrbits(comp, dimensionN),
            }))}
            numTerms={numTerms}
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
