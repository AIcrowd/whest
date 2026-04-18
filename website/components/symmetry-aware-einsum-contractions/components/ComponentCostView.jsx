import { useState } from 'react';
import CaseBadge from './CaseBadge.jsx';
import Latex from './Latex.jsx';
import OrbitInspector from './OrbitInspector.jsx';
import RoleBadge from './RoleBadge.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';
import { AnchorLink } from './ExplorerSectionCard.jsx';
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

// Per-component orbit count M_a, sourced from the engine field that
// `decomposeClassifyAndCount` now populates. The value is what the Act 5
// hero formula multiplies into ∏_a M_a.
function multiplicationCount(comp) {
  return comp.multiplication?.count ?? null;
}

function accumulationCount(comp) {
  return comp.accumulation?.count ?? null;
}

function accumulationFormula(comp) {
  return comp.accumulation?.latex ?? null;
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
        <span>Orbits (Mₐ)</span>
        <span>Accumulation (αₐ)</span>
        <span>Savings vs dense</span>
      </div>

      {components.map((comp, idx) => {
        const M_a = multiplicationCount(comp);
        const canOpenOrbits = supportsOrbitEnumeration(comp) && (orbitRows?.length ?? 0) > 0;
        // Per-component dense baseline = n^|L_a| (one tuple per cell, no
        // (k-1) factor — that lives globally on ∏_a M_a, not per-component).
        // Mul  savings = 1 - M_a   / n^|L_a|
        // Acc  savings = 1 - α_a   / n^|L_a|
        // Total savings = 1 - (M_a + α_a) / (2 · n^|L_a|)
        //   This is the per-component combined reduction at unit (k=2) cost.
        //   The honest global Total% lives in TotalCostView; this pill is a
        //   quick "is this component pulling its weight?" indicator.
        const labelCount = comp.labels?.length ?? 0;
        const denseCell = dimensionN ** labelCount;
        const actualAcc = accumulationCount(comp);
        const pct = (actual, dense) =>
          dense > 0 ? Math.max(0, Math.round((1 - actual / dense) * 100)) : null;
        const multSavingsPct = M_a !== null ? pct(M_a, denseCell) : null;
        const accSavingsPct = actualAcc !== null ? pct(actualAcc, denseCell) : null;
        const totalSavingsPct = (M_a !== null && actualAcc !== null)
          ? pct(M_a + actualAcc, 2 * denseCell)
          : null;

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
                    className="inline-flex cursor-pointer items-center gap-1 text-[11px] font-medium text-primary underline decoration-primary/40 decoration-dotted underline-offset-[3px] transition-colors hover:decoration-primary"
                    onClick={() => onOpenOrbitModal?.(comp)}
                  >
                    Enumerate orbits →
                  </button>
                ) : null}
              </div>

              {/* Orbits Mₐ (with greyed-out dense reference) */}
              <div className="flex items-baseline gap-1">
                {M_a !== null ? (
                  <>
                    <code className="font-mono text-sm font-semibold text-foreground">
                      {M_a.toLocaleString()}
                    </code>
                    <span
                      className="font-mono text-[11px] text-muted-foreground/60"
                      title={`Dense orbit count = n^${labelCount} = ${denseCell.toLocaleString()} (one per assignment when no symmetry collapses any pair).`}
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
                  <span
                    className="cursor-help rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800"
                    title={(() => {
                      // Prefer the most specific refusal — the brute-force
                      // entry carries the actual `Π nₗ · |G|` estimate, which
                      // is the actionable signal. The 'fallthrough' sentinel
                      // is just the loop's exit marker; skip it.
                      const trace = comp.accumulation?.trace ?? [];
                      const declined = [...trace]
                        .reverse()
                        .find((t) => t.decision === 'refused' && t.regimeId !== 'fallthrough');
                      const reason = declined?.reason ?? 'no regime fired';
                      return `αₐ withheld: ${reason}.`;
                    })()}
                  >
                    Unavailable
                  </span>
                )}
              </div>

              {/* Savings — three pills: Total (the headline), then Mult and
                  Acc breakdowns. The honest global Total% lives in
                  TotalCostView; the per-row Total here is a quick visual
                  signal (green when this component saves anything, red when
                  it doesn't pull any weight at all). */}
              <div>
                {totalSavingsPct !== null ? (
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                        Total
                      </span>
                      <span
                        className={`rounded-full px-2 py-0.5 font-mono text-xs font-semibold ${
                          totalSavingsPct > 0
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-rose-100 text-rose-800'
                        }`}
                        title={`Per-component combined savings: 1 − (Mₐ + αₐ) / (2 · n^${labelCount}) = 1 − (${(M_a ?? 0).toLocaleString()} + ${(actualAcc ?? 0).toLocaleString()}) / ${(2 * denseCell).toLocaleString()}.`}
                      >
                        {totalSavingsPct}%
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-x-1.5 font-mono text-[10px] leading-tight">
                      {multSavingsPct !== null ? (
                        <span
                          className="font-semibold text-primary"
                          title={`Mult savings: dense Mₐ would be ${denseCell.toLocaleString()}; symmetry gives ${M_a?.toLocaleString?.()}.`}
                        >
                          Mult {multSavingsPct}%
                        </span>
                      ) : null}
                      {multSavingsPct !== null && accSavingsPct !== null ? (
                        <span className="text-stone-300" aria-hidden="true">·</span>
                      ) : null}
                      {accSavingsPct !== null ? (
                        <span
                          className="font-semibold text-amber-700"
                          title={`Acc savings: dense αₐ would be ${denseCell.toLocaleString()}; symmetry gives ${actualAcc?.toLocaleString?.()}.`}
                        >
                          Acc {accSavingsPct}%
                        </span>
                      ) : null}
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
  const orbitRows = costModel.orbitRows ?? [];

  return (
    <div className="min-w-0 space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <div id="interaction-graph" className="rounded-xl border border-gray-200 bg-white p-4 scroll-mt-24">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                <AnchorLink anchorId="interaction-graph" labelText="Interaction Graph">
                  Interaction Graph
                </AnchorLink>
              </div>
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
              multiplicationCount: multiplicationCount(comp),
            }))}
          />
          <AccumulationHardCard />
        </div>
      </div>

      <div id="classification-tree" className="rounded-xl border border-gray-200 bg-white p-4 scroll-mt-24">
        <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">
          <AnchorLink anchorId="classification-tree" labelText="Classification Tree">
            Classification Tree
          </AnchorLink>
        </div>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          Each component is routed through a yes/no spine that dispatches to the
          cheapest applicable closed form, or to brute-force orbit projection
          when nothing else fits. The highlighted leaf on the left is where the
          current example lands.
        </p>
        <div className="mt-2 rounded-md border border-amber-200/60 bg-amber-50/50 px-3 py-2 text-[12.5px] leading-6 text-amber-900">
          <span className="font-semibold">Budget.</span> Brute-force orbit is the
          terminal leaf — it walks every <code className="font-mono">(tuple, g)</code> pair
          in <code className="font-mono">X × G</code>, so its cost is exactly
          {' '}<code className="font-mono">|X| · |G|</code>
          {' '}<em>pair-touches</em>, each ≈ one hash-map op. Both factors come
          from <em>einsum structure</em>: <code className="font-mono">|X|</code>
          {' '}is driven by the number of labels{' '}
          <code className="font-mono">|L|</code>, and{' '}
          <code className="font-mono">|G|</code> is the detected group order. We
          cap the count at <strong>1,500,000 pair-touches</strong> — a
          calibration, not a constant: roughly what a JS main thread handles in
          a few hundred ms without visibly hitching the UI. Above the cap, the
          regime declines and the αₐ cell reads{' '}
          <span className="rounded-full bg-amber-100 px-1.5 py-0.5 font-mono text-[11px] text-amber-800">Unavailable</span>.
          The cap is a demo-latency contract, not a statement about the einsum's
          structural cost. Hover the cell or the leaf to see the live estimate.
        </div>
        <div className="mt-4">
          <DecisionLadder
            activeLeafIds={components
              .flatMap((c) => [c.accumulation?.regimeId, c.shape])
              .filter(Boolean)}
            spotlightLeafIds={spotlightLeafIds}
            liveReasonsByLeaf={(() => {
              // For each leaf actually visited by some component (whether
              // it fired or refused), surface the verdict.reason string —
              // the brute-force regime in particular formats it with the
              // concrete (Π nₗ · |G|) estimate, which is what readers want
              // to see when they hit Unavailable.
              const map = new Map();
              for (const comp of components) {
                const trace = comp.accumulation?.trace ?? [];
                for (const step of trace) {
                  if (!step?.regimeId || !step?.reason) continue;
                  const list = map.get(step.regimeId) ?? [];
                  if (!list.includes(step.reason)) list.push(step.reason);
                  map.set(step.regimeId, list);
                }
              }
              return map;
            })()}
          />
        </div>
      </div>

      <ComponentSummaryTable
        components={components}
        dimensionN={dimensionN}
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
