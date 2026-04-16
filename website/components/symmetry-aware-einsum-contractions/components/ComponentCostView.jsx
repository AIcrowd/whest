import { useMemo } from 'react';
import { burnsideCount } from '../engine/permutation.js';
import { CASE_META } from '../engine/componentDecomposition.js';
import CaseBadge from './CaseBadge.jsx';
import ExplorerMetricCard from './ExplorerMetricCard.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import OrbitInspector from './OrbitInspector.jsx';
import { DecisionTree, LabelInteractionGraph } from './ComponentView.jsx';

export const COMPONENT_STORY_TEXT = 'The detected group splits into independent components. Each component contributes its own multiplication representatives and accumulation rule, so the savings story can be read locally before the totals are assembled.';

function computeMultiplicationOrbits(comp, dimensionN) {
  try {
    const sizes = (comp.labels ?? []).map(() => dimensionN);
    if (!comp.elements?.length) return 0;
    return burnsideCount(comp.elements, sizes).uniqueCount;
  } catch {
    return 0;
  }
}

function computeAccumulationCost(comp, dimensionN, fallbackReductionCost) {
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

function methodLabel(comp) {
  return CASE_META[comp.caseType]?.method ?? 'orbit enumeration';
}

function ComponentCard({ comp, dimensionN, fallbackReductionCost }) {
  const multiplicationOrbits = computeMultiplicationOrbits(comp, dimensionN);
  const accumulationCost = computeAccumulationCost(comp, dimensionN, fallbackReductionCost);

  return (
    <div
      className="space-y-4 rounded-xl border border-gray-200 bg-white p-4"
      style={{ borderLeftWidth: 4, borderLeftColor: CASE_META[comp.caseType]?.color ?? '#D1D5DB' }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <CaseBadge caseType={comp.caseType} />
        <code className="text-sm text-foreground">{`{${(comp.labels ?? []).join(', ')}}`}</code>
        <span className="text-sm text-muted-foreground">{comp.groupName || 'trivial'}</span>
        {comp.order > 1 && <span className="text-sm font-mono text-muted-foreground">|G|={comp.order}</span>}
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <ExplorerMetricCard
          label="Mult Orbits"
          value={multiplicationOrbits.toLocaleString()}
          detail="Burnside on the component group"
          className="border-0 bg-gray-50 shadow-none"
        />
        <ExplorerMetricCard
          label="Accum ρ"
          value={accumulationCost.toLocaleString()}
          detail={comp.caseType === 'D' ? 'Burnside on Hₐ' : comp.caseType === 'A' ? 'No accumulation savings' : 'Orbit projection'}
          className="border-0 bg-gray-50 shadow-none"
        />
        <ExplorerMetricCard
          label="Method"
          value={methodLabel(comp)}
          detail={
            <>
              V: {(comp.va ?? []).join(', ') || '—'} | W: {(comp.wa ?? []).join(', ') || '—'}
            </>
          }
          className="border-0 bg-gray-50 shadow-none"
          valueClassName="font-sans text-sm font-semibold text-gray-700"
          detailClassName="text-muted-foreground"
        />
      </div>
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
  showComponentStory = true,
}) {
  if (!componentData || !costModel) return null;

  const components = componentData.components ?? [];
  const fallbackReductionCost = costModel.reductionCost ?? 0;
  const needsOrbitInspector = useMemo(
    () => components.some((comp) => comp.caseType === 'C' || comp.caseType === 'E'),
    [components],
  );

  return (
    <div className="space-y-6">
      {showComponentStory ? (
        <NarrativeCallout label="Component Story">
          {COMPONENT_STORY_TEXT}
        </NarrativeCallout>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Interaction Graph</div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            These independent components come from the label interaction graph induced by the detected group generators.
          </p>
          <div className="mt-4 flex justify-center">
            <LabelInteractionGraph
              allLabels={allLabels}
              vLabels={vLabels}
              interactionGraph={componentData.interactionGraph}
            />
          </div>
        </div>

        <DecisionTree components={components} />
      </div>

      <div className="grid gap-4">
        {components.map((comp, idx) => (
          <ComponentCard
            key={`${comp.caseType}-${idx}`}
            comp={comp}
            dimensionN={dimensionN}
            fallbackReductionCost={fallbackReductionCost}
          />
        ))}
      </div>

      {needsOrbitInspector && (
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Orbit Enumeration</div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Cases C and E need orbit enumeration because no closed-form stabilizer shortcut applies.
          </p>
          <div className="mt-4">
            <OrbitInspector
              orbitRows={costModel.orbitRows ?? []}
              selectedOrbitIdx={selectedOrbitIdx}
              onSelectOrbit={onSelectOrbit}
              kicker="Orbit Enumeration"
              title="Output projections for mixed components"
              description="Inspect how one multiplication representative can still scatter into multiple output bins."
            />
          </div>
        </div>
      )}
    </div>
  );
}
