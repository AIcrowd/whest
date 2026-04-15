import { useMemo } from 'react';
import { burnsideCount } from '../engine/permutation.js';
import { CASE_META } from '../engine/componentDecomposition.js';
import CaseBadge from './CaseBadge.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import OrbitInspector from './OrbitInspector.jsx';
import { DecisionTree, LabelInteractionGraph } from './ComponentView.jsx';

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

function MetricCard({ label, value, caption }) {
  return (
    <div className="rounded-lg bg-gray-50 p-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-500">{label}</div>
      <div className="mt-2 text-2xl font-mono font-bold text-gray-900">{value.toLocaleString()}</div>
      <div className="mt-1 text-xs text-gray-500">{caption}</div>
    </div>
  );
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
        <CaseBadge caseType={comp.caseType} interactive={false} />
        <code className="text-sm text-gray-700">{`{${(comp.labels ?? []).join(', ')}}`}</code>
        <span className="text-xs text-gray-400">{comp.groupName || 'trivial'}</span>
        {comp.order > 1 && <span className="text-xs font-mono text-gray-400">|G|={comp.order}</span>}
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <MetricCard
          label="Mult Orbits"
          value={multiplicationOrbits}
          caption="Burnside on the component group"
        />
        <MetricCard
          label="Accum ρ"
          value={accumulationCost}
          caption={comp.caseType === 'D' ? 'Burnside on Hₐ' : comp.caseType === 'A' ? 'No accumulation savings' : 'Orbit projection'}
        />
        <div className="rounded-lg bg-gray-50 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-500">Method</div>
          <div className="mt-2 text-sm text-gray-700">{methodLabel(comp)}</div>
          <div className="mt-2 text-xs text-gray-500">
            V: {(comp.va ?? []).join(', ') || '—'} | W: {(comp.wa ?? []).join(', ') || '—'}
          </div>
        </div>
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
      <NarrativeCallout label="Component Story">
        The detected group splits into independent components. Each component contributes its own multiplication representatives and accumulation rule, so the savings story can be read locally before the totals are assembled.
      </NarrativeCallout>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500">Interaction Graph</div>
          <p className="mt-1 text-sm text-gray-700">
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
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500">Orbit Enumeration</div>
          <p className="mt-1 text-sm text-gray-700">
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
