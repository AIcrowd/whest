import CaseBadge from './CaseBadge.jsx';
import ExplorerMetricCard from './ExplorerMetricCard.jsx';

function ComponentRecap({ components }) {
  if (!components?.length) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Component recap</span>
      {components.map((comp, idx) => (
        <span
          key={`component-recap-${idx}`}
          className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-raised px-2.5 py-1 text-xs text-muted-foreground"
        >
          <CaseBadge caseType={comp.caseType} size="xs" />
          <span className="font-mono">{`{${(comp.labels ?? []).join(', ')}}`}</span>
        </span>
      ))}
    </div>
  );
}

export default function TotalCostView({ costModel, componentData, dimensionN, numTerms = 1 }) {
  if (!costModel || !componentData) return null;

  const { orbitCount = 0, evaluationCost = 0, reductionCost = 0 } = costModel;
  const totalCost = evaluationCost + reductionCost;
  const allLabelCount = componentData?.components?.reduce((sum, comp) => sum + comp.labels.length, 0) ?? 0;
  const denseTuples = Math.pow(dimensionN, allLabelCount);
  const denseTotalCost = Math.max(numTerms - 1, 0) * denseTuples + denseTuples;
  const totalSpeedup = totalCost > 0 ? (denseTotalCost / totalCost).toFixed(1) : '1.0';
  const savings = denseTotalCost - totalCost;
  const savingsPct = denseTotalCost > 0 ? ((savings / denseTotalCost) * 100).toFixed(1) : '0';
  const { components = [] } = componentData;

  return (
    <div className="space-y-8">
      <ComponentRecap components={components} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ExplorerMetricCard
          label="Multiplication Cost"
          value={evaluationCost.toLocaleString()}
          detail={`${orbitCount.toLocaleString()} multiplication orbit${orbitCount !== 1 ? 's' : ''}`}
        />
        <ExplorerMetricCard
          label="Accumulation Cost"
          value={reductionCost.toLocaleString()}
          detail="distinct output-bin updates"
        />
        <ExplorerMetricCard
          label="Total Cost"
          value={totalCost.toLocaleString()}
          detail="multiplication + accumulation"
          className="border-coral/30 bg-coral-light"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ExplorerMetricCard
          label="Dense Cost"
          value={denseTotalCost.toLocaleString()}
          detail={
            <>
              {Math.max(numTerms - 1, 0)} × n<sup>{allLabelCount}</sup> + n<sup>{allLabelCount}</sup> with n={dimensionN}
            </>
          }
        />
        <ExplorerMetricCard
          label="Symmetry-Aware Cost"
          value={totalCost.toLocaleString()}
          detail="multiplication + accumulation under full orbit model"
        />
        <ExplorerMetricCard
          label="%age Savings"
          value={`${savingsPct}%`}
          detail={savings === 0 ? 'Cost: 0; Speedup: 1.0×' : `Cost: ${savings.toLocaleString()}; Speedup: ${totalSpeedup}×`}
          className="border-green-600/20 bg-green-600/5"
          valueClassName="text-green-700"
          detailClassName="text-gray-400"
        />
      </div>
    </div>
  );
}
