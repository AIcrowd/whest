import CaseBadge from './CaseBadge.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';

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
    <div className="space-y-6">
      <NarrativeCallout label="Why this matters">
        These totals combine the representative multiplications and the remaining output-bin updates into the final symmetry-aware contraction cost.
      </NarrativeCallout>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-surface-raised">
              <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted">Case</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted">Labels</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted">V (free)</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted">W (summed)</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted">Group</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {components.map((comp, idx) => (
              <tr key={`comp-row-${idx}`} className="bg-surface transition-colors hover:bg-surface-raised">
                <td className="px-4 py-2.5">
                  <CaseBadge caseType={comp.caseType} size="sm" interactive={false} />
                </td>
                <td className="px-4 py-2.5">
                  <code className="font-mono text-xs text-foreground">{(comp.labels ?? []).join(', ')}</code>
                </td>
                <td className="px-4 py-2.5">
                  {(comp.va ?? []).length > 0
                    ? <code className="font-mono text-xs text-foreground">{comp.va.join(', ')}</code>
                    : <span className="text-muted">—</span>}
                </td>
                <td className="px-4 py-2.5">
                  {(comp.wa ?? []).length > 0
                    ? <code className="font-mono text-xs text-foreground">{comp.wa.join(', ')}</code>
                    : <span className="text-muted">—</span>}
                </td>
                <td className="px-4 py-2.5">
                  <code className="font-mono text-xs text-foreground">{comp.groupName ?? '—'}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Multiplication Cost</div>
          <div className="text-2xl font-mono font-bold text-foreground">{evaluationCost.toLocaleString()}</div>
          <div className="mt-1 text-xs text-muted">
            {orbitCount.toLocaleString()} multiplication orbit{orbitCount !== 1 ? 's' : ''}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Accumulation Cost</div>
          <div className="text-2xl font-mono font-bold text-foreground">{reductionCost.toLocaleString()}</div>
          <div className="mt-1 text-xs text-muted">distinct output-bin updates</div>
        </div>

        <div className="rounded-lg border border-coral/30 bg-coral-light p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Total Cost</div>
          <div className="text-2xl font-mono font-bold text-foreground">{totalCost.toLocaleString()}</div>
          <div className="mt-1 text-xs text-muted">multiplication + accumulation</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Dense Cost</div>
          <div className="text-2xl font-mono font-bold text-foreground">{denseTotalCost.toLocaleString()}</div>
          <div className="mt-1 text-xs text-muted">
            {Math.max(numTerms - 1, 0)} × n<sup>{allLabelCount}</sup> + n<sup>{allLabelCount}</sup> with n={dimensionN}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Symmetry-Aware Cost</div>
          <div className="text-2xl font-mono font-bold text-foreground">{totalCost.toLocaleString()}</div>
          <div className="mt-1 text-xs text-muted">multiplication + accumulation under full orbit model</div>
        </div>

        <div className="rounded-lg border border-green-600/20 bg-green-600/5 p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Savings</div>
          <div className="text-2xl font-mono font-bold text-green-700">{savings.toLocaleString()}</div>
          <div className="mt-1 text-xs text-gray-400">
            {savings === 0 ? '1.0× (no savings)' : `${totalSpeedup}× speedup (${savingsPct}%)`}
          </div>
        </div>
      </div>

      <NarrativeCallout label="Takeaway" tone="accent">
        This is the payoff of the previous acts: once the full group is fixed, the dense cost collapses to the orbit counts and output updates shown here.
      </NarrativeCallout>
    </div>
  );
}
