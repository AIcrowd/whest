import CaseBadge from './CaseBadge.jsx';
import ExplorerMetricCard from './ExplorerMetricCard.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

function InlineCodeList({ values }) {
  if (!values?.length) return <span className="text-muted-foreground">—</span>;
  return <code className="font-mono text-sm text-foreground">{values.join(', ')}</code>;
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
      <NarrativeCallout label="Why this matters">
        These totals combine the representative multiplications and the remaining output-bin updates into the final symmetry-aware contraction cost.
      </NarrativeCallout>

      <div className="rounded-xl border border-border bg-white shadow-sm">
        <Table className="text-sm">
          <TableHeader className="bg-surface-raised">
            <TableRow className="border-border hover:bg-surface-raised">
              <TableHead className="px-4 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Case</TableHead>
              <TableHead className="px-4 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Labels</TableHead>
              <TableHead className="px-4 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">V (free)</TableHead>
              <TableHead className="px-4 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">W (summed)</TableHead>
              <TableHead className="px-4 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">Group</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody className="divide-y divide-border">
            {components.map((comp, idx) => (
              <TableRow key={`comp-row-${idx}`} className="border-0 bg-surface hover:bg-surface-raised">
                <TableCell className="px-4 py-3 text-sm">
                  <CaseBadge caseType={comp.caseType} size="sm" />
                </TableCell>
                <TableCell className="px-4 py-3 text-sm">
                  <InlineCodeList values={comp.labels ?? []} />
                </TableCell>
                <TableCell className="px-4 py-3 text-sm">
                  <InlineCodeList values={comp.va ?? []} />
                </TableCell>
                <TableCell className="px-4 py-3 text-sm">
                  <InlineCodeList values={comp.wa ?? []} />
                </TableCell>
                <TableCell className="px-4 py-3 text-sm">
                  <code className="font-mono text-sm text-foreground">{comp.groupName ?? '—'}</code>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

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
          label="Savings"
          value={savings.toLocaleString()}
          detail={savings === 0 ? '1.0× (no savings)' : `${totalSpeedup}× speedup (${savingsPct}%)`}
          className="border-green-600/20 bg-green-600/5"
          valueClassName="text-green-700"
          detailClassName="text-gray-400"
        />
      </div>

      <NarrativeCallout label="Takeaway" tone="accent">
        This is the payoff of the previous acts: once the full group is fixed, the dense cost collapses to the orbit counts and output updates shown here.
      </NarrativeCallout>
    </div>
  );
}
