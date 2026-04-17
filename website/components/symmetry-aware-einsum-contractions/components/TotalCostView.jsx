import CaseBadge from './CaseBadge.jsx';
import ExplorerMetricCard from './ExplorerMetricCard.jsx';
import GlossaryProse from './GlossaryProse.jsx';
import Latex from './Latex.jsx';

// Notation matches the Counting Convention band at the top of the page:
//   M — orbit count (size-aware Burnside, per component: M_a).
//   μ = (k − 1) · ∏_a M_a  — total multiplication cost.
//   α_a — accumulation cost per component; α = ∏_a α_a aggregates.
const AGGREGATION_FORMULA = String.raw`\mu = (k - 1)\!\prod_{a}\!M_a,\qquad \alpha = \prod_{a}\!\alpha_a,\qquad \text{Total} = \mu + \alpha`;

const AGGREGATION_LEGEND = [
  {
    symbol: String.raw`M_a`,
    definition: 'orbit count per component — the Burnside sum $\\tfrac{1}{|G_a|}\\sum_{g} \\prod_c n_c$. Shown as "M" in Act 4 per-component rows.',
  },
  {
    symbol: String.raw`\alpha_a`,
    definition: 'accumulation cost per component — distinct output bins written by that component. Shown as "α" in Act 4 per-component rows.',
  },
  {
    symbol: String.raw`\mu`,
    definition: 'total multiplication cost. $\\mu = (k-1)\\prod_a M_a$ — note the $(k-1)$ applies once to the product of orbit counts, so $\\mu \\neq \\prod_a \\mu_a$ (don\'t multiply per-component $\\mu$ values).',
  },
  {
    symbol: String.raw`\alpha`,
    definition: 'total accumulation cost. $\\alpha = \\prod_a \\alpha_a$ — per-component accumulations multiply cleanly.',
  },
  {
    symbol: 'k',
    definition: 'number of operand tensors in the einsum — $(k-1)$ multiplications combine each orbit representative, applied once to the global orbit product.',
  },
];

function AggregationExplainer() {
  return (
    <figure className="rounded-xl border border-border bg-white px-6 py-8 shadow-sm sm:px-10">
      <div className="mx-auto max-w-2xl">
        <div className="text-center text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          How components combine
        </div>
        <p className="mt-3 text-center text-sm leading-7 text-foreground/80">
          <GlossaryProse text="The group $G$ factors as $\prod_a G_a$, so per-component orbit counts $M_a$ and accumulations $\alpha_a$ multiply across components: $M = \prod_a M_a$ and $\alpha = \prod_a \alpha_a$. The $(k-1)$ factor that converts orbits into multiplications applies once to the global product, giving $\mu = (k-1)\cdot M$. Finally, the two totals $\mu$ and $\alpha$ add to give Total Cost." />
        </p>
      </div>

      <div className="mt-6 flex justify-center overflow-x-auto">
        <Latex display math={AGGREGATION_FORMULA} />
      </div>

      <figcaption className="mx-auto mt-6 max-w-md border-t border-border/60 pt-4">
        <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-2 text-[12px] leading-relaxed text-muted-foreground">
          {AGGREGATION_LEGEND.map((entry) => (
            <div key={entry.symbol} className="contents">
              <dt className="justify-self-end text-foreground">
                <Latex math={entry.symbol} />
              </dt>
              <dd>
                <GlossaryProse text={entry.definition} />
              </dd>
            </div>
          ))}
        </dl>
      </figcaption>
    </figure>
  );
}

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
          <CaseBadge
            regimeId={comp.accumulation?.regimeId ?? comp.shape ?? comp.caseType}
            caseType={comp.caseType}
            size="xs"
          />
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

      <AggregationExplainer />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ExplorerMetricCard
          label={<>Multiplication Cost <span className="normal-case">(μ)</span></>}
          value={evaluationCost.toLocaleString()}
          detail={`μ = (k−1)·M over ${orbitCount.toLocaleString()} orbit${orbitCount !== 1 ? 's' : ''}`}
        />
        <ExplorerMetricCard
          label={<>Accumulation Cost <span className="normal-case">(α)</span></>}
          value={reductionCost.toLocaleString()}
          detail="α = distinct output-bin updates"
        />
        <ExplorerMetricCard
          label="Total Cost"
          value={totalCost.toLocaleString()}
          detail="μ + α"
          className="border-coral/30 bg-coral-light"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ExplorerMetricCard
          label="Dense Cost"
          value={denseTotalCost.toLocaleString()}
          detail={
            <span className="flex flex-col gap-1">
              <Latex math={String.raw`\mu + \alpha = (k - 1)\,n^{|L|} + n^{|L|}`} />
              <span className="text-[11px] text-muted-foreground">
                <Latex math={String.raw`k=${numTerms},\ |L|=${allLabelCount},\ n=${dimensionN}`} />
              </span>
            </span>
          }
        />
        <ExplorerMetricCard
          label="Symmetry-Aware Cost"
          value={totalCost.toLocaleString()}
          detail="μ + α with the detected G applied"
        />
        <ExplorerMetricCard
          label="% savings"
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
