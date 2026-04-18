import CaseBadge from './CaseBadge.jsx';
import ExplorerMetricCard from './ExplorerMetricCard.jsx';
import ExplorerSectionCard, { AnchorLink } from './ExplorerSectionCard.jsx';
import GlossaryProse from './GlossaryProse.jsx';
import Latex from './Latex.jsx';
import { componentColor } from '../engine/componentPalette.js';

// Color palette — one hue per symbol ROLE (not per component). The same hex
// is used on the formula glyph, the glossary dt, and any related tooltip.
// Six roles total; see the design doc at
// ~/.claude/plans/this-is-a-bit-glistening-quiche.md for the locked palette.
const SYM = {
  k:       '#475569', // slate  — operand count
  group:   '#7C3AED', // violet — G, G_a, |G_a|, G_V (the -V/-W subscripts
                      //         themselves are colored as the V/W labels
                      //         below, not violet)
  element: '#EA580C', // burnt orange — g (group element summed over)
  cycle:   '#059669', // emerald — running indices and sizes: ℓ, c, L, R, Ω,
                      //           n_ℓ, n_c, n_Ω, c_Ω(g), c_R(g)
  alpha:   '#B45309', // amber  — α, α_a (accumulation)
  orbit:   '#0E7490', // cyan   — X, X/G_a, O, π (the -V/-W subscripts on X
                      //          and π are colored as the V/W labels below)
  vlabel:  '#4A7CFF', // blue   — V (free / output labels). Matches the V
                      //          dot in the Interaction Graph legend and
                      //          the IncidenceMatrix's inc-col-v.
  wlabel:  '#94A3B8', // slate  — W (summed / contracted labels). Matches
                      //          the W dot in the Interaction Graph legend.
};

// Canonical formula string — exported for tests and for readers who want the
// raw LaTeX without the color decorations. Intentionally spells out the full
// Burnside expansion (not just `\mu = (k-1)\prod_a M_a`) because the hero now
// renders the Burnside sum inline, and this constant is the single source of
// truth for "what the hero says".
const AGGREGATION_FORMULA = String.raw`\text{Total} \;=\; (k-1) \cdot \prod_{a} \tfrac{1}{|G_a|} \sum_{g \in G_a} \prod_{c} n_c \;+\; \prod_{a} \alpha_a`;

// Helper: \textcolor wrapper for composing LaTeX with a role color. The
// legend definitions below use these so that every math token inside a
// definition wears the same color as its dt symbol — and the same color as
// the token appears in the hero formula above.
const tc = (color, tex) => `\\textcolor{${color}}{${tex}}`;

// Glossary — "hybrid" policy: covers every symbol in the top line plus any
// piecewise symbol that appears in two or more rows. One-off symbols (Ω, R,
// n_Ω, c_Ω(g), H_a, h) are taught by leaf-badge tooltips where they appear.
// Seven rows total. Colors in the definition prose + math match the dt
// color for each symbol, so the reader's eye binds symbol ↔ definition ↔
// hero appearance by color.
const AGGREGATION_LEGEND = [
  {
    symbol: 'k',
    color: SYM.k,
    definition: `number of operand tensors in the einsum. $(${tc(SYM.k, 'k')}-1)$ binary multiplies combine each orbit representative.`,
  },
  {
    symbol: String.raw`G_a`,
    color: SYM.group,
    definition: `symmetry group acting on component $a$; $|${tc(SYM.group, 'G_a')}|$ is its order (number of group elements averaged over).`,
  },
  {
    symbol: 'g',
    color: SYM.element,
    definition: `one group element — a permutation of the component's labels. The sum averages over every $${tc(SYM.element, 'g')} \\in ${tc(SYM.group, 'G_a')}$.`,
  },
  {
    symbol: String.raw`n_c`,
    color: SYM.cycle,
    definition: `the common label-size inside cycle $${tc(SYM.cycle, 'c')}$ of $${tc(SYM.element, 'g')}$ — forced equal by the action, since $${tc(SYM.element, 'g')}$ permutes labels of equal size. The product $${tc(SYM.cycle, '\\prod_c n_c')}$ equals $|\\mathrm{Fix}(${tc(SYM.element, 'g')})| = |${tc(SYM.orbit, 'X')}^{${tc(SYM.element, 'g')}}|$ — the standard Burnside fixed-point set, written here as a product of cycle sizes. Cycles of the identity degenerate to singleton labels, so $${tc(SYM.cycle, '\\prod_c n_c')}$ collapses to $${tc(SYM.cycle, '\\prod_\\ell n_\\ell')}$ on the trivial / all-visible rows.`,
  },
  {
    // V and W carry their canonical page colors (blue / slate) — the same
    // hues used by the Interaction Graph legend and the Incidence Matrix
    // v/w columns. `entry.color` is a placeholder here; the inline colors
    // in `symbol` + `definition` carry the real visual binding.
    symbol: `${tc(SYM.vlabel, 'V')},\\ ${tc(SYM.wlabel, 'W')}`,
    color: SYM.vlabel,
    // Definition is a JSX fragment so we can color the prose phrases
    // "free (output) labels" / "summed (contracted) labels" directly.
    definition: (
      <>
        <span style={{ color: SYM.vlabel }}>free (output) labels</span>
        {' and '}
        <span style={{ color: SYM.wlabel }}>summed (contracted) labels</span>
        {', per component.'}
      </>
    ),
  },
  {
    symbol: `${tc(SYM.orbit, 'X')},\\ ${tc(SYM.orbit, 'X/G_a')},\\ ${tc(SYM.orbit, 'O')},\\ ${tc(SYM.orbit, '\\pi')}_{${tc(SYM.vlabel, 'V')}}(${tc(SYM.orbit, 'O')})`,
    color: SYM.orbit,
    definition: `assignment space $${tc(SYM.orbit, 'X')} = [n]^L$; its $${tc(SYM.group, 'G_a')}$-orbit decomposition $${tc(SYM.orbit, 'X/G_a')}$; a single orbit $${tc(SYM.orbit, 'O')}$; and its projection onto the free labels $${tc(SYM.orbit, '\\pi')}_{${tc(SYM.vlabel, 'V')}}(${tc(SYM.orbit, 'O')})$ — the distinct output bins that orbit touches.`,
  },
  {
    symbol: `${tc(SYM.alpha, '\\alpha')},\\ ${tc(SYM.alpha, '\\alpha_a')}`,
    color: SYM.alpha,
    definition: `accumulation cost. Per-component accumulation is $${tc(SYM.alpha, '\\alpha_a')}$ — one of the six case-specific formulas above. Global total is $${tc(SYM.alpha, '\\alpha')} = ${tc(SYM.alpha, '\\prod_a \\alpha_a')}$.`,
  },
];

// Six leaves of the current SHAPE × REGIME classification (see shapeSpec.js +
// regimeSpec.js). Each entry bundles the α_a formula and its layer tag; the
// leaf *id* is the canonical regime/shape id so CaseBadge can resolve its
// color + tooltip from the live spec — no duplicated content here.
const AGGREGATION_LEAVES = [
  {
    id: 'trivial',
    layer: 'shape',
    formula: String.raw`\textcolor{${SYM.cycle}}{\prod_{\ell \in L} n_\ell}`,
  },
  {
    id: 'allVisible',
    layer: 'shape',
    formula: String.raw`\textcolor{${SYM.cycle}}{\prod_{\ell \in \textcolor{${SYM.vlabel}}{V}} n_\ell}`,
  },
  {
    id: 'allSummed',
    layer: 'shape',
    formula: String.raw`|\textcolor{${SYM.orbit}}{X}/\textcolor{${SYM.group}}{G_a}| = \tfrac{1}{|\textcolor{${SYM.group}}{G_a}|} \textcolor{${SYM.element}}{\sum_{g}} \textcolor{${SYM.cycle}}{\prod_c n_c}`,
  },
  {
    id: 'singleton',
    layer: 'regime',
    formula: String.raw`\tfrac{\textcolor{${SYM.cycle}}{n_\Omega}}{|\textcolor{${SYM.group}}{G_a}|} \textcolor{${SYM.element}}{\sum_{g}} \Bigl(\textcolor{${SYM.cycle}}{\prod_{c \in R} n_c}\Bigr)\!\Bigl(\textcolor{${SYM.cycle}}{n_\Omega^{\,c_\Omega(g)}} - (\textcolor{${SYM.cycle}}{n_\Omega} - 1)^{\,c_\Omega(g)}\Bigr)`,
  },
  {
    id: 'directProduct',
    layer: 'regime',
    formula: String.raw`\Bigl(\textcolor{${SYM.cycle}}{\prod_{\ell \in \textcolor{${SYM.vlabel}}{V}} n_\ell}\Bigr) \cdot |\textcolor{${SYM.orbit}}{X}_{\textcolor{${SYM.wlabel}}{W}} / \textcolor{${SYM.group}}{G}_{\textcolor{${SYM.wlabel}}{W}}|`,
  },
  {
    id: 'bruteForceOrbit',
    layer: 'regime',
    formula: String.raw`\textcolor{${SYM.orbit}}{\sum_{O \in X/G_a}} |\textcolor{${SYM.orbit}}{\pi}_{\textcolor{${SYM.vlabel}}{V}}(\textcolor{${SYM.orbit}}{O})|`,
  },
];

// ---------------------------------------------------------------------------
// Hero formula — top line + piecewise definition of α_a.
// ---------------------------------------------------------------------------

const TOP_LINE = String.raw`\text{Total} \;=\; (\textcolor{${SYM.k}}{k}-1) \cdot \prod_a \tfrac{1}{|\textcolor{${SYM.group}}{G_a}|} \textcolor{${SYM.element}}{\sum_{g \in G_a}} \textcolor{${SYM.cycle}}{\prod_c n_c} \;+\; \prod_a \textcolor{${SYM.alpha}}{\alpha_a}`;
// The `g \in G_a` subscript above is fully orange; G_a in the subscript is
// close enough to the rest of the sum that unified orange reads better than
// splitting the colors inside a 7-point subscript.

function HeroFormulaBlock() {
  return (
    <div className="space-y-7">
      {/* Top line */}
      <div className="flex justify-center overflow-x-auto">
        <div className="min-w-0 text-[17px] sm:text-[19px]">
          <Latex display math={TOP_LINE} />
        </div>
      </div>

      {/* Piecewise — α_a defined by six leaves of the shape × regime ladder */}
      <div className="flex justify-center overflow-x-auto">
        <div
          className="grid items-center gap-x-5 gap-y-2 text-[14px]"
          style={{ gridTemplateColumns: 'auto auto 1fr auto' }}
        >
          {/* Prefix (α_a =) and big brace both span all 6 rows */}
          <div
            className="flex items-center gap-2 self-center pr-1"
            style={{ gridColumn: 1, gridRow: '1 / span 6' }}
          >
            <span className="text-[20px]" style={{ color: SYM.alpha }}>
              <Latex math={String.raw`\alpha_a`} />
            </span>
            <span className="text-[18px] text-muted-foreground">=</span>
          </div>
          <div
            aria-hidden="true"
            className="flex select-none items-center self-stretch font-serif font-thin leading-none"
            style={{
              gridColumn: 2,
              gridRow: '1 / span 6',
              fontSize: '140px',
              color: SYM.alpha,
            }}
          >
            {'{'}
          </div>

          {AGGREGATION_LEAVES.map((leaf) => (
            <FormulaRow key={leaf.id} leaf={leaf} />
          ))}
        </div>
      </div>

      <p className="text-center text-[12px] text-muted-foreground">
        Six paths through Section 4&rsquo;s decision ladder, in priority order — first
        matching leaf wins. Three Shape-layer leaves terminate immediately;
        Mixed components dispatch to one of three Regime-layer leaves. Hover
        any leaf to see the case details.
      </p>
    </div>
  );
}

function FormulaRow({ leaf }) {
  return (
    <>
      {/* Formula cell wrapped in CaseBadge passthrough mode — hovering the
          formula opens the same shape/regime tooltip as the leaf pill. */}
      <div className="py-1 pr-4" style={{ gridColumn: 3 }}>
        <CaseBadge regimeId={leaf.id} className="whitespace-nowrap">
          <Latex math={leaf.formula} />
        </CaseBadge>
      </div>
      <div className="flex items-center gap-2 whitespace-nowrap pl-2 text-[12px] text-muted-foreground" style={{ gridColumn: 4 }}>
        <span className="italic">if</span>
        <span className="rounded border border-border bg-surface-raised px-1.5 py-[1px] text-[9px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
          {leaf.layer}
        </span>
        <CaseBadge regimeId={leaf.id} size="xs" />
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// AggregationExplainer — wraps the hero and the glossary.
// ---------------------------------------------------------------------------

function AggregationExplainer() {
  return (
    <ExplorerSectionCard
      eyebrow={<AnchorLink anchorId="how-components-combine" labelText="How components combine">How components combine</AnchorLink>}
      id="how-components-combine"
      className="scroll-mt-24"
    >
      <div className="rounded-xl border border-border/60 bg-gradient-to-br from-surface-raised/60 to-white px-6 py-8">
        <HeroFormulaBlock />
      </div>

      <div className="mx-auto mt-8 max-w-2xl border-t border-border/60 pt-5">
        <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-5 gap-y-3 text-[12.5px] leading-relaxed text-muted-foreground">
          {AGGREGATION_LEGEND.map((entry) => (
            <div key={entry.symbol} className="contents">
              <dt
                className="justify-self-end whitespace-nowrap text-[15px]"
                style={{ color: entry.color }}
              >
                <Latex math={entry.symbol} />
              </dt>
              <dd>
                {typeof entry.definition === 'string'
                  ? <GlossaryProse text={entry.definition} />
                  : entry.definition}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </ExplorerSectionCard>
  );
}

// ---------------------------------------------------------------------------
// ComponentRecap — unchanged: color dot + case badge + label set per
// component. Component colors live only here; the formula above uses symbol
// roles instead.
// ---------------------------------------------------------------------------

function ComponentRecap({ components }) {
  if (!components?.length) return null;

  return (
    <div id="component-recap" className="flex flex-wrap items-center gap-2 scroll-mt-24">
      <span className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
        <AnchorLink anchorId="component-recap" labelText="Component recap">
          Component recap
        </AnchorLink>
      </span>
      {components.map((comp, idx) => (
        <span
          key={`component-recap-${idx}`}
          className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-raised py-1 pl-2 pr-2.5 text-[11px] leading-none text-muted-foreground"
          style={{ borderLeftColor: componentColor(idx), borderLeftWidth: 3 }}
        >
          <span
            aria-hidden="true"
            className="inline-block h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: componentColor(idx) }}
          />
          <span className="inline-flex items-center">
            <CaseBadge
              regimeId={comp.accumulation?.regimeId ?? comp.shape ?? comp.caseType}
              caseType={comp.caseType}
              size="xs"
            />
          </span>
          <span className="font-mono leading-none">{`{${(comp.labels ?? []).join(', ')}}`}</span>
        </span>
      ))}
    </div>
  );
}

export default function TotalCostView({ componentCosts, componentData, dimensionN, numTerms = 1 }) {
  if (!componentCosts || !componentData) return null;

  const { mu = 0, alpha = 0, mTotal = 0 } = componentCosts;
  const totalCost = mu + alpha;
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
          value={mu.toLocaleString()}
          detail={`μ = (k−1) · ∏ₐ Mₐ = (${Math.max(numTerms - 1, 0)}) · ${mTotal.toLocaleString()}`}
        />
        <ExplorerMetricCard
          label={<>Accumulation Cost <span className="normal-case">(α)</span></>}
          value={alpha.toLocaleString()}
          detail="α = ∏ₐ αₐ (distinct output-bin updates)"
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

// Exported for tests — catches silent regressions in the pedagogy.
export { AGGREGATION_FORMULA, AGGREGATION_LEGEND, AGGREGATION_LEAVES };
