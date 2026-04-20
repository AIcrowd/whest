import CaseBadge from './CaseBadge.jsx';
import { AnchorLink } from './ExplorerSectionCard.jsx';
import GlossaryProse from './GlossaryProse.jsx';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import { getRegimePresentation } from './regimePresentation.js';
import { componentColor } from '../engine/componentPalette.js';
import {
  notationColor,
  notationLatex,
} from '../lib/notationSystem.js';

// Color palette — one hue per symbol ROLE (not per component). The same hex
// is used on the formula glyph, the glossary dt, and any related tooltip.
// Six roles total; see the design doc at
// ~/.claude/plans/this-is-a-bit-glistening-quiche.md for the locked palette.
const SYM = {
  k: notationColor('k_operands'),
  group: notationColor('g_component'),
  element: notationColor('g_element'),
  cycle: notationColor('n_cycle'),
  alpha: notationColor('alpha_total'),
  orbit: notationColor('x_space'),
  vlabel: notationColor('v_free'),
  wlabel: notationColor('w_summed'),
};

// Canonical formula string — exported for tests and for readers who want the
// raw LaTeX without the color decorations. Intentionally spells out the full
// Burnside expansion (not just `\mu = (k-1)\prod_a M_a`) because the hero now
// renders the Burnside sum inline, and this constant is the single source of
// truth for "what the hero says".
const AGGREGATION_FORMULA = String.raw`\text{Total} \;=\; (k-1) \cdot \prod_{a} \tfrac{1}{|G_a|} \sum_{g \in G_a} \prod_{c} n_c \;+\; \prod_{a} \alpha_a`;
const PIECEWISE_BRACE = String.raw`\left\{\vphantom{\begin{matrix}x\\x\\x\\x\\x\\x\end{matrix}}\right.`;
const PIECEWISE_PREFIX = String.raw`\textcolor{${notationColor('alpha_component')}}{${notationLatex('alpha_component')}} \, = \, \textcolor{#ef5a4c}{${PIECEWISE_BRACE}}`;

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
    definition: `number of operand tensors in the einsum. $(${tc(SYM.k, notationLatex('k_operands'))}-1)$ binary multiplies combine each orbit representative.`,
  },
  {
    symbol: notationLatex('g_component'),
    color: SYM.group,
    definition: `symmetry group acting on component $a$; $|${tc(SYM.group, notationLatex('g_component'))}|$ is its order (number of group elements averaged over).`,
  },
  {
    symbol: notationLatex('g_element'),
    color: SYM.element,
    definition: `one group element — a permutation of the component's labels. The sum averages over every $${tc(SYM.element, notationLatex('g_element'))} \\in ${tc(SYM.group, notationLatex('g_component'))}$.`,
  },
  {
    symbol: notationLatex('n_cycle'),
    color: SYM.cycle,
    definition: `the common label-size inside cycle $${tc(SYM.cycle, 'c')}$ of $${tc(SYM.element, notationLatex('g_element'))}$ — forced equal by the action, since $${tc(SYM.element, notationLatex('g_element'))}$ permutes labels of equal size. The product $${tc(SYM.cycle, `\\prod_c ${notationLatex('n_cycle')}`)}$ equals $|\\mathrm{Fix}(${tc(SYM.element, notationLatex('g_element'))})| = |${tc(SYM.orbit, notationLatex('x_space'))}^{${tc(SYM.element, notationLatex('g_element'))}}|$ — the standard Burnside fixed-point set, written here as a product of cycle sizes. Cycles of the identity degenerate to singleton labels, so $${tc(SYM.cycle, `\\prod_\\ell ${notationLatex('n_label')}`)}$ collapses to $${tc(SYM.cycle, `\\prod_\\ell ${notationLatex('n_label')}`)}$ on the trivial / all-visible rows.`,
  },
  {
    // V and W carry their canonical page colors (blue / slate) — the same
    // hues used by the Interaction Graph legend and the Incidence Matrix
    // v/w columns. `entry.color` is a placeholder here; the inline colors
    // in `symbol` + `definition` carry the real visual binding.
    symbol: `${tc(SYM.vlabel, notationLatex('v_free'))},\\ ${tc(SYM.wlabel, notationLatex('w_summed'))}`,
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
    symbol: `${notationLatex('x_space')},\\ ${notationLatex('orbit_space_component')},\\ ${notationLatex('orbit_o')},\\ ${notationLatex('projection_pi_v_free')}`,
    color: SYM.orbit,
    definition: `assignment space $${notationLatex('x_space')} = [n]^{${notationLatex('l_labels')}}$; its $${notationLatex('orbit_space_component')}$-orbit decomposition; a single orbit $${notationLatex('orbit_o')}$; and its projection onto the free labels $${notationLatex('projection_pi_v_free')}$ — the distinct output bins that orbit touches.`,
  },
  {
    symbol: `${tc(SYM.alpha, notationLatex('alpha_total'))},\\ ${tc(SYM.alpha, notationLatex('alpha_component'))}`,
    color: SYM.alpha,
    definition: `accumulation cost. Per-component accumulation is $${tc(SYM.alpha, notationLatex('alpha_component'))}$ — one of the shape- and regime-specific formulas above. Global total is $${tc(SYM.alpha, notationLatex('alpha_total'))} = ${tc(SYM.alpha, `\\prod_a ${notationLatex('alpha_component')}`)}$.`,
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
    formula: String.raw`\textcolor{${SYM.cycle}}{\prod_{\ell \in ${notationLatex('l_labels')}} ${notationLatex('n_label')}}`,
  },
  {
    id: 'allVisible',
    layer: 'shape',
    formula: String.raw`\textcolor{${SYM.cycle}}{\prod_{\ell \in \textcolor{${SYM.vlabel}}{${notationLatex('v_free')}}} ${notationLatex('n_label')}}`,
  },
  {
    id: 'allSummed',
    layer: 'shape',
    formula: String.raw`|\textcolor{${SYM.orbit}}{X}/\textcolor{${SYM.group}}{${notationLatex('g_component')}}| = \tfrac{1}{|\textcolor{${SYM.group}}{${notationLatex('g_component')}}|} \textcolor{${SYM.element}}{\sum_{g}} \textcolor{${SYM.cycle}}{\prod_c ${notationLatex('n_cycle')}}`,
  },
  {
    id: 'singleton',
    layer: 'regime',
    formula: String.raw`\tfrac{\textcolor{${SYM.cycle}}{${notationLatex('n_omega')}}}{|\textcolor{${SYM.group}}{${notationLatex('g_component')}}|} \textcolor{${SYM.element}}{\sum_{g}} \Bigl(\textcolor{${SYM.cycle}}{\prod_{c \in ${notationLatex('r_complement')}} ${notationLatex('n_cycle')}}\Bigr)\!\Bigl(\textcolor{${SYM.cycle}}{${notationLatex('n_omega')}^{\,c_\Omega(g)}} - (\textcolor{${SYM.cycle}}{${notationLatex('n_omega')}} - 1)^{\,c_\Omega(g)}\Bigr)`,
  },
  {
    id: 'directProduct',
    layer: 'regime',
    formula: String.raw`\Bigl(\textcolor{${SYM.cycle}}{\prod_{\ell \in \textcolor{${SYM.vlabel}}{${notationLatex('v_free')}}} ${notationLatex('n_label')}}\Bigr) \cdot |\textcolor{${SYM.orbit}}{X}_{\textcolor{${SYM.wlabel}}{${notationLatex('w_summed')}}} / \textcolor{${SYM.group}}{${notationLatex('g_w_factor')}}|`,
  },
  {
    id: 'young',
    layer: 'regime',
    formula: String.raw`\textcolor{${SYM.cycle}}{n_L^{|\textcolor{${SYM.vlabel}}{${notationLatex('v_free')}}|}} \cdot \binom{\textcolor{${SYM.cycle}}{n_L} + |\textcolor{${SYM.wlabel}}{${notationLatex('w_summed')}}| - 1}{|\textcolor{${SYM.wlabel}}{${notationLatex('w_summed')}}|}`,
  },
  {
    id: 'bruteForceOrbit',
    layer: 'regime',
    formula: String.raw`\textcolor{${SYM.orbit}}{\sum_{${notationLatex('orbit_o')} \in X/G_a}} |\textcolor{${SYM.orbit}}{\pi}_{\textcolor{${SYM.vlabel}}{${notationLatex('v_free')}}}(\textcolor{${SYM.orbit}}{${notationLatex('orbit_o')}})|`,
  },
];

// ---------------------------------------------------------------------------
// Hero formula — top line + piecewise definition of α_a.
// ---------------------------------------------------------------------------

const TOP_LINE = String.raw`\text{Total} \;=\; (\textcolor{${SYM.k}}{${notationLatex('k_operands')}}-1) \cdot \prod_a \tfrac{1}{|\textcolor{${SYM.group}}{${notationLatex('g_component')}}|} \textcolor{${SYM.element}}{\sum_{g \in ${notationLatex('g_component')}}} \textcolor{${SYM.cycle}}{\prod_c ${notationLatex('n_cycle')}} \;+\; \prod_a \textcolor{${SYM.alpha}}{${notationLatex('alpha_component')}}`;
// The `g \in G_a` subscript above is fully orange; G_a in the subscript is
// close enough to the rest of the sum that unified orange reads better than
// splitting the colors inside a 7-point subscript.

function HeroFormulaBlock() {
  return (
    <div className="space-y-7">
      {/* Top line */}
      <div className="flex justify-center overflow-x-auto overflow-y-visible">
        <div className="min-w-0 text-[17px] sm:text-[19px]">
          <Latex display math={TOP_LINE} />
        </div>
      </div>

      {/* Piecewise — α_a defined by six leaves of the shape × regime ladder */}
      <div className="flex justify-center overflow-x-auto overflow-y-visible">
        <div
          className="grid items-center gap-x-5 gap-y-2 text-[14px]"
          style={{ gridTemplateColumns: 'auto auto 1fr auto' }}
        >
          {/* Keep α_a, =, and the brace in one KaTeX fragment so alignment comes from math layout, not CSS boxes. */}
          <div
            className="flex items-center justify-center self-stretch overflow-visible pr-1"
            style={{
              gridColumn: '1 / span 2',
              gridRow: '1 / span 6',
            }}
          >
            <Latex math={PIECEWISE_PREFIX} />
          </div>

          {AGGREGATION_LEAVES.map((leaf) => (
            <FormulaRow key={leaf.id} leaf={leaf} />
          ))}
        </div>
      </div>
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
  // Editorial flat layout — no outer card, no inner gradient-box. The
  // formula and its glossary float inside Section 5's ExplorerSectionCard
  // with only whitespace framing them, so the reader's eye travels
  // directly from the top 'Total = …' line down through the six
  // piecewise α_a cases without being boxed three times over.
  return (
    <section
      id="how-components-combine"
      aria-labelledby="how-components-combine-sr"
      className="scroll-mt-24"
    >
      <h3 id="how-components-combine-sr" className="sr-only">
        How components combine
      </h3>
      <div className="py-4">
        <HeroFormulaBlock />
      </div>

      <div className="mx-auto mt-10 max-w-2xl border-t border-gray-100 pt-6">
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
    </section>
  );
}

// ---------------------------------------------------------------------------
// ComponentRecap — inline legend entries (not pills). Component identity
// color lives as a thin vertical rule left of each entry; the CaseBadge
// carries regime semantics; mono-italic labels carry the axis set. Drops
// the triple-boundary pill (rounded-full border + 3px left rail + inner
// color dot) which read as a chip row rather than an editorial legend.
// Component identity colors (componentColor) still thread through to the
// Interaction Graph hulls in Act 4 and the combine formula above — only
// the visual treatment here flattens.
// ---------------------------------------------------------------------------

function ComponentRecap({ components }) {
  if (!components?.length) return null;

  return (
    <div id="component-recap" className="flex flex-wrap items-center gap-x-6 gap-y-2 scroll-mt-24">
      <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
        <AnchorLink anchorId="component-recap" labelText="Component recap">
          Component recap
        </AnchorLink>
      </span>
      {components.map((comp, idx) => {
        const regimeId = comp.accumulation?.regimeId ?? comp.shape;
        const regimeColor = getRegimePresentation(regimeId)?.color;
        return (
          <span
            key={`component-recap-${idx}`}
            className="inline-flex items-center gap-2 text-[12px] leading-none text-gray-600"
          >
            <span
              aria-hidden="true"
              className="inline-block h-4 w-[3px] shrink-0 rounded-[2px]"
              style={{ backgroundColor: regimeColor ?? componentColor(idx) }}
            />
            <CaseBadge
              regimeId={regimeId}
              size="xs"
            />
            <span className="font-mono text-gray-900">{`{${(comp.labels ?? []).join(', ')}}`}</span>
          </span>
        );
      })}
    </div>
  );
}

function ComparisonMetric({ label, value, valueClassName }) {
  return (
    <div className="flex min-h-[160px] flex-col items-center justify-center px-4 py-[28px] text-center sm:px-6">
      <div
        className={[
          'font-serif text-[54px] leading-[0.95] tracking-[-0.03em] text-gray-900',
          valueClassName,
        ].filter(Boolean).join(' ')}
      >
        {value}
      </div>
      <div className="mt-2 text-[10px] font-semibold uppercase tracking-[0.22em] text-gray-600">
        {label}
      </div>
    </div>
  );
}

function SupportingMetric({ label, value, valueClassName }) {
  return (
    <div className="flex min-h-[126px] flex-col items-center justify-center px-4 py-[22px] text-center sm:px-5">
      <div
        className={[
          'font-serif text-[24px] leading-none tracking-[-0.02em] text-gray-900',
          valueClassName,
        ].filter(Boolean).join(' ')}
      >
        {value}
      </div>
      <div className="mt-2 text-[10px] font-semibold uppercase tracking-[0.22em] text-gray-600">
        {label}
      </div>
    </div>
  );
}

function EditorialComparisonSpread({ topMetrics, supportingMetrics }) {
  return (
    <div className="section5-editorial-spread mx-auto max-w-[44rem] bg-white px-[6px] py-[6px] text-center">
      <div className="section5-editorial-header mb-2 flex items-center justify-center gap-[14px]">
        <span aria-hidden className="h-px w-[64px] bg-[#f3c5bf]" />
        <h3 className="font-serif text-[24px] text-gray-800">Cost Savings</h3>
        <span aria-hidden className="h-px w-[64px] bg-[#f3c5bf]" />
      </div>

      <div className="section5-band-top relative grid grid-cols-1 sm:grid-cols-2">
        <div
          aria-hidden="true"
          className="section5-floating-sep section5-pair-sep absolute bottom-[18%] left-1/2 top-[18%] hidden w-px -translate-x-1/2 bg-gray-100 sm:block"
        />
        {topMetrics.map((metric) => (
          <ComparisonMetric
            key={metric.label}
            label={metric.label}
            value={metric.value}
            valueClassName={metric.valueClassName}
          />
        ))}
      </div>

      <div className="section5-band-bottom relative border-t border-b border-gray-100">
        <div
          aria-hidden="true"
          className="section5-floating-sep absolute bottom-[18%] left-1/2 top-[18%] hidden w-px -translate-x-1/2 bg-gray-100 sm:block lg:hidden"
        />
        <div
          aria-hidden="true"
          className="section5-floating-sep section5-quad-sep-1 absolute bottom-[18%] left-1/4 top-[18%] hidden w-px -translate-x-1/2 bg-gray-100 lg:block"
        />
        <div
          aria-hidden="true"
          className="section5-floating-sep section5-quad-sep-2 absolute bottom-[18%] left-1/2 top-[18%] hidden w-px -translate-x-1/2 bg-gray-100 lg:block"
        />
        <div
          aria-hidden="true"
          className="section5-floating-sep section5-quad-sep-3 absolute bottom-[18%] left-3/4 top-[18%] hidden w-px -translate-x-1/2 bg-gray-100 lg:block"
        />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {supportingMetrics.map((metric) => (
            <SupportingMetric
              key={metric.label}
              label={metric.label}
              value={metric.value}
              valueClassName={metric.valueClassName}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function TotalCostView({
  componentCosts,
  componentData,
  dimensionN,
  numTerms = 1,
}) {
  if (!componentCosts || !componentData) return null;

  const { mu = 0, alpha = 0 } = componentCosts;
  const totalCost = mu + alpha;
  const allLabelCount = componentData?.components?.reduce((sum, comp) => sum + comp.labels.length, 0) ?? 0;
  const denseTuples = Math.pow(dimensionN, allLabelCount);
  const denseTotalCost = Math.max(numTerms - 1, 0) * denseTuples + denseTuples;
  const totalSpeedup = totalCost > 0 ? (denseTotalCost / totalCost).toFixed(1) : '1.0';
  const savingsPct = denseTotalCost > 0 ? (((denseTotalCost - totalCost) / denseTotalCost) * 100).toFixed(1) : '0';
  const { components = [] } = componentData;
  const TOP_COMPARISON_METRICS = [
    {
      label: 'Dense Cost',
      value: denseTotalCost.toLocaleString(),
    },
    {
      label: 'Symmetry-Aware Cost',
      value: totalCost.toLocaleString(),
      valueClassName: 'text-coral',
    },
  ];
  const SUPPORTING_METRICS = [
    {
      label: 'Multiplication Cost (μ)',
      value: mu.toLocaleString(),
    },
    {
      label: 'Accumulation Cost (α)',
      value: alpha.toLocaleString(),
    },
    {
      label: 'Speedup',
      value: (
        <>
          <span>{totalSpeedup}</span>
          <span className="ml-1 text-gray-900">×</span>
        </>
      ),
    },
    {
      label: '% Savings',
      value: `${savingsPct}%`,
      valueClassName: Number(savingsPct) > 0 ? 'text-green-700' : undefined,
    },
  ];

  return (
    <div className="space-y-8">
      <ComponentRecap components={components} />

      <AggregationExplainer />

      <EditorialComparisonSpread
        topMetrics={TOP_COMPARISON_METRICS}
        supportingMetrics={SUPPORTING_METRICS}
      />
    </div>
  );
}

// Exported for tests — catches silent regressions in the pedagogy.
export { AGGREGATION_FORMULA, AGGREGATION_LEGEND, AGGREGATION_LEAVES };
