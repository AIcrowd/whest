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
import {
  explorerThemeColor,
  getExplorerThemeCssVariables,
} from '../lib/explorerTheme.js';

// Canonical formula string — exported for tests and for readers who want the
// raw LaTeX without color decorations. The multiplication side expands M_a by
// Burnside; the accumulation side remains α_a because α_a is an
// orbit-projection count over product orbits, not generally a simple quotient
// of the visible labels alone.
const AGGREGATION_FORMULA = String.raw`\text{Total Cost} \;=\; (k-1) \cdot \prod_{a} \tfrac{1}{|G_a|} \sum_{g \in G_a} \prod_{c} n_c \;+\; \prod_{a} \alpha_a`;
const SECTION_FIVE_INTRO_PARAGRAPH =
  'The preceding sections have produced a detected pointwise group and decomposed its label action into independent components. The final step is not to divide the dense computation by the group order, but to combine two exact orbit counts: representative products and the output updates induced by those representatives.';

const SECTION_FIVE_INTRO_LEAD =
  'For each component, let M_a be the number of product orbits and let α_a be the number of output-bin updates induced by those orbits. With k operand tensors, the direct symmetry-aware cost is';

const SECTION_FIVE_INTRO_CLOSE =
  'The expanded equation below shows how M_a is computed by Burnside when a closed form applies, while α_a is selected from the shape and regime ladder. The numerical spread then compares this direct symmetry-aware count with the naive dense baseline.';
const SECTION_FIVE_TOTAL_FORMULA = String.raw`\mathrm{Total\ Cost} = \mu + \alpha`;
const SECTION_FIVE_MU_FORMULA = String.raw`\mu = (k-1)\prod_a M_a`;
const SECTION_FIVE_ALPHA_FORMULA = String.raw`\alpha = \prod_a \alpha_a`;
const SECTION_FIVE_THEME_OVERRIDE = 'editorial-noir-math';
const PIECEWISE_BRACE = String.raw`\left\{\vphantom{\begin{matrix}x\\x\\x\\x\\x\\x\end{matrix}}\right.`;
const PIECEWISE_SCOPE_NOTE =
  `The brace below defines only the per-component accumulation term $${notationLatex('alpha_component')}$. It counts output projections of product orbits: an orbit that touches several output bins contributes once to each such bin.`;


// Helper: \textcolor wrapper for composing LaTeX with a role color. The
// legend definitions below use these so that every math token inside a
// definition wears the same color as its dt symbol — and the same color as
// the token appears in the hero formula above.
const tc = (color, tex) => `\\textcolor{${color}}{${tex}}`;
const sumOver = (binder) => String.raw`\sum_{${binder}}`;
const productOver = (binder, body) => String.raw`\prod_{${binder}} ${body}`;
const inSet = (lhs, rhs) => `${lhs}\\,\\in\\,${rhs}`;

function getSymPalette(themeOverride = SECTION_FIVE_THEME_OVERRIDE) {
  return {
    k: notationColor('k_operands', themeOverride),
    localGroup: notationColor('g_component', themeOverride),
    summedGroup: notationColor('g_w_factor', themeOverride),
    element: notationColor('g_element', themeOverride),
    cycleCount: notationColor('n_cycle', themeOverride),
    labelCount: notationColor('n_label', themeOverride),
    youngCount: notationColor('n_l', themeOverride),
    omegaSize: notationColor('n_omega', themeOverride),
    omegaExponent: notationColor('c_omega_cycles', themeOverride),
    alpha: notationColor('alpha_total', themeOverride),
    ambient: notationColor('x_space', themeOverride),
    orbitObject: notationColor('orbit_o', themeOverride),
    projection: notationColor('projection_pi_v_free', themeOverride),
    shapeSet: notationColor('l_labels', themeOverride),
    vlabel: notationColor('v_free', themeOverride),
    wlabel: notationColor('w_summed', themeOverride),
  };
}

function getPiecewisePrefix(themeOverride = SECTION_FIVE_THEME_OVERRIDE) {
  return String.raw`\textcolor{${notationColor('alpha_component', themeOverride)}}{${notationLatex('alpha_component')}} \, = \, ${PIECEWISE_BRACE}`;
}

// Glossary — "hybrid" policy: covers every symbol in the top line plus any
// piecewise symbol that appears in two or more rows. One-off symbols (Ω, R,
// n_Ω, c_Ω(g), H_a, h) are taught by leaf-badge tooltips where they appear.
// Seven rows total. Colors in the definition prose + math match the dt
// color for each symbol, so the reader's eye binds symbol ↔ definition ↔
// hero appearance by color.
function getAggregationLegend(themeOverride = SECTION_FIVE_THEME_OVERRIDE) {
  const SYM = getSymPalette(themeOverride);
  return [
    {
      symbol: 'a',
      color: notationColor('l_labels', themeOverride),
      definition: `component index. Products over $a$ run across the independent components, and subscripts like $${notationLatex('g_component')}$ or $${tc(SYM.alpha, notationLatex('alpha_component'))}$ mean “restricted to component $a$.”`,
    },
    {
      symbol: 'k',
      color: SYM.k,
      definition: `number of operand tensors in the einsum. $(${tc(SYM.k, notationLatex('k_operands'))}-1)$ binary multiplies combine each orbit representative.`,
    },
    {
      symbol: notationLatex('g_component'),
      color: SYM.localGroup,
      definition: `detected pointwise symmetry group restricted to component $a$. Its elements act on full label assignments and are the relabelings accepted by the $${tc(notationColor('sigma_row_move', themeOverride), notationLatex('sigma_row_move'))}$-loop under the declared equality symmetries; $|${tc(SYM.localGroup, notationLatex('g_component'))}|$ is the order averaged over in Burnside counts.`,
    },
    {
      symbol: notationLatex('g_element'),
      color: SYM.element,
      definition: `one group element — a permutation of the component's labels. The sum averages over every $${tc(SYM.element, notationLatex('g_element'))} \\in ${tc(SYM.localGroup, notationLatex('g_component'))}$.`,
    },
    {
      symbol: notationLatex('n_cycle'),
      color: SYM.cycleCount,
      definition: `the common label-size inside cycle $${tc(SYM.cycleCount, 'c')}$ of $${tc(SYM.element, notationLatex('g_element'))}$. The product $${productOver('c', tc(SYM.cycleCount, notationLatex('n_cycle')))}$ equals the fixed-assignment count $|\\mathrm{Fix}(${tc(SYM.element, notationLatex('g_element'))})|$, which is the Burnside ingredient for the product-orbit count $${notationLatex('m_component')}.`,
    },
    {
      symbol: `${tc(SYM.vlabel, notationLatex('v_free'))},\\ ${tc(SYM.wlabel, notationLatex('w_summed'))}`,
      color: SYM.vlabel,
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
      symbol: `${tc(SYM.ambient, notationLatex('x_space'))},\\ ${tc(SYM.ambient, notationLatex('orbit_space_component'))},\\ ${tc(SYM.orbitObject, notationLatex('orbit_o'))},\\ ${tc(SYM.projection, notationLatex('projection_pi_v_free'))}`,
      color: SYM.ambient,
      definition: `assignment space $${tc(SYM.ambient, notationLatex('x_space'))}$; quotient $${tc(SYM.ambient, notationLatex('orbit_space_component'))}$ of full assignments by the pointwise component group; one product orbit $${tc(SYM.orbitObject, notationLatex('orbit_o'))}$; and the projection $${tc(SYM.projection, notationLatex('projection_pi_v_free'))}$ that records which output bins that orbit touches.`,
    },
    {
      symbol: `${tc(SYM.orbitObject, notationLatex('omega_orbit'))},\\ ${tc(SYM.omegaSize, notationLatex('n_omega'))},\\ ${tc(SYM.omegaExponent, notationLatex('c_omega_cycles'))}`,
      color: SYM.orbitObject,
      definition: `singleton-regime symbols: $${tc(SYM.orbitObject, notationLatex('omega_orbit'))} = ${tc(SYM.localGroup, notationLatex('g_component'))} \\cdot v$ is the orbit of the single free label $v$ under the component symmetry group. $${tc(SYM.omegaSize, notationLatex('n_omega'))}$ is the size shared by the labels in $${tc(SYM.orbitObject, notationLatex('omega_orbit'))}$, and $${tc(SYM.omegaExponent, notationLatex('c_omega_cycles'))}$ counts how many cycles of the group element $${tc(SYM.element, notationLatex('g_element'))}$ lie inside that distinguished class.`,
    },
    {
      symbol: `${tc(SYM.alpha, notationLatex('alpha_total'))},\\ ${tc(SYM.alpha, notationLatex('alpha_component'))}`,
      color: SYM.alpha,
      definition: `accumulation/output-update cost. Per component, $${tc(SYM.alpha, notationLatex('alpha_component'))}$ counts one update for each output bin touched by each product orbit. Equivalently, it is the sum over product orbits of the number of distinct free-label projections they touch. Globally, independent components multiply to $${tc(SYM.alpha, notationLatex('alpha_total'))} = ${productOver('a', tc(SYM.alpha, notationLatex('alpha_component')))}$.`,
    },
  ];
}

// Six leaves of the current SHAPE × REGIME classification (see shapeSpec.js +
// regimeSpec.js). Each entry bundles the α_a formula and its layer tag; the
// leaf *id* is the canonical regime/shape id so CaseBadge can resolve its
// color + tooltip from the live spec — no duplicated content here.
function getAggregationLeaves(themeOverride = SECTION_FIVE_THEME_OVERRIDE) {
  const SYM = getSymPalette(themeOverride);
  return [
    {
      id: 'trivial',
      layer: 'shape',
      formula: String.raw`\prod_{\ell \in ${notationLatex('l_labels')}} ${tc(SYM.labelCount, notationLatex('n_label'))}`,
    },
    {
      id: 'allVisible',
      layer: 'shape',
      formula: String.raw`\prod_{\ell \in ${tc(SYM.vlabel, notationLatex('v_free'))}} ${tc(SYM.labelCount, notationLatex('n_label'))}`,
    },
    {
      id: 'allSummed',
      layer: 'shape',
      formula: String.raw`|${tc(SYM.ambient, notationLatex('x_space'))}/${tc(SYM.localGroup, notationLatex('g_component'))}| = \tfrac{1}{|${tc(SYM.localGroup, notationLatex('g_component'))}|} ${sumOver(tc(SYM.element, notationLatex('g_element')))} ${productOver('c', tc(SYM.cycleCount, notationLatex('n_cycle')))}`,
    },
    {
      id: 'singleton',
      layer: 'regime',
      formula: String.raw`\tfrac{${tc(SYM.omegaSize, notationLatex('n_omega'))}}{|${tc(SYM.localGroup, notationLatex('g_component'))}|} ${sumOver(tc(SYM.element, notationLatex('g_element')))} \Bigl(${productOver(`c \\in ${notationLatex('r_complement')}`, tc(SYM.cycleCount, notationLatex('n_cycle')))}\Bigr)\!\Bigl(${tc(SYM.omegaSize, notationLatex('n_omega'))}^{\,${tc(SYM.omegaExponent, notationLatex('c_omega_cycles'))}} - (${tc(SYM.omegaSize, notationLatex('n_omega'))} - 1)^{\,${tc(SYM.omegaExponent, notationLatex('c_omega_cycles'))}}\Bigr)`,
    },
    {
      id: 'directProduct',
      layer: 'regime',
      formula: String.raw`\Bigl(${productOver(`\\ell \\in ${tc(SYM.vlabel, notationLatex('v_free'))}`, tc(SYM.labelCount, notationLatex('n_label')))}\Bigr) \cdot |${tc(SYM.ambient, 'X')}_{${tc(SYM.wlabel, notationLatex('w_summed'))}} / ${tc(SYM.summedGroup, notationLatex('g_w_factor'))}|`,
    },
    {
      id: 'young',
      layer: 'regime',
      formula: String.raw`${tc(SYM.youngCount, notationLatex('n_l'))}^{|${tc(SYM.vlabel, notationLatex('v_free'))}|} \cdot \binom{${tc(SYM.youngCount, notationLatex('n_l'))} + |${tc(SYM.wlabel, notationLatex('w_summed'))}| - 1}{|${tc(SYM.wlabel, notationLatex('w_summed'))}|}`,
    },
    {
      id: 'bruteForceOrbit',
      layer: 'regime',
      formula: String.raw`${sumOver(`${tc(SYM.orbitObject, notationLatex('orbit_o'))} \\in ${tc(SYM.ambient, notationLatex('x_space'))}/${tc(SYM.localGroup, notationLatex('g_component'))}`)} |${tc(SYM.projection, notationLatex('projection_pi_v_free'))}_{${tc(SYM.vlabel, notationLatex('v_free'))}}(${tc(SYM.orbitObject, notationLatex('orbit_o'))})|`,
    },
  ];
}

function getPiecewiseRowSpan() {
  return getAggregationLeaves().length;
}

// ---------------------------------------------------------------------------
// Hero formula — top line + piecewise definition of α_a.
// ---------------------------------------------------------------------------

function getTopLine(themeOverride = SECTION_FIVE_THEME_OVERRIDE) {
  const SYM = getSymPalette(themeOverride);
  return String.raw`\text{Total Cost} \;=\; (${tc(SYM.k, notationLatex('k_operands'))}-1) \cdot \prod_a \tfrac{1}{|${tc(SYM.localGroup, notationLatex('g_component'))}|} ${sumOver(inSet(tc(SYM.element, notationLatex('g_element')), tc(SYM.localGroup, notationLatex('g_component'))))} ${productOver('c', tc(SYM.cycleCount, notationLatex('n_cycle')))} \;+\; \prod_a ${tc(SYM.alpha, notationLatex('alpha_component'))}`;
}
// The Burnside binder above now colors `g` and `G_a` separately so the
// acting element and the local group stay visually distinct even inside
// the tight 7-point sum subscript.

function HeroFormulaBlock({ themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  const topLine = getTopLine(themeOverride);
  const piecewisePrefix = getPiecewisePrefix(themeOverride);
  const aggregationLeaves = getAggregationLeaves(themeOverride);
  const piecewiseRowSpan = getPiecewiseRowSpan();
  return (
    <div className="space-y-7">
      {/* Top line */}
      <div className="flex justify-center overflow-x-auto overflow-y-visible">
        <div className="min-w-0 text-[17px] sm:text-[19px]">
          <Latex display math={topLine} themeOverride={themeOverride} />
        </div>
      </div>

      {/* Piecewise — α_a defined by six leaves of the shape × regime ladder */}
      <div className="text-center">
        <div className="mx-auto max-w-2xl font-serif text-[14px] leading-[1.7] text-gray-700">
          <InlineMathText themeOverride={themeOverride}>{PIECEWISE_SCOPE_NOTE}</InlineMathText>
        </div>
      </div>
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
              gridRow: `1 / span ${piecewiseRowSpan}`,
            }}
          >
            <Latex math={piecewisePrefix} themeOverride={themeOverride} />
          </div>

          {aggregationLeaves.map((leaf) => (
            <FormulaRow key={leaf.id} leaf={leaf} themeOverride={themeOverride} />
          ))}
        </div>
      </div>
    </div>
  );
}

function FormulaRow({ leaf, themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  return (
    <>
      {/* Formula cell wrapped in CaseBadge passthrough mode — hovering the
          formula opens the same shape/regime tooltip as the leaf pill. */}
      <div className="py-1 pr-4" style={{ gridColumn: 3 }}>
        <CaseBadge
          regimeId={leaf.id}
          className="whitespace-nowrap"
          themeOverride={themeOverride}
          presentationThemeOverride={null}
        >
          <Latex math={leaf.formula} themeOverride={themeOverride} />
        </CaseBadge>
      </div>
      <div className="flex items-center gap-2 whitespace-nowrap pl-2 text-[12px] text-muted-foreground" style={{ gridColumn: 4 }}>
        <span className="italic">if</span>
        <span className="rounded border border-border bg-surface-raised px-1.5 py-[1px] text-[9px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
          {leaf.layer}
        </span>
        <CaseBadge
          regimeId={leaf.id}
          size="xs"
          themeOverride={themeOverride}
          presentationThemeOverride={null}
        />
        {leaf.cue ? (
          <span className="text-[11px] italic text-gray-500">
            {leaf.cue}
          </span>
        ) : null}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// AggregationExplainer — wraps the hero and the glossary.
// ---------------------------------------------------------------------------

function AggregationExplainer({ themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  const aggregationLegend = getAggregationLegend(themeOverride);
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
        <HeroFormulaBlock themeOverride={themeOverride} />
      </div>

      <div className="mx-auto mt-10 max-w-2xl border-t border-gray-100 pt-6">
        <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-5 gap-y-3 text-[12.5px] leading-relaxed text-muted-foreground">
          {aggregationLegend.map((entry) => (
            <div key={entry.symbol} className="contents">
              <dt
                className="justify-self-end whitespace-nowrap text-[15px]"
                style={{ color: entry.color }}
              >
                <Latex math={entry.symbol} themeOverride={themeOverride} />
              </dt>
              <dd>
                {typeof entry.definition === 'string'
                  ? <GlossaryProse text={entry.definition} themeOverride={themeOverride} />
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
              themeOverride={SECTION_FIVE_THEME_OVERRIDE}
              presentationThemeOverride={null}
            />
            <span className="font-mono text-gray-900">{`{${(comp.labels ?? []).join(', ')}}`}</span>
          </span>
        );
      })}
    </div>
  );
}

function SectionFiveIntroBlock({ themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  return (
    <div className="mx-auto max-w-[48rem] space-y-7 text-center">
      <p
        className="font-serif text-[17px] leading-[1.75] text-gray-700"
        style={{ textAlign: 'justify' }}
      >
        <InlineMathText themeOverride={themeOverride}>{SECTION_FIVE_INTRO_PARAGRAPH}</InlineMathText>
      </p>

      <div className="space-y-7">
        <p
          className="font-serif text-[17px] leading-[1.75] text-gray-700"
          style={{ textAlign: 'justify' }}
        >
          <InlineMathText themeOverride={themeOverride}>{SECTION_FIVE_INTRO_LEAD}</InlineMathText>
        </p>

        <div className="space-y-6">
          <div className="flex justify-center overflow-x-auto overflow-y-visible">
            <div className="min-w-0 text-[18px] sm:text-[20px]">
              <Latex display math={SECTION_FIVE_TOTAL_FORMULA} themeOverride={themeOverride} />
            </div>
          </div>

          <div className="relative mx-auto grid max-w-[30rem] grid-cols-1 gap-3 sm:grid-cols-2 sm:gap-2">
            <div
              aria-hidden="true"
              className="absolute bottom-[18%] left-1/2 top-[18%] hidden w-px -translate-x-1/2 bg-gray-100 sm:block"
            />
            <div className="flex justify-center overflow-x-auto overflow-y-visible sm:justify-end sm:pr-5">
              <div className="min-w-0 text-[17px] sm:text-[19px]">
                <Latex display math={SECTION_FIVE_MU_FORMULA} themeOverride={themeOverride} />
              </div>
            </div>
            <div className="flex justify-center overflow-x-auto overflow-y-visible sm:justify-start sm:pl-5">
              <div className="min-w-0 text-[17px] sm:text-[19px]">
                <Latex display math={SECTION_FIVE_ALPHA_FORMULA} themeOverride={themeOverride} />
              </div>
            </div>
          </div>
        </div>

        <p
          className="font-serif text-[17px] leading-[1.75] text-gray-700"
          style={{ textAlign: 'justify' }}
        >
          <InlineMathText themeOverride={themeOverride}>{SECTION_FIVE_INTRO_CLOSE}</InlineMathText>
        </p>
      </div>
    </div>
  );
}

function MetricSupport({ formula, detail, themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  if (!formula && !detail) return null;

  return (
    <div className="mt-2 space-y-1">
      {formula ? (
        <div className="flex justify-center text-[10.5px] leading-[1.35] text-gray-400">
          <div className="max-w-[24rem] overflow-x-auto overflow-y-hidden">
            <Latex math={formula} colorize={false} themeOverride={themeOverride} />
          </div>
        </div>
      ) : null}
      {detail ? (
        <div className="text-[10.5px] leading-[1.4] text-gray-400">
          <code className="font-mono">{detail}</code>
        </div>
      ) : null}
    </div>
  );
}

function ComparisonMetric({ label, value, valueClassName, valueStyle, formula, detail, themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  return (
    <div className="flex min-h-[160px] flex-col items-center justify-center px-4 py-[28px] text-center sm:px-6">
      <div
        className={[
          'font-serif text-[54px] leading-[0.95] tracking-[-0.03em] text-gray-900',
          valueClassName,
        ].filter(Boolean).join(' ')}
        style={valueStyle}
      >
        {value}
      </div>
      <div className="mt-2 text-[10px] font-semibold uppercase tracking-[0.22em] text-gray-600">
        {label}
      </div>
      <MetricSupport formula={formula} detail={detail} themeOverride={themeOverride} />
    </div>
  );
}

function SupportingMetric({ label, value, valueClassName, valueStyle, formula, detail, themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
  return (
    <div className="flex min-h-[126px] flex-col items-center justify-center px-4 py-[22px] text-center sm:px-5">
      <div
        className={[
          'font-serif text-[24px] leading-none tracking-[-0.02em] text-gray-900',
          valueClassName,
        ].filter(Boolean).join(' ')}
        style={valueStyle}
      >
        {value}
      </div>
      <div className="mt-2 text-[10px] font-semibold uppercase tracking-[0.22em] text-gray-600">
        {label}
      </div>
      <MetricSupport formula={formula} detail={detail} themeOverride={themeOverride} />
    </div>
  );
}

function EditorialComparisonSpread({ topMetrics, supportingMetrics, themeOverride = SECTION_FIVE_THEME_OVERRIDE }) {
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
            valueStyle={metric.valueStyle}
            formula={metric.formula}
            detail={metric.detail}
            themeOverride={themeOverride}
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
              valueStyle={metric.valueStyle}
              formula={metric.formula}
              detail={metric.detail}
              themeOverride={themeOverride}
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
  explorerThemeId,
}) {
  if (!componentCosts || !componentData) return null;

  const sectionFiveThemeCssVars = getExplorerThemeCssVariables(SECTION_FIVE_THEME_OVERRIDE);
  const { mu = 0, alpha = 0, mTotal = 0, perComponent = [] } = componentCosts;
  const totalCost = mu + alpha;
  const allLabelCount = componentData?.components?.reduce((sum, comp) => sum + comp.labels.length, 0) ?? 0;
  const denseTuples = Math.pow(dimensionN, allLabelCount);
  const multiplicationFactor = Math.max(numTerms - 1, 0);
  const denseTotalCost = multiplicationFactor * denseTuples + denseTuples;
  const totalSpeedup = totalCost > 0 ? (denseTotalCost / totalCost).toFixed(1) : '1.0';
  const savingsPct = denseTotalCost > 0 ? (((denseTotalCost - totalCost) / denseTotalCost) * 100).toFixed(1) : '0';
  const savingsPositive = Number(savingsPct) > 0;
  const { components = [] } = componentData;
  const denseExpansion = `(${numTerms} - 1) × ${denseTuples.toLocaleString()} + ${denseTuples.toLocaleString()} = ${denseTotalCost.toLocaleString()}`;
  const multiplicationExpansion = `(${numTerms} - 1) × ${mTotal.toLocaleString()} = ${mu.toLocaleString()}`;
  const alphaFactors = perComponent.map((comp) => comp.alpha_a.toLocaleString());
  const accumulationExpansion = alphaFactors.length > 1
    ? `${alphaFactors.join(' × ')} = ${alpha.toLocaleString()}`
    : alpha.toLocaleString();
  const TOP_COMPARISON_METRICS = [
    {
      label: 'Dense Cost',
      value: denseTotalCost.toLocaleString(),
      formula: String.raw`(k-1)\cdot n^{|L|} + n^{|L|}`,
      detail: denseExpansion,
    },
    {
      label: 'Symmetry-Aware Cost',
      value: totalCost.toLocaleString(),
      valueClassName: 'text-coral',
      formula: String.raw`\mu + \alpha`,
      detail: `${mu.toLocaleString()} + ${alpha.toLocaleString()} = ${totalCost.toLocaleString()}`,
    },
  ];
  const SUPPORTING_METRICS = [
    {
      label: 'Multiplication Cost (μ)',
      value: mu.toLocaleString(),
      formula: String.raw`\mu = (k-1)\prod_a M_a`,
      detail: multiplicationExpansion,
    },
    {
      label: 'Accumulation Cost (α)',
      value: alpha.toLocaleString(),
      formula: String.raw`\alpha = \prod_a \alpha_a`,
      detail: accumulationExpansion,
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
      valueStyle: savingsPositive
        ? {
            color: explorerThemeColor(SECTION_FIVE_THEME_OVERRIDE, 'quantity'),
          }
        : undefined,
    },
  ];

  return (
    <div className="space-y-8" style={sectionFiveThemeCssVars}>
      <ComponentRecap components={components} />

      <SectionFiveIntroBlock themeOverride={SECTION_FIVE_THEME_OVERRIDE} />

      <AggregationExplainer themeOverride={SECTION_FIVE_THEME_OVERRIDE} />

      <EditorialComparisonSpread
        topMetrics={TOP_COMPARISON_METRICS}
        supportingMetrics={SUPPORTING_METRICS}
        themeOverride={SECTION_FIVE_THEME_OVERRIDE}
      />
    </div>
  );
}

// Exported for tests — catches silent regressions in the pedagogy.
export { AGGREGATION_FORMULA, getAggregationLegend as AGGREGATION_LEGEND, getAggregationLeaves as AGGREGATION_LEAVES };
