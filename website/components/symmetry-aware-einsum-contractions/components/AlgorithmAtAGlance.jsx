import ExplorerSectionCard, { AnchorLink } from './ExplorerSectionCard.jsx';
import Latex from './Latex.jsx';
import MentalFrameworkCode from './MentalFrameworkCode.jsx';
import InlineMathText from './InlineMathText.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import { notationLatex } from '../lib/notationSystem.js';
import { explorerThemeColor, getActiveExplorerThemeId } from '../lib/explorerTheme.js';
import { buildSection1ExampleView } from '../lib/section1ExampleView.js';

/**
 * Distill-style preamble for the symmetry-aware einsum explorer.
 *
 * Two-column top section:
 *   LEFT  — Einsum Contraction: prose + a 4-operand matrix-chain example
 *           rendered via color-coded KaTeX with explicit coral/slate label
 *           cues, followed by a "Where symmetry enters" transition block
 *           that names the pointwise group, product orbits, and output updates.
 *   RIGHT — The Mental Framework: two step panels (Multiplication · μ,
 *           Accumulation · α) above a single uninterrupted pseudocode block.
 *           Each code line carries a coloured left-rule tying it back to
 *           its step; the indentation of the inner loop stays visible.
 *
 * Below the two-column block, a single hand-off sentence leads into Act 1.
 * The concrete numeric payoff is left to the explorer itself — the reader
 * discovers it by picking a preset and watching the cost counts update.
 */

// Dense scaling: the brute-force cost of evaluating the chain above for
// uniform axis size n. Shown symbolically to motivate why symmetry matters.
const DENSE_SCALING = String.raw`\mathcal{O}(n^{5})`;

function ColorLegend({ freeLabelColor, summedLabelColor }) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-stone-700">
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" aria-hidden="true" style={{ backgroundColor: summedLabelColor }} />
        <span>
          <strong className="font-semibold text-stone-900">summed</strong> label (collapses)
        </span>
      </span>
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" aria-hidden="true" style={{ backgroundColor: freeLabelColor }} />
        <span>
          <strong className="font-semibold text-stone-900">free</strong> labels (stay on output)
        </span>
      </span>
    </div>
  );
}

const JUSTIFIED_PROSE_STYLE = { textAlign: 'justify' };

function EinsumIntroColumn({ example }) {
  const explorerThemeId = getActiveExplorerThemeId();
  const freeLabelColor = explorerThemeColor(explorerThemeId, 'hero');
  const summedLabelColor = explorerThemeColor(explorerThemeId, 'summedSide');
  const view = buildSection1ExampleView(example, {
    freeLabelColor: freeLabelColor,
    summedLabelColor: summedLabelColor,
  });
  if (!view) return null;
  const coloredVFreeNotation = String.raw`\textcolor{${freeLabelColor}}{${notationLatex('v_free')}}`;
  const coloredWSummedNotation = String.raw`\textcolor{${summedLabelColor}}{${notationLatex('w_summed')}}`;

  return (
    <div id="einsum-contraction" className="flex h-full flex-col scroll-mt-24">
      <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
        <AnchorLink anchorId="einsum-contraction" labelText="Einsum contraction">
          Einsum contraction
        </AnchorLink>
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        A tensor operation, written as one equation
      </h3>

      <p
        className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700"
        style={JUSTIFIED_PROSE_STYLE}
      >
        Every index label that appears on an input but not on the output is{' '}
        <strong className="font-semibold">summed over</strong>; labels on the output are{' '}
        <strong className="font-semibold">free</strong>. A dense implementation pays for every cell of
        the full input grid — so even modestly sized examples explode quickly.
      </p>

      <div className="mt-6 overflow-x-auto px-5 py-6">
        <div className="flex justify-center">
          <code className="rounded-md px-2 py-1 font-mono text-[16px] font-semibold tracking-[0.01em] text-stone-800">
            {view.exactEinsumText}
          </code>
        </div>
        <div className="mt-2.5 flex justify-center text-[19px]">
          <Latex display math={view.expandedEquationLatex} />
        </div>
        <p
          className="mx-auto mt-4 max-w-[46rem] text-[13px] leading-6 text-stone-600"
          style={JUSTIFIED_PROSE_STYLE}
        >
          <strong className="font-semibold text-stone-900">
            {view.operandCount} operand{view.operandCount === 1 ? '' : 's'}, {view.labelCount} label{view.labelCount === 1 ? '' : 's'}.
          </strong>{' '}
          <InlineMathText>
            {`The summed labels $${coloredWSummedNotation} = \\{${view.wSummedSummary}\\}$ collapse under $\\sum$; the free labels $${coloredVFreeNotation} = \\{${view.vFreeSummary}\\}$ survive as the axes of $R$. Declared symmetries: ${view.declaredSymmetrySummary}. Dense cost scales as $${DENSE_SCALING}$ for uniform axis size n.`}
          </InlineMathText>
        </p>
        <ColorLegend freeLabelColor={freeLabelColor} summedLabelColor={summedLabelColor} />
      </div>

      {/* Spacer: anchors the callout to the bottom when the right column is taller
          (so both columns end at the same y), with a mt-6 minimum gap when not. */}
      <div className="mt-6 flex-1" aria-hidden="true" />

      <div id="where-symmetry-enters" className="scroll-mt-24">
        <NarrativeCallout
          label={(
            <AnchorLink
              anchorId="where-symmetry-enters"
              labelText="Where symmetry enters"
              hashGlyphClassName="text-[12px] text-primary/70"
            >
              Where symmetry enters
            </AnchorLink>
          )}
          title="Not every product is distinct"
          tone="preamble"
        >
          <p
            className="text-[14px] leading-7 text-foreground"
            style={JUSTIFIED_PROSE_STYLE}
          >
            If several operand occurrences are identical, or if an input tensor has a
            declared equality symmetry such as <Latex math={String.raw`A_{ij} = A_{ji}`} />,
            some relabelings are accepted as preserving the scalar summand for this
            model. Those accepted relabelings form a detected pointwise group{' '}
            <strong className="font-semibold">G</strong>.
            The explorer evaluates one representative product for each orbit of full label
            assignments, while still updating every output bin touched by that orbit.
          </p>
          <p
            className="text-[13px] italic leading-6 text-stone-600"
            style={JUSTIFIED_PROSE_STYLE}
          >
            Thus the computation is not reduced by simply dividing by <Latex math="|G|" />.
            Fixed loci, diagonal assignments, and output projections are counted explicitly
            through Burnside formulas or exact orbit enumeration.
          </p>
        </NarrativeCallout>
      </div>
    </div>
  );
}

function MentalFrameworkColumn({ example }) {
  return (
    <div id="mental-framework" className="flex h-full flex-col scroll-mt-24">
      <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
        <AnchorLink anchorId="mental-framework" labelText="The mental framework">
          The mental framework
        </AnchorLink>
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        The same two loops — dense or symmetric
      </h3>

      <p
        className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700"
        style={JUSTIFIED_PROSE_STYLE}
      >
        Every contraction has the same shape. Symmetry only changes the content of three things:{' '}
        <code className="explorer-inline-code">RepSet</code>, <code className="explorer-inline-code">Outs(rep)</code>, and{' '}
        <code className="explorer-inline-code">coeff(rep, out)</code>. The rest of the explorer is about
        counting them without enumerating the full grid.
      </p>

      <div className="mt-6 flex flex-1 flex-col">
        <MentalFrameworkCode example={example} />
      </div>
    </div>
  );
}

export default function AlgorithmAtAGlance({ example }) {
  return (
    <section id="algorithm-at-a-glance" aria-labelledby="algorithm-at-a-glance-title" className="mb-10 scroll-mt-24">
      <ExplorerSectionCard
        eyebrow={
          <AnchorLink anchorId="algorithm-at-a-glance" labelText="Einsum at a Glance">
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
              Einsum at a Glance
            </span>
          </AnchorLink>
        }
        title={<span id="algorithm-at-a-glance-title">What this explorer is counting</span>}
        description="The direct indexed computation, the representative products, and the output-bin updates it induces."
        className="border-gray-200 bg-white"
        contentClassName="pt-6"
      >
        {/* Two-column top: einsum notation (L) ↔ mental framework code (R).
            items-stretch makes both columns reach the same y by design:
            whichever side is naturally shorter grows a spacer to fill. On the
            left the "Where symmetry enters" callout sticks to the bottom; on
            the right the MentalFrameworkCode figure stretches so its Counting
            Convention band anchors to the bottom. */}
        <div className="editorial-two-col-divider-lg grid items-stretch gap-8 lg:grid-cols-2 lg:gap-10">
          <EinsumIntroColumn example={example} />
          <MentalFrameworkColumn example={example} />
        </div>

        <p
          className="mt-10 border-t border-stone-200 pt-8 font-serif text-[17px] leading-[1.75] text-gray-700"
          style={JUSTIFIED_PROSE_STYLE}
        >
          The rest of this page shows how the explorer moves from a declared contraction
          to accepted pointwise relabelings, then from those relabelings to the two
          quantities that drive the direct cost: product orbits{' '}
          <Latex math={String.raw`\mu`} /> and output-bin updates{' '}
          <Latex math={String.raw`\alpha`} />. Start with{' '}
          <strong className="font-semibold">Section 1</strong> below to pick or build a
          contraction.
        </p>
      </ExplorerSectionCard>
    </section>
  );
}
