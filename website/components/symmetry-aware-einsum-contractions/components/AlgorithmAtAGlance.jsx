import ExplorerSectionCard, { AnchorLink } from './ExplorerSectionCard.jsx';
import Latex from './Latex.jsx';
import MentalFrameworkCode from './MentalFrameworkCode.jsx';
import InlineMathText from './InlineMathText.jsx';
import { notationColor, notationLatex } from '../lib/notationSystem.js';
import { buildSection1ExampleView, LANDING_FREE_LABEL_COLOR } from '../lib/section1ExampleView.js';

/**
 * Distill-style preamble for the symmetry-aware einsum explorer.
 *
 * Two-column top section:
 *   LEFT  — Einsum Contraction: prose + a 4-operand matrix-chain example
 *           rendered via color-coded KaTeX (summed labels coral, free labels
 *           stone), followed by a "Where symmetry enters" transition block
 *           that names G, orbits, and previews μ / α.
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

function ColorLegend() {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-stone-700">
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" aria-hidden="true" style={{ backgroundColor: notationColor('w_summed') }} />
        <span>
          <strong className="font-semibold text-stone-900">summed</strong> label (collapses)
        </span>
      </span>
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" aria-hidden="true" style={{ backgroundColor: LANDING_FREE_LABEL_COLOR }} />
        <span>
          <strong className="font-semibold text-stone-900">free</strong> labels (stay on output)
        </span>
      </span>
    </div>
  );
}

function EinsumIntroColumn({ example }) {
  const view = buildSection1ExampleView(example);
  if (!view) return null;
  const coloredVFreeNotation = String.raw`\textcolor{${LANDING_FREE_LABEL_COLOR}}{${notationLatex('v_free')}}`;

  return (
    <div id="einsum-contraction" className="flex h-full flex-col scroll-mt-24">
      <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
        <AnchorLink anchorId="einsum-contraction" labelText="Einsum contraction">
          Einsum contraction
        </AnchorLink>
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        A tensor operation, written as one formula
      </h3>

      <p className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700">
        Every index label that appears on an input but not on the output is{' '}
        <strong className="font-semibold">summed over</strong>; labels on the output are{' '}
        <strong className="font-semibold">free</strong>. A dense implementation pays for every cell of
        the full input grid — so even modestly sized examples explode quickly.
      </p>

      <div className="mt-6 overflow-x-auto rounded-2xl border border-stone-200 bg-white px-5 py-6">
        <p className="text-center font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
          Example — the selected contraction
        </p>
        <div className="mt-4 flex justify-center">
          <code className="rounded-md px-2 py-1 font-mono text-[16px] font-semibold tracking-[0.01em] text-stone-800">
            {view.exactEinsumText}
          </code>
        </div>
        <div className="mt-3 flex justify-center text-[19px]">
          <Latex display math={view.expandedEquationLatex} />
        </div>
        <p className="mt-4 text-center text-[13px] leading-6 text-stone-600">
          <strong className="font-semibold text-stone-900">
            {view.operandCount} operand{view.operandCount === 1 ? '' : 's'}, {view.labelCount} label{view.labelCount === 1 ? '' : 's'}.
          </strong>{' '}
          <InlineMathText>
            {`The summed labels $${notationLatex('w_summed')} = \\{${view.wSummedSummary}\\}$ collapse under $\\sum$; the free labels $${coloredVFreeNotation} = \\{${view.vFreeSummary}\\}$ survive as the axes of $R$. Declared symmetries: ${view.declaredSymmetrySummary}. Dense cost scales as $${DENSE_SCALING}$ for uniform axis size n.`}
          </InlineMathText>
        </p>
      </div>

      <ColorLegend />

      {/* Spacer: anchors the callout to the bottom when the right column is taller
          (so both columns end at the same y), with a mt-6 minimum gap when not. */}
      <div className="mt-6 flex-1" aria-hidden="true" />

      <div id="where-symmetry-enters" className="rounded-2xl border border-primary/20 bg-accent/40 px-5 py-5 scroll-mt-24">
        <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
          <AnchorLink
            anchorId="where-symmetry-enters"
            labelText="Where symmetry enters"
            hashGlyphClassName="text-[12px] text-primary/70"
          >
            Where symmetry enters
          </AnchorLink>
        </span>
        <h4 className="mt-1 font-heading text-base font-semibold text-foreground">
          Not every product is distinct
        </h4>
        <p className="mt-2 text-[14px] leading-7 text-foreground">
          If several operands are identical or individually symmetric
          (e.g. <Latex math={String.raw`A_{ij} = A_{ji}`} />), the formula is invariant under certain
          permutations of the labels. Those permutations form a{' '}
          <strong className="font-semibold">group G</strong>, and whole{' '}
          <em>orbits</em> of products collapse to a single distinct computation — the dense{' '}
          <Latex math={DENSE_SCALING} /> drops to <Latex math={String.raw`n^{5}/|G|`} /> in the best
          case (a free action), and to a Burnside count <Latex math={String.raw`(1/|G|)\sum_g |\mathrm{Fix}(g)|`} /> in general.
        </p>
        <p className="mt-3 text-[13px] italic leading-6 text-stone-600">
          The explorer finds G automatically, then counts the distinct products (<Latex math="\mu" />)
          and distinct output‑bin updates (<Latex math="\alpha" />) — the two numbers driving the
          code on the right.
        </p>
      </div>
    </div>
  );
}

function MentalFrameworkColumn() {
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

      <p className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700">
        Every contraction has the same shape. Symmetry only changes the content of three things:{' '}
        <code className="font-mono">RepSet</code>, <code className="font-mono">Outs(rep)</code>, and{' '}
        <code className="font-mono">coeff(rep, out)</code>. The rest of the explorer is about
        counting them without enumerating the full grid.
      </p>

      <div className="mt-6 flex flex-1 flex-col">
        <MentalFrameworkCode />
      </div>
    </div>
  );
}

export default function AlgorithmAtAGlance({ example }) {
  return (
    <section id="algorithm-at-a-glance" aria-labelledby="algorithm-at-a-glance-title" className="mb-10 scroll-mt-24">
      <ExplorerSectionCard
        eyebrow={
          <AnchorLink anchorId="algorithm-at-a-glance" labelText="Algorithm at a Glance">
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
              Algorithm at a Glance
            </span>
          </AnchorLink>
        }
        title={<span id="algorithm-at-a-glance-title">What this explorer is counting</span>}
        description="The whole algorithm in ten lines, paired with the notation it operates on."
        className="border-gray-200 bg-white"
        contentClassName="pt-6"
      >
        {/* Two-column top: einsum notation (L) ↔ mental framework code (R).
            items-stretch makes both columns reach the same y by design:
            whichever side is naturally shorter grows a spacer to fill. On the
            left the "Where symmetry enters" callout sticks to the bottom; on
            the right the MentalFrameworkCode figure stretches so its Counting
            Convention band anchors to the bottom. */}
        <div className="grid items-stretch gap-8 lg:grid-cols-2 lg:gap-10">
          <EinsumIntroColumn example={example} />
          <MentalFrameworkColumn />
        </div>

        <p className="mt-10 border-t border-stone-200 pt-8 font-serif text-[17px] leading-[1.75] text-gray-700">
          The rest of this page shows how the explorer detects the symmetry group{' '}
          <Latex math="G" /> from a contraction and computes{' '}
          <Latex math={String.raw`\mu`} /> and <Latex math={String.raw`\alpha`} /> automatically.
          Start with <strong className="font-semibold">Section 1</strong> below to pick or build a
          contraction.
        </p>
      </ExplorerSectionCard>
    </section>
  );
}
