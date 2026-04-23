import ExplorerSectionCard, { AnchorLink } from './ExplorerSectionCard.jsx';
import EditorialCallout from './EditorialCallout.jsx';
import Latex from './Latex.jsx';
import MentalFrameworkCode from './MentalFrameworkCode.jsx';
import InlineMathText from './InlineMathText.jsx';
import SectionReferenceLink from './SectionReferenceLink.jsx';
import { default as renderProseBlocks } from '../content/renderProseBlocks.jsx';
import mainPreamble from '../content/main/preamble.ts';
import { notationColor, notationLatex } from '../lib/notationSystem.js';
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

function renderSingleProseBlock(blocks = []) {
  return renderProseBlocks(blocks)[0] ?? null;
}

function ColorLegend() {
  const landingFreeLabelColor = notationColor('v_free');
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-stone-700">
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" aria-hidden="true" style={{ backgroundColor: notationColor('w_summed') }} />
        <span>
          <strong className="font-semibold text-stone-900">summed</strong> label (collapses)
        </span>
      </span>
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" aria-hidden="true" style={{ backgroundColor: landingFreeLabelColor }} />
        <span>
          <strong className="font-semibold text-stone-900">free</strong> labels (stay on output)
        </span>
      </span>
    </div>
  );
}

const JUSTIFIED_PROSE_STYLE = { textAlign: 'justify' };

function EinsumIntroColumn({ example }) {
  const view = buildSection1ExampleView(example);
  if (!view) return null;
  const coloredVFreeNotation = String.raw`\textcolor{${notationColor('v_free')}}{${notationLatex('v_free')}}`;

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

      <p
        className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700"
        style={JUSTIFIED_PROSE_STYLE}
      >
        {renderSingleProseBlock(mainPreamble.slots.einsumIntroBeforeSummed)}
        <strong className="font-semibold">summed over</strong>
        {renderSingleProseBlock(mainPreamble.slots.einsumIntroBetweenSummedAndFree)}
        <strong className="font-semibold">free</strong>
        {renderSingleProseBlock(mainPreamble.slots.einsumIntroAfterFree)}
      </p>

      <div className="mt-6 overflow-x-auto rounded-2xl border border-stone-200 bg-white px-5 py-6">
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
            {`The summed labels $${notationLatex('w_summed')} = \\{${view.wSummedSummary}\\}$ collapse under $\\sum$; the free labels $${coloredVFreeNotation} = \\{${view.vFreeSummary}\\}$ survive as the axes of $R$. Declared symmetries: ${view.declaredSymmetrySummary}. The dense direct grid has $${view.denseGridScalingLatex}$ assignments before symmetry is used.`}
          </InlineMathText>
        </p>
      </div>

      <ColorLegend />

      {/* Spacer: anchors the callout to the bottom when the right column is taller
          (so both columns end at the same y), with a mt-6 minimum gap when not. */}
      <div className="mt-6 flex-1" aria-hidden="true" />

      <EditorialCallout
        id="where-symmetry-enters"
        className="scroll-mt-24"
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
        bodyClassName="mt-2 text-[14px] leading-7 text-foreground"
        footer={(
          <>
            {renderSingleProseBlock(mainPreamble.slots.calloutFooter)}
          </>
        )}
      >
        <>
          {renderSingleProseBlock(mainPreamble.slots.calloutBodyBeforeGroup)}
          <strong className="font-semibold">
            group <Latex math="G" />
          </strong>
          {renderSingleProseBlock(mainPreamble.slots.calloutBodyBetweenGroupAndOrbits)}
          <em>orbits</em>
          {renderSingleProseBlock(mainPreamble.slots.calloutBodyAfterOrbits)}
        </>
      </EditorialCallout>
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
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroBeforeRepSet)}
        <code className="explorer-inline-code">RepSet</code>
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroBetweenRepSetAndOuts)}
        <code className="explorer-inline-code">Outs(rep)</code>
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroBetweenOutsAndCoeff)}
        <code className="explorer-inline-code">coeff(rep, out)</code>
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroAfterCoeff)}
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
        title={<span id="algorithm-at-a-glance-title">{mainPreamble.title}</span>}
        description={mainPreamble.deck}
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
          {renderSingleProseBlock(mainPreamble.slots.handoffBeforeSectionLink)}
          <SectionReferenceLink href="#setup">Section 1</SectionReferenceLink>
          {renderSingleProseBlock(mainPreamble.slots.handoffAfterSectionLink)}
        </p>
      </ExplorerSectionCard>
    </section>
  );
}
