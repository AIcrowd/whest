import ExplorerSectionCard from './ExplorerSectionCard.jsx';
import Latex from './Latex.jsx';
import MentalFrameworkCode from './MentalFrameworkCode.jsx';

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

// Theme tokens (from global.css) exposed for KaTeX \textcolor, which needs hex.
const CORAL = '#F0524D';           // --primary
const MUTED_FOREGROUND = '#5D5F60'; // --muted-foreground (gray-600)

// A 4-operand matrix chain — big enough to feel expensive, compact enough to
// fit in the left column. 5 labels total: i, m are free; j, k, l are summed.
const CHAIN_FORMULA = String.raw`R[\textcolor{${MUTED_FOREGROUND}}{i},\textcolor{${MUTED_FOREGROUND}}{m}] \;=\; \sum_{\textcolor{${CORAL}}{j},\textcolor{${CORAL}}{k},\textcolor{${CORAL}}{l}}\; A[\textcolor{${MUTED_FOREGROUND}}{i},\textcolor{${CORAL}}{j}]\,\cdot\,B[\textcolor{${CORAL}}{j},\textcolor{${CORAL}}{k}]\,\cdot\,C[\textcolor{${CORAL}}{k},\textcolor{${CORAL}}{l}]\,\cdot\,D[\textcolor{${CORAL}}{l},\textcolor{${MUTED_FOREGROUND}}{m}]`;

// Dense scaling: the brute-force cost of evaluating the chain above for
// uniform axis size n. Shown symbolically to motivate why symmetry matters.
const DENSE_SCALING = String.raw`\mathcal{O}(n^{5})`;

function ColorLegend() {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-stone-700">
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full bg-primary" aria-hidden="true" />
        <span>
          <strong className="font-semibold text-stone-900">summed</strong> label (collapses)
        </span>
      </span>
      <span className="inline-flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full bg-stone-500" aria-hidden="true" />
        <span>
          <strong className="font-semibold text-stone-900">free</strong> labels (stay on output)
        </span>
      </span>
    </div>
  );
}

function EinsumIntroColumn() {
  return (
    <div className="flex h-full flex-col">
      <span className="font-heading text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
        Einsum contraction
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        A tensor operation, written as one formula
      </h3>

      <p className="mt-3 text-[15px] leading-7 text-foreground">
        Every index label that appears on an input but not on the output is{' '}
        <strong className="font-semibold">summed over</strong>; labels on the output are{' '}
        <strong className="font-semibold">free</strong>. A dense implementation pays for every cell of
        the full input grid — so even modestly sized examples explode quickly.
      </p>

      <div className="mt-6 overflow-x-auto rounded-2xl border border-stone-200 bg-white px-5 py-6">
        <p className="text-center font-heading text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">
          Example — a 4‑operand matrix chain
        </p>
        <div className="mt-3 flex justify-center text-[19px]">
          <Latex display math={CHAIN_FORMULA} />
        </div>
        <p className="mt-4 text-center text-[13px] leading-6 text-stone-600">
          <strong className="font-semibold text-stone-900">4 operands, 5 labels.</strong> The summed
          labels{' '}
          <span className="font-mono font-semibold" style={{ color: CORAL }}>j, k, l</span>{' '}
          collapse under <Latex math={String.raw`\sum`} />; the free labels{' '}
          <span className="font-mono font-semibold" style={{ color: MUTED_FOREGROUND }}>i, m</span>{' '}
          survive as the axes of R. Dense cost scales as <Latex math={DENSE_SCALING} /> for uniform
          axis size n.
        </p>
      </div>

      <ColorLegend />

      {/* Spacer: anchors the callout to the bottom when the right column is taller
          (so both columns end at the same y), with a mt-6 minimum gap when not. */}
      <div className="mt-6 flex-1" aria-hidden="true" />

      <div className="rounded-2xl border border-primary/20 bg-accent/40 px-5 py-5">
        <span className="font-heading text-[11px] font-semibold uppercase tracking-[0.18em] text-primary">
          Where symmetry enters
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
    <div className="flex h-full flex-col">
      <span className="font-heading text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
        The mental framework
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        The same two loops — dense or symmetric
      </h3>

      <p className="mt-3 text-[15px] leading-7 text-foreground">
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

export default function AlgorithmAtAGlance() {
  return (
    <section aria-labelledby="algorithm-at-a-glance-title" className="mb-10 scroll-mt-24">
      <ExplorerSectionCard
        eyebrow="Algorithm at a Glance"
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
          <EinsumIntroColumn />
          <MentalFrameworkColumn />
        </div>

        <p className="mt-10 border-t border-stone-200 pt-8 text-[15px] leading-7 text-foreground">
          The rest of this page shows how the explorer detects the symmetry group{' '}
          <Latex math="G" /> from a contraction and computes{' '}
          <Latex math={String.raw`\mu`} /> and <Latex math={String.raw`\alpha`} /> automatically.
          Start with <strong className="font-semibold">Act 1</strong> below to pick or build a
          contraction.
        </p>
      </ExplorerSectionCard>
    </section>
  );
}
