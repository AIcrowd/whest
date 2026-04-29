// Main-page intuitive primer for the typed partition-counting method.
// Theorem-level statement lives in appendix/section6.ts; this component is a
// compressed reader-first explanation that introduces equality patterns,
// the typed-partition idea, and the |A_x̃/H| reach factor before readers
// encounter `partitionCount` as an active leaf in Section 5.

import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import { notationLatex } from '../lib/notationSystem.js';

const TITLE = 'When projection branches, count equality patterns';
const DECK = 'Partition counting is the exact compressed counter behind the general branching case.';

const BODY = [
  'A product orbit may contain many full assignments. When those assignments are projected to the output labels, they may reach one stored output representative or several. Enumerating every concrete assignment is correct but can be wasteful.',
  'The partition-counting method groups assignments by equality pattern: which label positions carry the same value. With heterogeneous dimensions, this is typed — only labels from the same domain can be placed in the same equality block.',
  'For a fixed typed equality pattern, the block-labeling factor counts how many product orbits live above that pattern. The map set $A_{\\tilde{x}}$ records which input equality block supplies each output coordinate as we move through the product orbit. Quotienting $A_{\\tilde{x}}$ by $H$ gives the number of stored output representatives reached.',
  'This is not an approximation. When both are feasible, the typed partition count and corrected brute-force orbit enumeration must give the same $\\alpha$.',
];

const FORMULA = String.raw`\alpha_a = \sum_{\tilde{x}\in P_{\mathrm{typed}}(L_a)/G_a} \frac{\prod_s (n_s)_{b_s(\tilde{x})}}{|\overline{G}_{\tilde{x}}|}\, |A_{\tilde{x}}/H_a|`;

const EXAMPLE_TITLE = 'Why branching happens';
const EXAMPLE_BODY = 'For $R[i,j] = \\sum_k T[i,j,k]$ with $T$ fully symmetric, the product orbit of three distinct values $\\{a,b,c\\}$ can project to the stored output representatives $\\{a,b\\}$, $\\{a,c\\}$, and $\\{b,c\\}$. One product orbit therefore contributes three accumulation updates.';
const EXAMPLE_FORMULA = String.raw`\alpha = \binom{n+1}{2}n = \frac{n^2(n+1)}{2}`;

export default function PartitionCountingExplainer() {
  return (
    <section
      id="partition-counting-explainer"
      className="mx-auto w-full max-w-[var(--prose-max)] rounded-xl border border-stone-200 bg-white px-6 py-6 shadow-sm md:px-8 md:py-7"
      aria-labelledby="partition-counting-title"
    >
      <h3
        id="partition-counting-title"
        className="font-heading text-[20px] font-semibold leading-tight text-gray-900"
      >
        {TITLE}
      </h3>
      <p className="mt-2 max-w-[70ch] font-serif text-[15px] italic leading-7 text-stone-600">
        {DECK}
      </p>

      <div className="mt-5 space-y-4 max-w-[78ch] font-serif text-[16px] leading-[1.75] text-gray-900">
        {BODY.map((paragraph, idx) => (
          <p key={idx}>
            <InlineMathText>{paragraph}</InlineMathText>
          </p>
        ))}
      </div>

      <div className="mt-6 overflow-x-auto">
        <Latex math={FORMULA} display />
      </div>

      <div className="mt-6 rounded-lg border border-stone-200 bg-gray-50 px-5 py-4 md:px-6 md:py-5">
        <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
          {EXAMPLE_TITLE}
        </div>
        <p className="mt-2 max-w-[70ch] font-serif text-[14px] leading-7 text-gray-800">
          <InlineMathText>{EXAMPLE_BODY}</InlineMathText>
        </p>
        <div className="mt-3 overflow-x-auto">
          <Latex math={EXAMPLE_FORMULA} display />
        </div>
      </div>

      <p className="mt-5 max-w-[78ch] text-[12.5px] leading-6 text-stone-600">
        Theorem-level statement and proof live in the appendix. The notation
        registry encodes <Latex math={notationLatex('typed_partition_pattern')} />,{' '}
        <Latex math={notationLatex('induced_maps')} />, and{' '}
        <Latex math={notationLatex('block_action_group')} /> for use across the
        main page and modal.
      </p>
    </section>
  );
}
