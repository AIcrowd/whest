import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const c = (text: string) => ({ kind: 'caption', text } as const);

const section6 = {
  title: 'How the unified accumulation count uses output symmetry',
  deck: 'Output symmetry is not a separate storage-time post-processing step. The output representative action $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ is derived from the same detected pointwise group and shapes how product orbits update stored output representatives.',
  slots: {
    intro: [
      p('The detected pointwise group $G_{\\text{pt}}$ supplies two quotients to the direct evaluator. The first is the product-orbit quotient $X/G_{\\text{pt}}$, which fixes the count of representative products $M$. The second is the output representative action $H$, which fixes the count of stored output representatives $|Y/H|$.'),
      p('There is no separate $G_{\\mathrm{out}}$ as a standalone storage-side concept. The output-side action is induced from $G_{\\text{pt}}$ by the rule $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$: keep every $g \\in G_{\\text{pt}}$ that maps the visible-label set $V$ into itself, then restrict that $g$ to $V$ using local coordinates. The accumulation count $\\alpha = \\#\\{(O, Q) \\in X/G_{\\text{pt}} \\times Y/H : \\pi_V(O) \\cap Q \\neq \\varnothing\\}$ is computed once with $H$ already in place.'),
      p('When projection is functional — every $g \\in G_{\\text{pt}}$ preserves $V$ as a set — each product orbit reaches exactly one stored output representative, and $\\alpha = M$. When projection branches, the typed partition count or corrected brute force counts the pairs exactly. The dummy-label renamings of the formal group $G_{\\text{f}}$ do not enter; they describe symmetries of the completed expression after summation, not pre-summation pairings.'),
    ],
    footer: [
      p('Symmetry-aware storage in the older sense — independently quotienting the output by some external group — is subsumed by this construction. Any storage layout coarser than $Y/H$ would reduce the number of stored representatives, but the accumulation events are already counted relative to $Y/H$.'),
      p('Algebraic restructuring such as factoring $R = v v^\\top$, contraction re-ordering, and memory-traffic optimization sit outside this model and require their own cost definitions.'),
    ],
  },
} satisfies SectionCopy;

export default section6;
