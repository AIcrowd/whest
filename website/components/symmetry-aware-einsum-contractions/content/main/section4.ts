import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section4 = {
  title: 'Count Product Orbits and Output Projections',
  deck: 'How does the pointwise group become multiplication and accumulation cost?',
  slots: {
    intro: [
      p('The group $G_{\\mathrm{pt}}$ acts on the full assignment grid $X$, not only on output labels. The number of representative products is the orbit count $M = |X/G_{\\mathrm{pt}}|$. In components this becomes $M = \\prod_a M_a$, with each $M_a$ computed by size-aware Burnside or exact orbit enumeration.'),
      p('Accumulation is the part that is easy to over-compress. For a product orbit $O \\subset X$, one representative product may touch one output bin, several output bins, or the single scalar output. The direct update count is therefore $\\alpha = \\sum_{O \\in X/G_{\\mathrm{pt}}} |\\pi_{V_{\\mathrm{free}}}(O)|$, where $\\pi_{V_{\\mathrm{free}}}$ projects an assignment orbit onto visible/output labels.'),
      p('This is why output symmetry and accumulation symmetry are different. A free-label symmetry can reduce the number of products while still requiring separate updates to each output bin touched by the orbit.'),
    ],
    produces: [
      p('Per-component product counts $M_a$ and update counts $\\alpha_a$, then global $M=\\prod_a M_a$ and $\\alpha=\\prod_a\\alpha_a$ when the support-connected components factor independently.'),
    ],
  },
} satisfies SectionCopy;

export default section4;
