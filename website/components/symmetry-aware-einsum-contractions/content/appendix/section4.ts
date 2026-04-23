import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section4 = {
  title: 'The formal group is a product, not a cost group',
  deck: 'The completed expression combines the visible output action with domain-compatible dummy-label renamings.',
  slots: {
    intro: [
      p('We can now name the formal label-renaming group considered in this appendix. It has two independent factors: the visible output action $G_{\\mathrm{out}}$, and the domain-compatible dummy-label renaming group $\\prod_d S(W_d)$. Their direct product is'),
      p('An element of $G_{\\text{f}} = G_{\\mathrm{out}} \\times \\prod_d S(W_d)$ acts by applying an output relabeling on $V$ and an independent same-domain dummy renaming on $W$. This is a symmetry of the completed formal expression. It is not automatically a symmetry of the accumulation process.'),
    ],
    takeaway: [
      p('$G_{\\text{f}}$ explains label-renaming symmetry after summation. It should not be substituted for $G_{\\text{pt}}$ in the direct accumulation cost.'),
    ],
    constructionTitle: [
      p('Formal-group construction'),
    ],
    constructionNote: [
      p('The construction above enumerates the output factor and the same-domain dummy-renaming factors for the selected preset.'),
    ],
  },
} satisfies SectionCopy;

export default section4;
