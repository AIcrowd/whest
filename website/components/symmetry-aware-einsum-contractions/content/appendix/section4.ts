import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section4 = {
  title: 'The formal group is a product, not a cost group',
  deck: 'The completed expression combines the visible output action with dummy-label renamings.',
  slots: {
    intro: [
      p('We can now name the label-renaming formal group considered in this appendix. It has two independent factors: the visible output action $G_{\\mathrm{out}}$, and the dummy-label renaming group $S(W_{\\mathrm{summed}})$. Their direct product is'),
      p('An element of $G_{\\text{f}}$ acts by applying an output relabeling on $V_{\\mathrm{free}}$ and an independent dummy renaming on $W_{\\mathrm{summed}}$. This is a symmetry of the completed formal expression. It is not automatically a symmetry of the accumulation process.'),
    ],
    takeaway: [
      p('$G_{\\text{f}}$ explains label-renaming symmetry after summation. It should not be substituted for $G_{\\text{pt}}$ in the accumulation cost.'),
    ],
    constructionTitle: [
      p('Formal-group construction'),
    ],
    constructionNote: [
      p('The construction above enumerates the two factors and their product for the selected preset.'),
    ],
  },
} satisfies SectionCopy;

export default section4;
