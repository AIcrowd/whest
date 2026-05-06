import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section4 = {
  title: 'The completed-expression formal group',
  deck: 'After summation, the expression may have more label-renaming symmetry than the direct evaluator may use.',
  slots: {
    intro: [
      p('The completed expression combines two kinds of label-renaming symmetry: the output representative action $H$ and the domain-compatible dummy-label renaming group $\\prod_d S(W_d)$. Their product is the formal completed-expression group'),
      p('An element of $G_{\\text{f}} = H \\times \\prod_d S(W_d)$ applies an output relabeling on $V$ and an independent same-domain dummy renaming on $W$. This is a symmetry of the completed formal expression. It is not automatically a symmetry of the pre-summation products or of the accumulation relation counted on the main page.'),
    ],
    takeaway: [
      p('$G_{\\text{f}}$ explains expression-level label-renaming symmetry after summation. The main cost uses $G_{\\text{pt}}$ on products and $H$ on stored outputs; it does not use the dummy-label factor to merge pre-summation events.'),
    ],
    constructionTitle: [
      p('Formal-group construction'),
    ],
    presetPickerLabel: [
      p('Interesting presets:'),
    ],
    constructionNote: [
      p('The construction above enumerates the output factor and the same-domain dummy-renaming factors for the selected preset.'),
    ],
  },
} satisfies SectionCopy;

export default section4;
