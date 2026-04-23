import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section5 = {
  title: 'Assemble the Direct Cost Model',
  deck: 'What is the final direct-event cost of the symmetry-aware computation?',
  slots: {
    appendixNoteTitle: [
      l('Appendix note'),
      p('Is this the full symmetry of the final expression?'),
    ],
    appendixNoteBody: [
      p('The direct cost above uses $G_{\\mathrm{pt}}$ because accumulation happens before the summed labels have disappeared. The completed expression can have a larger formal label-renaming group, $G_{\\mathrm{f}} = G_{\\mathrm{out}} \\times \\prod_d S(W_d)$, where each $W_d$ is a same-domain block of summed labels. That larger group explains expression-level equality after summation. It must not replace $G_{\\mathrm{pt}}$ for accumulation, although its visible output part $G_{\\mathrm{out}}$ can still reduce output storage.'),
    ],
  },
} satisfies SectionCopy;

export default section5;
