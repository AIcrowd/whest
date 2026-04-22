import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section5 = {
  title: 'Assemble the Direct Cost Model',
  deck: 'What is the final cost of the symmetry-aware direct computation?',
  slots: {
    appendixNoteTitle: [
      l('Appendix note'),
      p('Is this the full symmetry of the final expression?'),
    ],
    appendixNoteBody: [
      p('The cost above uses $G_{\\mathrm{pt}}$ for accumulation. The fully summed expression can have a larger label-renaming formal symmetry, $G_{\\mathrm{f}} = G_{\\mathrm{out}} \\times S(W_{\\mathrm{summed}})$. That larger group cannot reduce the accumulation count, but its visible output part $G_{\\mathrm{out}}$ can still reduce output storage.'),
    ],
  },
} satisfies SectionCopy;

export default section5;
