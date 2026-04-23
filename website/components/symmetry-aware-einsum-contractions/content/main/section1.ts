import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section1 = {
  title: 'Specify the Contraction',
  deck: 'What exact indexed computation is being counted?',
  slots: {
    intro: [
      p('The first step is to fix the mathematical object, not to detect symmetry. We specify the ordered operand occurrences, one subscript for each occurrence, the output labels, the label sizes, and the equality symmetries declared on individual inputs.'),
      p('This produces a normalized contraction instance: a label set $L = V_{\\mathrm{free}} \\sqcup W_{\\mathrm{summed}}$, where $V_{\\mathrm{free}}$ are visible/output labels and $W_{\\mathrm{summed}}$ are summed labels; an assignment grid $X = \\prod_{\\ell\\in L}[n_\\ell]$; and declared slot actions for each operand. Reusing the same operand name means the same tensor object appears more than once. Distinct names are treated as distinct objects even if their shapes or numerical values happen to match.'),
    ],
    produces: [
      p('A normalized direct-index contraction instance with operands, slots, output labels, label domains, and declared equality symmetries.'),
    ],
  },
} satisfies SectionCopy;

export default section1;
