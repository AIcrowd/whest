import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section1 = {
  title: 'Specify the Contraction',
  deck: 'What exact indexed computation is being counted?',
  slots: {
    intro: [
      p('The first step is to fix the mathematical object, not to detect symmetry. We specify the ordered operands, the subscript carried by each operand, the output labels, the label sizes, and the equality symmetries declared on individual inputs.'),
      p('This produces a normalized contraction instance: an ordered list of tensor occurrences, one ordered slot list per occurrence, a visible label set, a summed label set, and the declared slot actions that are allowed inside each operand. Later sections may use those declarations, but they do not infer additional tensor identities from numerical values.'),
    ],
    produces: [
      p('A normalized contraction instance with explicit operands, slots, output labels, label sizes, and declared equality symmetries.'),
    ],
  },
} satisfies SectionCopy;

export default section1;
