import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section5 = {
  title: 'Why formal symmetry cannot replace pointwise symmetry in the cost',
  deck: 'The tempting shortcut is to apply Burnside to G_f. That shortcut is invalid for accumulation because formal orbits may contain unequal summands.',
  slots: {
    intro: [
      p('The main page’s α counts accumulation representatives. Such a representative is valid only when every assignment in its orbit contributes the same indexed product. That condition holds for $G_{\\mathrm{pt}}$ by construction. It need not hold for $G_{\\mathrm{f}}$, because the $S(W_{\\mathrm{summed}})$ factor can rename dummy variables only after the terms have already been summed.'),
      p('Therefore, Burnside over $G_{\\mathrm{f}}$ answers a different question: it counts orbits of formal label renamings in the completed expression. It does not, in general, count the number of distinct summand computations required by the accumulation.'),
    ],
    mismatchLead: [
      p('For the selected preset, the difference is visible numerically. A naive formal count using $G_{\\mathrm{f}}$ gives $\\alpha_{\\text{formal}}$. The pointwise accumulation count used by the engine is $\\alpha_{\\text{engine}}$. The mismatch is not a bug; it is exactly the gap between formal equivalence and pointwise equality.'),
    ],
    coincidentLead: [
      p('For the selected preset, the formal count and the pointwise count happen to agree at the current size. This numerical coincidence does not change the rule. The valid accumulation group is still $G_{\\mathrm{pt}}$, because only $G_{\\mathrm{pt}}$ is guaranteed to identify equal summands.'),
    ],
    noneLead: [
      p('For the selected preset, the formal group does not produce a distinct accumulation count to compare. The conceptual distinction remains the same: $G_{\\mathrm{f}}$ describes post-summation label-renaming symmetry, while $G_{\\mathrm{pt}}$ is the group used for accumulation.'),
    ],
    workedExampleTitle: [
      l('Worked example — bilinear trace'),
      p('Then the expanded form of the einsum is:'),
    ],
    rule: [
      p('Use $G_{\\mathrm{pt}}$ for accumulation. Use $G_{\\mathrm{f}}$ to describe formal symmetry of the completed expression. Do not use the dummy-label factor $S(W_{\\mathrm{summed}})$ to remove summand computations.'),
    ],
  },
} satisfies SectionCopy;

export default section5;
