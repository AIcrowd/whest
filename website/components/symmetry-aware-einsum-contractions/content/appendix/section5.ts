import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section5 = {
  title: 'Why formal symmetry cannot replace pointwise symmetry in the cost',
  deck: 'The tempting shortcut is to apply Burnside to $G_{\\text{f}}$. That shortcut is invalid for accumulation because formal orbits may contain unequal summands.',
  slots: {
    intro: [
      p('The main page’s $\\alpha$ counts direct output-bin updates. Such a representative is valid only when every assignment in its orbit contributes the same indexed product. That condition holds for $G_{\\text{pt}}$ by construction. It need not hold for $G_{\\text{f}}$, because the domain-compatible dummy-label factor can rename bound variables only after the terms have already been summed.'),
      p('Therefore, Burnside over $G_{\\text{f}}$ answers a different question: it counts orbits of formal label renamings in the completed expression. It does not, in general, count the number of distinct summand computations or direct updates required during accumulation.'),
    ],
    mismatchLead: [
      p('For the selected preset, the difference is visible numerically. A naive formal count using $G_{\\text{f}}$ gives $\\alpha_{\\text{formal}}$. The pointwise accumulation count used by the engine is $\\alpha_{\\text{engine}}$. The mismatch is not a bug; it is exactly the gap between formal equivalence and pointwise equality.'),
    ],
    coincidentLead: [
      p('For the selected preset, the formal count and the pointwise count happen to agree at the current size. This numerical coincidence does not change the rule. The valid accumulation group is still $G_{\\text{pt}}$, because only $G_{\\text{pt}}$ is guaranteed to identify equal summands.'),
    ],
    noneLead: [
      p('For the selected preset, the formal group does not produce a distinct accumulation count to compare. The conceptual distinction remains the same: $G_{\\text{f}}$ describes post-summation label-renaming symmetry, while $G_{\\text{pt}}$ is the group used for accumulation.'),
    ],
    presetPickerLabel: [
      l('Presets with a visible mismatch:'),
    ],
    workedExampleLabelPrefix: [
      l('Worked example —'),
    ],
    workedExampleLead: [
      p('Then the expanded form of the einsum is:'),
    ],
    assignmentLead: [
      p('Now set'),
    ],
    bilinearOrbitLead: [
      p('Swapping the summed labels $k$ and $l$ sends $(0,1)$ to $(1,0)$, so $G_{\\text{f}}$ places these two assignments in the same formal orbit.'),
    ],
    bilinearFirstAssignment: [
      p('If $k = 0$ and $l = 1$, then'),
    ],
    bilinearSecondAssignment: [
      p('If $k = 1$ and $l = 0$, then'),
    ],
    bilinearNote: [
      p('Both terms contribute to the same output entry, and both must be accumulated. Formal symmetry has related them, but it has not made them equal.'),
    ],
    directIntro: [
      p('Let $T[0,0,0,1,2] = 1$ and $T[0,0,0,2,1] = 2$, and let every other entry of $T$ be $0$.'),
    ],
    directOrbitLead: [
      p('The transposition $(d\\,e)$ is allowed as a dummy relabeling in $G_{\\text{f}}$, so it sends $(c,d,e) = (0,1,2)$ to $(0,2,1)$. That move is formal, but it is not a pointwise symmetry of the summands.'),
    ],
    directFirstAssignment: [
      p('If $c = 0$, $d = 1$, and $e = 2$, then'),
    ],
    directSecondAssignment: [
      p('If $c = 0$, $d = 2$, and $e = 1$, then'),
    ],
    directNote: [
      p('Both terms contribute to the same output entry $R[0,0]$, but $1 \\neq 2$. Formal symmetry has related them, but it has not made them equal.'),
    ],
    mixedOrbitLead: [
      p('Swapping the summed labels $j$ and $k$ sends $(0,1)$ to $(1,0)$, so $G_{\\text{f}}$ places these two assignments in the same formal orbit.'),
    ],
    mixedFirstAssignment: [
      p('If $j = 0$ and $k = 1$, then'),
    ],
    mixedSecondAssignment: [
      p('If $j = 1$ and $k = 0$, then'),
    ],
    mixedNote: [
      p('Both terms contribute to the same output entry $R[0,0]$, but $6 \\neq 8$. Formal symmetry has related them, but it has not made them equal.'),
    ],
    genericOrbitLead: [
      p('A formal relabeling on the summed labels places these two assignments in the same formal orbit.'),
    ],
    genericAssignmentTemplate: [
      p('If {{assignment}}, then'),
    ],
    genericNoteTemplate: [
      p('Both assignments contribute to the same {{targetNoun}} ${{outputEntry}}$, but they remain different pointwise products and must be counted separately.'),
    ],
    rule: [
      p('Use $G_{\\text{pt}}$ for direct multiplication and accumulation. Use $G_{\\text{f}} = G_{\\mathrm{out}} \\times \\prod_d S(W_d)$ to describe formal symmetry of the completed expression. Do not use the domain-compatible dummy-label factor $\\prod_d S(W_d)$ to remove pre-summation summand computations.'),
    ],
  },
} satisfies SectionCopy;

export default section5;
