import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section5 = {
  title: 'Why formal symmetry cannot replace pointwise symmetry in the cost',
  deck: 'Burnside over $G_{\\text{f}}$ counts formal expression orbits, not product-orbit-to-output-representative updates.',
  slots: {
    intro: [
      p('The main alpha counts accumulation updates from product-orbit representatives into stored output representatives — pairs $(O, Q) \\in X/G_{\\text{pt}} \\times Y/H$ where $\\pi_V(O) \\cap Q \\neq \\varnothing$ and $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$. Such a representative is valid only when every assignment in its orbit contributes the same indexed product. That condition holds for $G_{\\text{pt}}$ by construction. It need not hold for $G_{\\text{f}}$, because the domain-compatible dummy-label renamings act only after the terms have already been summed.'),
      p('Therefore, Burnside over $G_{\\text{f}}$ answers a different question: it counts orbits of formal label renamings in the completed expression. It does not, in general, count the number of distinct summand computations or direct updates required during accumulation.'),
    ],
    summaryCaption: [
      p('Formal symmetry relates these terms after summation. It does not make them equal products.'),
    ],
    mismatchLead: [
      p('For the selected preset, the difference is visible numerically. A naive formal count using $G_{\\text{f}}$ would give $\\alpha_{\\text{formal, naive}}$. This is shown as a tempting shortcut; it is not an alternative cost measure. The accepted accumulation cost is the main $\\alpha$, and the gap is exactly the gap between formal equivalence and pointwise equality.'),
    ],
    coincidentLead: [
      p('For the selected preset, the formal count and the main count happen to agree at the current size. This numerical coincidence does not validate the formal shortcut. The valid cost group is still $G_{\\text{pt}}$ with output action $H$.'),
    ],
    noneLead: [
      p('For the selected preset, the formal group does not produce a distinct accumulation count to compare. The conceptual distinction remains: $G_{\\text{f}}$ is post-summation; $G_{\\text{pt}}$ is pre-summation.'),
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
    bilinearInequality: [
      p('Therefore $A[0,0]\\,A[1,1] = 4 \\neq 6 = A[0,1]\\,A[1,0]$.'),
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
    directInequality: [
      p('Therefore $T[0,0,0,1,2] = 1 \\neq 2 = T[0,0,0,2,1]$.'),
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
    mixedInequality: [
      p('Therefore $A[0,0]\\,B[0,1]\\,A[1,0] = 6 \\neq 8 = A[0,1]\\,B[1,0]\\,A[0,0]$.'),
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
      p('Use $G_{\\text{pt}}$ for product representatives. Use $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ for stored output representatives. Use $G_{\\text{f}} = H \\times \\prod_d S(W_d)$ only to describe formal symmetry of the completed expression. Do not use the dummy-label factor $\\prod_d S(W_d)$ to remove pre-summation product or update events.'),
    ],
  },
} satisfies SectionCopy;

export default section5;
