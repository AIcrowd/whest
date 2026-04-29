import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section3 = {
  title: 'Output representatives induced by the product-side group',
  deck: 'The output action used by the main cost is $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$.',
  slots: {
    definitionLead: [
      p('Among the elements of $G_{\\text{pt}}$, keep exactly those that preserve the output-label set $V$ as a set. Restrict each such element to the output labels, using local output coordinates. The resulting action is'),
      p('This is the stored-output representative action used by the main accumulation count. If two output assignments lie in the same $H$-orbit, they share one stored output representative. Unlike dummy renaming, this action is inherited from pointwise product equality.'),
    ],
    workedExampleLabelPrefix: [
      l('Worked example —'),
    ],
    workedExamplePresetLabel: [
      l('bilinear trace'),
    ],
    workedExampleLead: [
      p('For bilinear trace at $n = 2$, the detected pointwise symmetry swaps $i$ and $j$ together with the corresponding summed structure. Its restriction to $V$ is the transposition $(i\\;j)$, so the output tensor satisfies $R[i,j] = R[j,i]$.'),
      p('A direct check shows $R[0,1] = R[1,0]$.'),
    ],
    workedExampleNote: [
      p('The equality $R[0,1] = R[1,0]$ is therefore genuine on the computed output tensor itself.'),
    ],
    takeaway: [
      p('$H$ is not a separate storage-only afterthought. It is the output-side quotient already present in $\\alpha = \\#\\{(O,Q) : \\pi_V(O) \\cap Q \\neq \\varnothing\\}$.'),
    ],
  },
} satisfies SectionCopy;

export default section3;
