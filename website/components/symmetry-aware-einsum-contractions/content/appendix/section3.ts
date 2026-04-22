import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section3 = {
  title: 'The pointwise group also induces output symmetry',
  deck: 'Some pointwise relabelings act visibly on the output tensor. This visible action is the part that will later matter for storage.',
  slots: {
    definitionLead: [
      p('Among the elements of $G_{\\text{pt}}$, some preserve the output-label set $V_{\\mathrm{free}}$. Restricting those elements to $V_{\\mathrm{free}}$ gives the output group'),
      p('This group acts on output cells. If two output assignments lie in the same $G_{\\mathrm{out}}$-orbit, the corresponding entries of the computed output tensor are equal. Unlike $S(W_{\\mathrm{summed}})$, this symmetry is inherited from pointwise equality, so it is legitimate on the output tensor itself.'),
    ],
    workedExampleLabelPrefix: [
      l('Worked example —'),
    ],
    workedExamplePresetLabel: [
      l('bilinear trace'),
    ],
    workedExampleLead: [
      p('For bilinear trace at $n = 2$, the detected pointwise symmetry swaps $i$ and $j$ together with the corresponding summed structure. Its restriction to $V_{\\mathrm{free}}$ is the transposition $(i\\;j)$, so the output tensor satisfies $R[i,j] = R[j,i]$.'),
      p('A direct check shows $R[0,1] = R[1,0]$.'),
    ],
    workedExampleNote: [
      p('The equality $R[0,1] = R[1,0]$ is therefore genuine on the computed output tensor itself.'),
    ],
    takeaway: [
      p('$G_{\\mathrm{out}}$ is not an extra formal artifact. It is the visible output action inherited from $G_{\\text{pt}}$.'),
    ],
  },
} satisfies SectionCopy;

export default section3;
