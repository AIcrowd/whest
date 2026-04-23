import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section2 = {
  title: 'Encode the Structural Candidates',
  deck: 'Which relabelings are even possible before checking summand equality?',
  slots: {
    intro: [
      p('Once the contraction is fixed, we forget numerical entries and keep only incidence. Let $U$ be the set of operand-axis classes, one for each axis of each operand occurrence. The bipartite graph on $U$ and $L$ records which axis slots carry which labels, and the incidence matrix $M$ gives each label a fingerprint.'),
      p('Declared per-operand symmetry and repeated operand names define the candidate row moves. They do not yet prove a label relabeling is valid; they only say which axis/operand rearrangements are allowed to be tested. Label-size compatibility is part of the setup: a relabeling can only map labels with the same domain size.'),
    ],
    produces: [
      p('A structural candidate space: labels, axis classes, incidence fingerprints, declared slot actions, and the size constraints any valid relabeling must preserve.'),
    ],
  },
} satisfies SectionCopy;

export default section2;
