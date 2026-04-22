import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section2 = {
  title: 'Encode the Structural Candidates',
  deck: 'Which relabelings are even possible before checking summand equality?',
  slots: {
    intro: [
      p('Once the contraction is fixed, we forget the numerical entries and keep only its incidence pattern. Let `$L$` be the set of index labels appearing in the expression, with a partition `$L = V_{\\mathrm{free}} \\sqcup W_{\\mathrm{summed}}$`, where `$V_{\\mathrm{free}}$` are the labels that remain visible at this stage and `$W_{\\mathrm{summed}}$` are the labels eliminated by summation. Let `$U$` be the set of operand-axis classes, one for each axis of each operand.'),
      p('We then build a bipartite graph on `$U$` and `$L$`, together with its incidence matrix `$M$`, recording which axis-classes meet which labels. This is the structural encoding the symmetry test acts on: the tensor values are gone, but the pattern that any relabeling must preserve is still visible. Declared per-operand symmetry is carried alongside this encoding and re-enters in [Section 3](#proof), when we generate candidate row moves.'),
    ],
    produces: [
      p('A structural candidate space: labels, axis classes, incidence fingerprints, and the metadata needed for the acceptance step.'),
    ],
  },
} satisfies SectionCopy;

export default section2;
