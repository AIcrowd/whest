const p = (text) => ({ kind: 'paragraph', text });

const section2 = {
  title: 'Product Symmetry',
  deck: 'What can be multiplied once?',
  slots: {
    intro: [
      p('The first question is pointwise: when do two full assignments produce the same pre-summation product? Once the contraction is fixed, we forget numerical entries and keep only incidence. Let $U$ be the set of operand-axis classes, one for each axis of each operand occurrence. The bipartite graph on $U$ and $L$ records which axis slots carry which labels, and the incidence matrix gives each label a fingerprint.'),
      p('Declared per-operand symmetry and repeated operand names define the candidate row moves. They do not yet prove a label relabeling is valid; they only say which axis/operand rearrangements are allowed to be tested. Label-size compatibility is part of the setup: a relabeling can only map labels with the same domain size. The same domain constraint will apply when a detected symmetry is restricted to output labels: only same-domain labels may be permuted.'),
    ],
    produces: [
      p('A structural candidate space: labels, axis classes, incidence fingerprints, declared slot actions, and the size constraints any valid relabeling must preserve.'),
    ],
  },
};

export default section2;
