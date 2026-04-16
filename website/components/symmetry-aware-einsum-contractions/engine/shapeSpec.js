// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'Trivial',
    shortLabel: '∅',
    description: 'No symmetry — every assignment is its own orbit.',
    latex: String.raw`A = \prod_{\ell \in L} n_\ell \quad (|G|=1)`,
    when: 'Detected group G is trivial.',
    color: '#94A3B8', // slate
  },
  allVisible: {
    id: 'allVisible',
    label: 'All-visible',
    shortLabel: 'V',
    description: 'No summed labels — output touches every bin.',
    latex: String.raw`A = \prod_{\ell \in V} n_\ell`,
    when: 'W = ∅.',
    color: '#4A7CFF', // blue
  },
  allSummed: {
    id: 'allSummed',
    label: 'All-summed',
    shortLabel: 'W',
    description: 'No free labels — orbits collapse multiplications and outputs together.',
    latex: String.raw`A = |X / G|`,
    when: 'V = ∅.',
    color: '#64748B', // darker slate
  },
  mixed: {
    id: 'mixed',
    label: 'Mixed',
    shortLabel: 'V+W',
    description: 'V and W both nonempty. Dispatch to regime ladder.',
    latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
    when: 'V, W both nonempty.',
    color: '#0F172A', // slate-900 (gateway)
  },
};
