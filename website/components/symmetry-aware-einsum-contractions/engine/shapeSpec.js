// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'Trivial',
    shortLabel: '∅',
    description: 'No symmetry — every assignment is its own orbit.',
    latex: String.raw`A = \prod_{\ell \in L} n_\ell \quad (|G|=1)`,
    when: 'Detected group G is trivial.',
    glossary: [
      { term: 'L', definition: 'the full label set of the component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$ (its dimension).' },
      { term: '|G| = 1', definition: 'means every assignment is its own singleton orbit, so $A$ collapses to $|X| = \\prod_\\ell n_\\ell$.' },
    ],
    color: '#94A3B8', // slate
  },
  allVisible: {
    id: 'allVisible',
    label: 'All-visible',
    shortLabel: 'V',
    description: 'No summed labels — output touches every bin.',
    latex: String.raw`A = \prod_{\ell \in V} n_\ell`,
    when: 'W = ∅.',
    glossary: [
      { term: 'V', definition: 'the free (output) labels.' },
      { term: 'W = \\varnothing', definition: 'no summed labels in this component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$. Every output bin is written exactly once, so $A$ is the Cartesian product of free-label sizes.' },
    ],
    color: '#4A7CFF', // blue
  },
  allSummed: {
    id: 'allSummed',
    label: 'All-summed',
    shortLabel: 'W',
    description: 'No free labels — orbits collapse multiplications and outputs together.',
    latex: String.raw`A = |X / G| = \frac{1}{|G|} \sum_{g \in G} \prod_{c \in \mathrm{cycles}(g)} n_c`,
    when: 'V = ∅.',
    glossary: [
      { term: 'X = [n]^L', definition: 'the full assignment space for this component.' },
      { term: 'X / G', definition: 'the $G$-orbits on $X$. Each orbit writes exactly one output bin, so $A$ equals the orbit count.' },
      { term: 'n_c', definition: 'the common label-size inside cycle $c$ of $g$ (forced equal by the action).' },
    ],
    color: '#64748B', // darker slate
  },
  mixed: {
    id: 'mixed',
    label: 'Mixed',
    shortLabel: 'V+W',
    description: 'V and W both nonempty. Dispatch to regime ladder.',
    latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
    when: 'V, W both nonempty.',
    glossary: [
      { term: 'O', definition: 'a $G$-orbit of full assignments.' },
      { term: '\\pi_V(O)', definition: 'its projection onto the free labels — the distinct output bins that orbit touches.' },
      { term: 'A', definition: 'the sum of projection sizes across all orbits — equivalent to counting (multiplication-orbit, output-bin) pairs.' },
    ],
    color: '#0F172A', // slate-900 (gateway)
  },
};
