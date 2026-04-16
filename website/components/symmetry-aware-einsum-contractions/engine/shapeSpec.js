// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'Trivial',
    shortLabel: '∅',
    description: 'No symmetry — every assignment is its own orbit.',
    latex: String.raw`A = \prod_{\ell \in L} n_\ell \quad (|G|=1)`,
    when: 'Detected group G is trivial.',
    glossary: '$L$: full label set. $n_\\ell$: size of label $\\ell$. With $|G| = 1$ every assignment is its own singleton orbit, so $A$ collapses to $|X| = \\prod_\\ell n_\\ell$.',
    color: '#94A3B8', // slate
  },
  allVisible: {
    id: 'allVisible',
    label: 'All-visible',
    shortLabel: 'V',
    description: 'No summed labels — output touches every bin.',
    latex: String.raw`A = \prod_{\ell \in V} n_\ell`,
    when: 'W = ∅.',
    glossary: '$V$: free (output) labels. With no summed labels, every output bin is written exactly once per multiplication orbit, so the accumulation count is just the Cartesian product of free-label sizes.',
    color: '#4A7CFF', // blue
  },
  allSummed: {
    id: 'allSummed',
    label: 'All-summed',
    shortLabel: 'W',
    description: 'No free labels — orbits collapse multiplications and outputs together.',
    latex: String.raw`A = |X / G| = \frac{1}{|G|} \sum_{g \in G} \prod_{c \in \mathrm{cycles}(g)} n_c`,
    when: 'V = ∅.',
    glossary: '$X = [n]^L$: assignment space. With no free labels each orbit writes exactly one output bin, so $A$ equals the orbit count $|X/G|$, computed via size-aware Burnside. $n_c$: common size within cycle $c$ of $g$.',
    color: '#64748B', // darker slate
  },
  mixed: {
    id: 'mixed',
    label: 'Mixed',
    shortLabel: 'V+W',
    description: 'V and W both nonempty. Dispatch to regime ladder.',
    latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
    when: 'V, W both nonempty.',
    glossary: '$O$: a $G$-orbit of full assignments. $\\pi_V(O)$: its projection onto free labels (the set of distinct output bins that orbit touches). $A$ is the sum of these projection sizes — equivalent to counting multiplication-orbit / output-bin pairs.',
    color: '#0F172A', // slate-900 (gateway)
  },
};
