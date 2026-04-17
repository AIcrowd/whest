// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'Trivial',
    shortLabel: '∅',
    description: 'Direct count — no symmetry to exploit, so every cell is distinct work.',
    latex: String.raw`\alpha = \prod_{\ell \in L} n_\ell \quad (|G|=1)`,
    when: 'Detected group G is trivial.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — number of distinct output-bin updates.' },
      { term: 'L', definition: 'the full label set of the component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$ (its dimension).' },
      { term: 'G', definition: 'the detected symmetry group of the component; here $|G| = 1$.' },
      { term: '|G| = 1', definition: 'means every assignment is its own singleton orbit, so $\\alpha$ collapses to $|X| = \\prod_\\ell n_\\ell$.' },
    ],
    color: '#94A3B8', // slate
  },
  allVisible: {
    id: 'allVisible',
    label: 'All-visible',
    shortLabel: 'V',
    description: 'Cartesian product — W is empty, so each free-label tuple is its own output bin.',
    latex: String.raw`\alpha = \prod_{\ell \in V} n_\ell`,
    when: 'W = ∅.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — number of distinct output-bin updates.' },
      { term: 'V', definition: 'the free (output) labels.' },
      { term: 'W = \\varnothing', definition: 'no summed labels in this component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$. Every output bin is written exactly once, so $\\alpha$ is the Cartesian product of free-label sizes.' },
    ],
    color: '#4A7CFF', // blue
  },
  allSummed: {
    id: 'allSummed',
    label: 'All-summed',
    shortLabel: 'W',
    description: 'Size-aware Burnside — V is empty, so every orbit maps to the single scalar output.',
    latex: String.raw`\alpha = |X / G| = \tfrac{1}{|G|} \sum_{g \in G} \prod_{c \in \mathrm{cycles}(g)} n_c`,
    when: 'V = ∅.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — equals the number of $G$-orbits here, since $V = \\varnothing$ means each orbit writes the single scalar output once.' },
      { term: 'X = [n]^L', definition: 'the full assignment space for this component.' },
      { term: 'G', definition: 'the detected symmetry group acting on $X$.' },
      { term: 'X / G', definition: 'the set of $G$-orbits on $X$; by Burnside, $|X/G| = \\tfrac{1}{|G|}\\sum_g |\\mathrm{Fix}(g)|$.' },
      { term: 'g', definition: 'an element of $G$; we sum over all of them.' },
      { term: '\\mathrm{cycles}(g)', definition: 'the disjoint cycles of $g$ viewed as a label permutation.' },
      { term: 'n_c', definition: 'the common label-size inside cycle $c$ of $g$ (forced equal by the action).' },
    ],
    color: '#64748B', // darker slate
  },
  mixed: {
    id: 'mixed',
    label: 'Mixed',
    shortLabel: 'V+W',
    description: 'Mixed shape — V and W both nonempty; dispatch to the regime ladder.',
    latex: String.raw`\alpha = \sum_{O \in X/G} |\pi_V(O)|`,
    when: 'V, W both nonempty.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — total distinct output-bin updates across all orbits.' },
      { term: 'V', definition: 'the free (output) labels.' },
      { term: 'W', definition: 'the summed (contracted) labels.' },
      { term: 'G', definition: 'the detected symmetry group acting on $X$.' },
      { term: 'X = [n]^L', definition: 'the full assignment space.' },
      { term: 'O', definition: 'a $G$-orbit of full assignments in $X$.' },
      { term: '\\pi_V(O)', definition: "its projection onto the free labels — the distinct output bins that orbit touches." },
    ],
    color: '#0F172A', // slate-900 (gateway)
  },
};
