export const EXPLORER_ACTS = [
  {
    id: 'setup',
    navTitle: 'Set Up',
    heading: 'Specify the Contraction',
    question: 'What exact einsum are we analyzing, and what symmetries are declared on its input tensors?',
    interpretation: 'We pin down the exact operands, subscripts, free labels, summed labels, and declared input symmetry. Nothing is inferred yet; this act just fixes the problem statement.',
    algorithmTitle: 'Normalize the input',
    algorithm: 'Normalize the user-provided expression into an operand list, subscript strings, V, W, and the declared input symmetry actions.',
    produces: 'A fully specified contraction problem.',
    why: 'Fix the exact contraction first, separate free vs summed labels, and make the input to the rest of the page explicit.',
    takeaway: 'The contraction is fixed, so we can move from notation to structure.',
  },
  {
    id: 'structure',
    navTitle: 'See Structure',
    heading: 'Encode the Structure',
    question: 'How do we represent this contraction so that possible relabeling symmetries become testable?',
    interpretation: 'We convert the contraction into a structural object that records which labels touch which operand-axis classes.',
    algorithmTitle: 'Build the structural encoding',
    algorithm: 'Build a bipartite graph and incidence matrix, with declared input symmetry collapsing equivalent axis-positions first.',
    produces: 'A structural signature for each label.',
    why: 'The graph gives the visual picture, while the incidence matrix gives the exact signature the detector works on.',
    bridge: 'Matching column fingerprints tell us which labels are even candidates for relabeling.',
    takeaway: 'We now know which structural features a valid symmetry must preserve.',
  },
  {
    id: 'proof',
    navTitle: 'Prove Symmetry',
    heading: 'Detect and Generate the Symmetry Group',
    question: 'Which label relabelings are genuine symmetries of the contraction, and what full group do they generate?',
    interpretation: 'This act moves from candidates to proofs: we only keep relabelings that can be justified by the contraction structure.',
    algorithmTitle: 'Search and generate the group',
    algorithm: 'Enumerate candidate σ, search for compatible π, keep valid σ, and use Dimino’s algorithm to generate the explicit finite group action.',
    produces: 'The detected contraction symmetry group `G`, an explicit generator set, and the V/W partition over the labels — everything Section 4 needs to decompose the action.',
    why: 'This act turns structural candidates into actual symmetries by checking which row moves still admit a recovering label permutation.',
    bridge: 'The σ-loop pairs each row move with a label permutation π; valid pairs feed Dimino, which closes them into the full group G that Section 4 will decompose.',
    takeaway: 'The full group is fixed, so we can count representatives and output-bin updates separately.',
  },
  {
    id: 'decompose',
    navTitle: 'Decompose Action',
    heading: 'Decompose the Group Action',
    question: 'Once the full symmetry group is known, how does its action split into components with different counting behavior?',
    interpretation: 'Knowing the group is not enough; we still need to analyze how it acts on different parts of the label set.',
    algorithmTitle: 'Decompose the action',
    algorithm: 'Perform component decomposition, classify each component into cases A through E, use closed-form counting where valid, and use orbit enumeration otherwise.',
    produces: 'A component-level cost model: per-component Mₐ (orbit count) and αₐ (accumulation), ready for the global aggregate (k−1)·∏ₐ Mₐ + ∏ₐ αₐ.',
    why: 'With the full group fixed, each independent component contributes its own representative count and output-update count.',
    bridge: 'Orbit enumeration is not a heuristic fallback but the exact procedure required when no analytic shortcut is valid.',
    takeaway: 'This act surfaces the per-component cost story before calculating final totals.',
  },
  {
    id: 'price-savings',
    navTitle: 'Price Savings',
    heading: 'Price Savings',
    question: 'How much runtime do these components save in total?',
    why: 'This is the payoff of the previous sections: once the full group is fixed, the dense cost collapses to the orbit counts and output updates shown here.',
    bridge: 'These totals are the final payoff of the earlier structure and symmetry analysis.',
    takeaway: 'The final cost table is the concrete price of symmetry-aware contraction.',
  },
];

/**
 * Distill-style article section anchors. Used by any future continuous-scroll
 * layout. Not yet consumed by the main app (which still uses EXPLORER_ACTS);
 * tracked in TODOS.md as a follow-up.
 */
export const ARTICLE_SECTIONS = [
  { id: 'problem', heading: 'The counting problem', lede: 'Define Mₐ, αₐ, and the V/W split. One micro-example to build intuition.' },
  { id: 'shape', heading: 'Shape first', lede: 'Trivial, all-visible, all-summed — the three fast shortcuts before the regime ladder.' },
  { id: 'ladder', heading: 'Mixed: the ladder', lede: 'Singleton, then direct-product, then brute-force orbit projection — the three-step priority ladder for V+W mixed components.' },
  { id: 'spotlight-direct-product', heading: 'Spotlight: Direct product', lede: 'When V and W decouple, the formula splits into a trivial V part and a Burnside W part.' },
  { id: 'spotlight-singleton', heading: 'Spotlight: Singleton', lede: 'Weighted Burnside with a single free label — compact and powerful.' },
  { id: 'spotlight-brute-force', heading: 'Spotlight: Brute-force orbit', lede: 'The always-correct fallback; enumerate X, apply G, count distinct projections, gated by a budget cap.' },
  { id: 'playground', heading: 'Playground', lede: 'Tinker freely. Watch which regime fires. Share via URL.' },
  { id: 'appendix', heading: 'Appendix', lede: 'Ladder pseudocode, budget configuration, verification pointers.' },
];

export function buildAnalysisCheckpoint({ example, group }) {
  if (!example || !group) return [];

  const subscripts = Array.isArray(example.subscripts)
    ? example.subscripts
    : Array.isArray(example.expression?.subscripts)
      ? example.expression.subscripts
      : typeof example.expression?.subscripts === 'string'
        ? example.expression.subscripts.split(',').map((part) => part.trim()).filter(Boolean)
        : [];
  const output = example.output ?? example.expression?.output ?? '';
  const expression = `${subscripts.join(',')} -> ${output || '∅'}`;
  const freeLabels = group.vLabels?.length ? group.vLabels.join(', ') : '—';
  const summedLabels = group.wLabels?.length ? group.wLabels.join(', ') : '—';

  return [
    { label: 'Expression', value: expression },
    { label: 'Free labels', value: freeLabels },
    { label: 'Summed labels', value: summedLabels },
    { label: 'Detected group', value: group.fullGroupName || 'trivial' },
  ];
}
