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
    id: 'two-kinds',
    navTitle: 'Two Kinds',
    heading: 'Two Kinds of Symmetry',
    question: 'When we say the einsum has a symmetry, what exactly is preserved — the individual terms, or just the total sum?',
    interpretation: 'Label permutations can be symmetries in two senses. Per-tuple symmetry preserves each summand; expression-level symmetry preserves the total sum but may reshuffle individual terms. Compression needs the stronger notion.',
    algorithmTitle: 'Build both groups',
    algorithm: 'The σ-loop emits Sources A (declared axis symmetries) and B (identical-operand swaps) — these produce the per-tuple group $G_{\\text{pt}}$. The expression-level group $G_{\\text{expr}}$ is then $V_{\\text{sub}} \\times S(W)$: the per-tuple group projected to V-labels, times all permutations of W-labels (dummy-rename symmetries, always valid).',
    produces: 'Two groups — $G_{\\text{pt}}$ (per-tuple, drives compression) and $G_{\\text{expr}}$ (expression-level, counting symmetry) — plus the Sources-A-and-B generator set that Section 4 materializes.',
    why: 'Without this distinction, compression over-counts reuse opportunities. Without both groups in the UI, readers see a single "detected symmetry" with no way to know which notion it represents or why the cost formulas use one and not the other.',
    bridge: 'Section 4 takes the $G_{\\text{pt}}$ generators, runs Dimino to enumerate the full group, and names it. $G_{\\text{expr}}$ is computed directly from the $V_{\\text{sub}} \\times S(W)$ formula; no Dimino needed.',
    takeaway: 'Two groups, two jobs. $G_{\\text{pt}}$ powers the cost formulas; $G_{\\text{expr}}$ tells the counting story.',
  },
  {
    id: 'proof',
    navTitle: 'Prove Symmetry',
    heading: 'Materialize and Name the Group',
    question: "Given the σ-loop's generators for $G_{\\text{pt}}$, what's the full enumerated group and what is it called?",
    interpretation: "Generators don't tell us the group — we need to close them under composition. Dimino's algorithm does this in polynomial time.",
    algorithmTitle: 'Close the group; classify',
    algorithm: 'Run Dimino on the Sources A+B generators to materialize every element of $G_{\\text{pt}}$. Classify by order and cycle structure: $S_n$, $C_n$, $D_n$, or custom.',
    produces: 'Fully enumerated $G_{\\text{pt}}$ with its standard name. $G_{\\text{expr}}$ is derived in parallel via $V_{\\text{sub}} \\times S(W)$.',
    why: "Section 5's component decomposition needs the full group, not just generators. Names ground the detected contraction symmetry group in familiar terms readers can reason about.",
    bridge: 'Section 5 decomposes the component-restricted groups and dispatches each component to a closed-form regime or to orbit enumeration.',
    takeaway: 'Both groups are fully materialized and named; we can now count.',
  },
  {
    id: 'decompose',
    navTitle: 'Decompose Action',
    heading: 'Decompose the Group Action',
    question: 'Once the full symmetry group is known, how does its action split into components with different counting behavior?',
    interpretation: 'Knowing the group is not enough; we still need to analyze how it acts on different parts of the label set.',
    algorithmTitle: 'Decompose the action',
    algorithm: 'Perform component decomposition, classify each component by shape and regime, use closed-form counting where valid, and fall back to brute-force orbit enumeration otherwise.',
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
