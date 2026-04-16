export const EXPLORER_ACTS = [
  {
    id: 'setup',
    navTitle: 'Set Up',
    heading: 'Set Up The Contraction',
    question: 'What contraction are we studying, and what will the walkthrough derive from it?',
    why: 'Fix the exact contraction first, separate free vs summed labels, and make the input to the rest of the page explicit.',
    takeaway: 'The contraction is fixed, so we can move from notation to structure.',
  },
  {
    id: 'structure',
    navTitle: 'See Structure',
    heading: 'See The Structure',
    question: 'How is this contraction represented structurally, and where can symmetry arise?',
    why: 'The graph gives the visual picture, while the incidence matrix gives the exact signature the detector works on.',
    bridge: 'Matching column fingerprints tell us which labels are even candidates for relabeling.',
    takeaway: 'We now know which structural features a valid symmetry must preserve.',
  },
  {
    id: 'proof',
    navTitle: 'Prove Symmetry',
    heading: 'Prove The Symmetry',
    question: 'Which relabelings survive, and what full group do they generate?',
    why: 'This act turns structural candidates into actual symmetries by checking which row moves still admit a recovering label permutation.',
    bridge: 'The surviving π relabelings generate the full group used by the later cost model.',
    takeaway: 'The full group is fixed, so we can count representatives and output-bin updates separately.',
  },
  {
    id: 'savings',
    navTitle: 'Detect Structure',
    heading: 'Detect the Structure',
    question: 'How do the discovered symmetries break into components with different cost behavior?',
    why: 'With the full group fixed, each independent component contributes its own representative count and output-update count.',
    bridge: 'Component-level structure (including orbit enumeration) explains where savings are coming from.',
    takeaway: 'This act surfaces the per-component cost story before calculating final totals.',
  },
  {
    id: 'price-savings',
    navTitle: 'Price Savings',
    heading: 'Price Savings',
    question: 'How much runtime do these components save in total?',
    why: 'The final table combines all components and compares full-cost and symmetry-aware cost, including multiplication, accumulation, and speedup.',
    bridge: 'These totals are the final payoff of the earlier structure and symmetry analysis.',
    takeaway: 'The final cost table is the concrete price of symmetry-aware contraction.',
  },
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
