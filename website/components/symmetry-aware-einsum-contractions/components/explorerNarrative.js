import { notationLatex } from '../lib/notationSystem.js';

export const EXPLORER_ACTS = [
  {
    id: 'setup',
    navTitle: 'Set Up',
    heading: 'Specify the Contraction',
    question: 'What exact indexed computation is being counted?',
    introParagraphs: [
      'The first step is to fix the mathematical object, not to detect symmetry. We specify the ordered operands, the subscript carried by each operand, the output labels, the label sizes, and the equality symmetries declared on individual inputs.',
      'This produces a normalized contraction instance: an ordered list of tensor occurrences, one ordered slot list per occurrence, a visible label set, a summed label set, and the declared slot actions that are allowed inside each operand. Later sections may use those declarations, but they do not infer additional tensor identities from numerical values.',
    ],
    produces: 'A normalized contraction instance with explicit operands, slots, output labels, label sizes, and declared equality symmetries.',
  },
  {
    id: 'structure',
    navTitle: 'See Structure',
    heading: 'Encode the Structural Candidates',
    question: 'Which relabelings are even possible before checking summand equality?',
    introParagraphs: [
      `Once the contraction is fixed, we forget the numerical entries and keep only its incidence pattern. Let $${notationLatex('l_labels')}$ be the set of index labels appearing in the expression, with a partition $${notationLatex('l_labels')} = ${notationLatex('v_free')} \\sqcup ${notationLatex('w_summed')}$, where $${notationLatex('v_free')}$ are the labels that remain visible at this stage and $${notationLatex('w_summed')}$ are the labels eliminated by summation. Let $${notationLatex('u_axis_classes')}$ be the set of operand-axis classes, one for each axis of each operand.`,
      `We then build a bipartite graph on $${notationLatex('u_axis_classes')}$ and $${notationLatex('l_labels')}$, together with its incidence matrix $${notationLatex('m_incidence')}$, recording which axis-classes meet which labels. This is the structural encoding the symmetry test acts on: the tensor values are gone, but the pattern that any relabeling must preserve is still visible. Declared per-operand symmetry is carried alongside this encoding and re-enters in [Section 3](#proof), when we generate candidate row moves.`,
    ],
    produces: 'A structural candidate space: labels, axis classes, incidence fingerprints, and the metadata needed for the acceptance step.',
  },
  {
    id: 'proof',
    navTitle: 'Prove Symmetry',
    heading: 'Certify the Pointwise Symmetry Group',
    question: 'Which candidate relabelings preserve the summand itself?',
    introParagraphs: [
      `Candidate row moves come from the wreath product of declared axis symmetries and permutations of identical operand occurrences. If a repeated-operand family has $m_i$ copies and declared internal group $${notationLatex('h_family')}$, its row candidates contribute $${notationLatex('h_family')} \\wr S_{m_i}$; the product over families gives the candidate space $${notationLatex('g_wreath')}$ that the σ-loop enumerates.`,
      `For each row move $${notationLatex('sigma_row_move')}$, the explorer asks whether some label relabeling $${notationLatex('pi_relabeling')}$ restores the incidence pattern. An accepted pair $(\\sigma, \\pi)$ is the lifted witness used by this model for the detected pointwise action under the declared equality symmetries. The accepted relabelings are then closed under composition to obtain the detected pointwise group $${notationLatex('g_detected')}$ used by the cost model.`,
    ],
    produces: `A detected pointwise group $${notationLatex('g_detected')}$, represented by accepted $(\\sigma, \\pi)$ pairs and the generated label action.`,
  },
  {
    id: 'decompose',
    navTitle: 'Decompose Action',
    heading: 'Count Product Orbits and Output Projections',
    question: 'How does the group action become multiplication and accumulation cost?',
    introParagraphs: [
      `The cost model acts on full label assignments, not only on output labels. For each component, the multiplication count $${notationLatex('m_component')}$ is the number of representative product orbits under the restricted pointwise group, computed by Burnside or by exact orbit enumeration.`,
      `Accumulation is subtler. A product orbit may project to one output bin, several output bins, or no visible labels at all. Therefore the per-component accumulation count $${notationLatex('alpha_component')}$ is an orbit-projection count: one update for each distinct free-label projection touched by each product orbit. This is the step that prevents a free-label symmetry from being mistaken for an automatic reduction in output updates.`,
    ],
    produces: `Per-component quantities $${notationLatex('m_component')}$ and $${notationLatex('alpha_component')}$, with each component routed through the cheapest applicable exact formula or fallback enumeration.`,
  },
  {
    id: 'cost-savings',
    navTitle: 'Cost Savings',
    heading: 'Assemble the Direct Cost Model',
    question: 'What is the final cost of the symmetry-aware direct computation?',
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
  { id: 'spotlight-direct-product', heading: 'Spotlight: Direct product', lede: 'When V and W decouple, the equation splits into a trivial V part and a Burnside W part.' },
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
