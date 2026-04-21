import { notationLatex } from '../lib/notationSystem.js';

export const EXPLORER_ACTS = [
  {
    id: 'setup',
    navTitle: 'Set Up',
    heading: 'Specify the Contraction',
    question: 'What exact einsum are we analyzing, and what symmetries are declared on its input tensors?',
    introParagraphs: [
      'Before we can analyze symmetry, we have to fix the contraction itself. This section makes the object of study explicit: which tensors appear, which subscripts they carry, which output is being formed, and which input symmetries are declared rather than inferred.',
      'We rewrite that mathematical statement into one normalized internal form: an ordered list of operands, one subscript string per operand, an output string, and the declared symmetry actions attached to the inputs. Nothing has been detected yet. We are fixing the exact contraction problem that the later structural and group-theoretic steps will act on.',
    ],
    produces: 'A fully specified contraction problem.',
  },
  {
    id: 'structure',
    navTitle: 'See Structure',
    heading: 'Encode the Structure',
    question: 'How do we represent this contraction so that possible relabeling symmetries become testable?',
    introParagraphs: [
      `Once the contraction is fixed, we forget the numerical entries and keep only its incidence pattern. Let $${notationLatex('l_labels')}$ be the set of index labels appearing in the expression, with a partition $${notationLatex('l_labels')} = ${notationLatex('v_free')} \\sqcup ${notationLatex('w_summed')}$, where $${notationLatex('v_free')}$ are the labels that remain visible at this stage and $${notationLatex('w_summed')}$ are the labels eliminated by summation. Let $${notationLatex('u_axis_classes')}$ be the set of operand-axis classes, one for each axis of each operand.`,
      `We then build a bipartite graph on $${notationLatex('u_axis_classes')}$ and $${notationLatex('l_labels')}$, together with its incidence matrix $${notationLatex('m_incidence')}$, recording which axis-classes meet which labels. This is the structural encoding the symmetry test acts on: the tensor values are gone, but the pattern that any relabeling must preserve is still visible. Declared per-operand symmetry is carried alongside this encoding and re-enters in Section 3, when we generate candidate row moves.`,
    ],
    produces: 'A structural signature for each label.',
  },
  {
    id: 'proof',
    navTitle: 'Prove Symmetry',
    heading: 'Detect and Generate the Symmetry Group',
    question: 'Which label relabelings are genuine symmetries of the contraction, and what full group do they generate?',
    introParagraphs: [
      `With $${notationLatex('u_axis_classes')}$, $${notationLatex('l_labels')}$, and $${notationLatex('m_incidence')}$ fixed, we can now say where symmetry candidates come from. Suppose the repeated-operand families are indexed by $i$, each family has $m_i$ identical copies, and one copy carries a declared axis-symmetry group $${notationLatex('h_family')}$. These row symmetries combine into the wreath product $${notationLatex('g_wreath')} = \\prod_i (${notationLatex('h_family')} \\wr S_{m_i})$. We build this group first because it contains exactly the row moves worth testing: internal symmetries inside each copy together with permutations of identical copies.`,
      `Each element $${notationLatex('sigma_row_move')} \\in ${notationLatex('g_wreath')}$ acts by moving the rows of $${notationLatex('m_incidence')}$. We then ask whether there exists a relabeling $${notationLatex('pi_relabeling')} \\in ${notationLatex('sym_l')}$ that restores the same incidence pattern. When such a $${notationLatex('pi_relabeling')}$ exists, we keep it; when it does not, we discard that candidate row move. The surviving relabelings form the detected symmetry group $${notationLatex('g_detected')}$. The panels below show this progression in order: the candidate wreath product, the accepted $(\\sigma, \\pi)$ pairs, and the resulting group.`,
    ],
    produces: `The detected symmetry group $${notationLatex('g_detected')}$, an explicit set of accepted relabelings, and the label action needed for decomposition.`,
  },
  {
    id: 'decompose',
    navTitle: 'Decompose Action',
    heading: 'Decompose the Group Action',
    question: 'Once the full symmetry group is known, how does its action split into components with different counting behavior?',
    introParagraphs: [
      `Once $${notationLatex('g_detected')}$ is known, the remaining question is not whether symmetry exists but how its action decomposes. Different parts of the label set can move together or independently, and the cost model depends on that finer structure.`,
      `We therefore decompose the action of $${notationLatex('g_detected')}$ into components $a$, each with its own label set $${notationLatex('l_component')} = ${notationLatex('v_free_component')} \\sqcup ${notationLatex('w_summed_component')}$ and restricted symmetry group $${notationLatex('g_component')}$. For each component we compute two counts: $${notationLatex('m_component')}$, the number of representative multiplication orbits, and $${notationLatex('alpha_component')}$, the number of distinct accumulation targets. When a component satisfies the hypotheses of a closed-form regime, we use it; when it does not, we count exactly by orbit enumeration. This is where the symmetry group becomes the quantities that drive the runtime model.`,
    ],
    produces: `A component-level cost model: per-component $${notationLatex('m_component')}$ (orbit count) and $${notationLatex('alpha_component')}$ (accumulation), ready for the global aggregate.`,
  },
  {
    id: 'cost-savings',
    navTitle: 'Cost Savings',
    heading: 'Assemble the Cost Model',
    question: 'How does the detected symmetry determine the total cost of the contraction?',
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
