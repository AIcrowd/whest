import {
  mainSection1,
  mainSection2,
  mainSection3,
  mainSection4,
  mainSection5,
} from '../content/main/index.js';

const getParagraphTexts = (blocks = []) => blocks
  .filter(({ kind }) => kind === 'paragraph')
  .map(({ text }) => text);
const getFirstParagraphText = (blocks = []) => blocks.find(({ kind }) => kind === 'paragraph')?.text;

export const EXPLORER_ACTS = [
  {
    id: 'setup',
    navTitle: 'Set Up',
    heading: mainSection1.title,
    question: mainSection1.deck,
    introParagraphs: getParagraphTexts(mainSection1.slots.intro),
    produces: getFirstParagraphText(mainSection1.slots.produces),
  },
  {
    id: 'structure',
    navTitle: 'See Structure',
    heading: mainSection2.title,
    question: mainSection2.deck,
    introParagraphs: getParagraphTexts(mainSection2.slots.intro),
    produces: getFirstParagraphText(mainSection2.slots.produces),
  },
  {
    id: 'proof',
    navTitle: 'Prove Symmetry',
    heading: mainSection3.title,
    question: mainSection3.deck,
    introParagraphs: getParagraphTexts(mainSection3.slots.intro),
    produces: getFirstParagraphText(mainSection3.slots.produces),
  },
  {
    id: 'decompose',
    navTitle: 'Decompose Action',
    heading: mainSection4.title,
    question: mainSection4.deck,
    introParagraphs: getParagraphTexts(mainSection4.slots.intro),
    produces: getFirstParagraphText(mainSection4.slots.produces),
  },
  {
    id: 'cost-savings',
    navTitle: 'Cost Savings',
    heading: mainSection5.title,
    question: mainSection5.deck,
  },
];

/**
 * Distill-style article section anchors. Used by any future continuous-scroll
 * layout. Not yet consumed by the main app (which still uses EXPLORER_ACTS);
 * tracked in TODOS.md as a follow-up.
 */
export const ARTICLE_SECTIONS = [
  { id: 'problem', heading: 'The counting problem', lede: 'Define product orbits, stored output representatives, and the single accumulation count alpha.' },
  { id: 'pointwise-group', heading: 'Detect product-side symmetry', lede: 'Certified relabelings generate G_pt and representative products.' },
  { id: 'output-action', heading: 'Derive output representatives', lede: 'H = Stab_{G_pt}(V)|_V is induced from the same detected group.' },
  { id: 'branching', heading: 'When projection branches', lede: 'No branching gives alpha = M; branching requires an exact counter.' },
  { id: 'partition-counting', heading: 'Typed partition counting', lede: 'Equality patterns give an exact compressed counter for general branching cases.' },
  { id: 'appendix', heading: 'Appendix', lede: 'Formal dummy symmetries explain completed-expression equality but do not replace the direct cost relation.' },
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
