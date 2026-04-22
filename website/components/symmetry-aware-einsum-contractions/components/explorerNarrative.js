import {
  mainSection1,
  mainSection2,
  mainSection3,
  mainSection4,
  mainSection5,
} from '../content/main/index.ts';

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
