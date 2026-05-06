import {
  mainEinsumGlance,
  mainProductSymmetry,
  mainProjection,
  mainRowsCols,
  mainComponentFactorization,
  mainCertification,
  mainCountingShortcuts,
  mainTypedPartition,
  mainAssembleCost,
  mainAppendixTransition,
} from '../content/main/index.js';

const getParagraphTexts = (blocks = []) => blocks
  .filter(({ kind }) => kind === 'paragraph')
  .map(({ text }) => text);
const getFirstParagraphText = (blocks = []) => blocks.find(({ kind }) => kind === 'paragraph')?.text;
const getIntroBlocks = (blocks = []) => blocks.map((block) => ({ ...block }));

export const EXPLORER_ACTS = [
  {
    id: 'einsum-glance',
    navTitle: 'Einsum at a Glance',
    heading: mainEinsumGlance.title,
    question: mainEinsumGlance.deck,
    introBlocks: getIntroBlocks(mainEinsumGlance.slots.intro),
    introParagraphs: getParagraphTexts(mainEinsumGlance.slots.intro),
    produces: getFirstParagraphText(mainEinsumGlance.slots.produces),
  },
  {
    id: 'product-symmetry',
    navTitle: 'Product Symmetry',
    heading: mainProductSymmetry.title,
    question: mainProductSymmetry.deck,
    introBlocks: getIntroBlocks(mainProductSymmetry.slots.intro),
    introParagraphs: getParagraphTexts(mainProductSymmetry.slots.intro),
    produces: getFirstParagraphText(mainProductSymmetry.slots.produces),
  },
  {
    id: 'projection',
    navTitle: 'Projection',
    heading: mainProjection.title,
    question: mainProjection.deck,
    introBlocks: getIntroBlocks(mainProjection.slots.intro),
    introParagraphs: getParagraphTexts(mainProjection.slots.intro),
    produces: getFirstParagraphText(mainProjection.slots.produces),
  },
  {
    id: 'rows-cols',
    navTitle: 'Rows and Columns',
    heading: mainRowsCols.title,
    question: mainRowsCols.deck,
    introBlocks: getIntroBlocks(mainRowsCols.slots.intro),
    introParagraphs: getParagraphTexts(mainRowsCols.slots.intro),
    produces: getFirstParagraphText(mainRowsCols.slots.produces),
  },
  {
    id: 'component-factorization',
    navTitle: 'Component Factorization',
    heading: mainComponentFactorization.title,
    question: mainComponentFactorization.deck,
    introBlocks: getIntroBlocks(mainComponentFactorization.slots.intro),
    introParagraphs: getParagraphTexts(mainComponentFactorization.slots.intro),
    produces: getFirstParagraphText(mainComponentFactorization.slots.produces),
  },
  {
    id: 'certification',
    navTitle: 'Certification',
    heading: mainCertification.title,
    question: mainCertification.deck,
    introBlocks: getIntroBlocks(mainCertification.slots.intro),
    introParagraphs: getParagraphTexts(mainCertification.slots.intro),
    produces: getFirstParagraphText(mainCertification.slots.produces),
  },
  {
    id: 'counting-shortcuts',
    navTitle: 'Counting Shortcuts',
    heading: mainCountingShortcuts.title,
    question: mainCountingShortcuts.deck,
    introBlocks: getIntroBlocks(mainCountingShortcuts.slots.intro),
    introParagraphs: getParagraphTexts(mainCountingShortcuts.slots.intro),
    produces: getFirstParagraphText(mainCountingShortcuts.slots.produces),
  },
  {
    id: 'typed-partition',
    navTitle: 'Partition Counting',
    heading: mainTypedPartition.title,
    question: mainTypedPartition.deck,
    introBlocks: getIntroBlocks(mainTypedPartition.slots.intro),
    introParagraphs: getParagraphTexts(mainTypedPartition.slots.intro),
    produces: getFirstParagraphText(mainTypedPartition.slots.produces),
  },
  {
    id: 'assemble-cost',
    navTitle: 'Assemble the Cost',
    heading: mainAssembleCost.title,
    question: mainAssembleCost.deck,
    introBlocks: getIntroBlocks(mainAssembleCost.slots.intro),
    introParagraphs: getParagraphTexts(mainAssembleCost.slots.intro),
    produces: getFirstParagraphText(mainAssembleCost.slots.produces),
  },
  {
    id: 'appendix-transition',
    navTitle: 'Appendix',
    heading: mainAppendixTransition.title,
    question: mainAppendixTransition.deck,
    introBlocks: getIntroBlocks(mainAppendixTransition.slots.intro),
    introParagraphs: getParagraphTexts(mainAppendixTransition.slots.intro),
    produces: getFirstParagraphText(mainAppendixTransition.slots.produces),
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
