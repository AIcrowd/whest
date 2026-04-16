import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  EXPLORER_ACTS,
  buildAnalysisCheckpoint,
} from './components/symmetry-aware-einsum-contractions/components/explorerNarrative.js';
import {
  mergeObservedActEntries,
  pickTopVisibleAct,
} from './components/symmetry-aware-einsum-contractions/lib/activeAct.js';

test('EXPLORER_ACTS defines the five narrative acts in the updated story order', () => {
  assert.deepEqual(
    EXPLORER_ACTS.map(({ id, navTitle, heading }) => ({ id, navTitle, heading })),
    [
      { id: 'setup', navTitle: 'Set Up', heading: 'Specify the Contraction' },
      { id: 'structure', navTitle: 'See Structure', heading: 'Encode the Structure' },
      { id: 'proof', navTitle: 'Prove Symmetry', heading: 'Detect and Generate the Symmetry Group' },
      { id: 'savings', navTitle: 'Detect Structure', heading: 'Decompose the Group Action' },
      { id: 'price-savings', navTitle: 'Price Savings', heading: 'Price Savings' },
    ],
  );
});

test('Acts 1 through 4 ask algorithmic questions rather than product-tour questions', () => {
  assert.match(EXPLORER_ACTS[0].question, /what exact einsum/i);
  assert.match(EXPLORER_ACTS[1].question, /represent this contraction/i);
  assert.match(EXPLORER_ACTS[2].question, /what full group do they generate/i);
  assert.match(EXPLORER_ACTS[3].question, /group is known/i);
});

test('pickTopVisibleAct prefers the top-most visible act and falls back safely', () => {
  assert.equal(
    pickTopVisibleAct(
      [
        { isIntersecting: true, target: { id: 'proof' }, boundingClientRect: { top: 260 } },
        { isIntersecting: true, target: { id: 'structure' }, boundingClientRect: { top: 48 } },
      ],
      'setup',
    ),
    'structure',
  );
});

test('shell contract uses primitive-based chrome instead of legacy header classes', () => {
  const source = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');
  assert.match(source, /ExplorerSectionCard/);
  assert.doesNotMatch(source, /font-accent/);
});

test('mergeObservedActEntries preserves prior entries and overwrites by act id', () => {
  const previousEntries = new Map([
    ['setup', { target: { id: 'setup' }, isIntersecting: false }],
    ['proof', { target: { id: 'proof' }, isIntersecting: false }],
  ]);
  const nextEntries = [
    { target: { id: 'proof' }, isIntersecting: true, boundingClientRect: { top: 32 } },
    { target: { id: 'savings' }, isIntersecting: true, boundingClientRect: { top: 240 } },
  ];

  const merged = mergeObservedActEntries(previousEntries, nextEntries);

  assert.equal(merged.size, 3);
  assert.equal(merged.get('setup').isIntersecting, false);
  assert.equal(merged.get('proof').isIntersecting, true);
  assert.equal(merged.get('savings').boundingClientRect.top, 240);
});

test('buildAnalysisCheckpoint summarizes the analyzed contraction state', () => {
  assert.deepEqual(
    buildAnalysisCheckpoint({
      example: { subscripts: ['ia', 'ib'], output: 'ab' },
      group: { vLabels: ['a', 'b'], wLabels: ['i'], fullGroupName: 'S2' },
    }),
    [
      { label: 'Expression', value: 'ia,ib -> ab' },
      { label: 'Free labels', value: 'a, b' },
      { label: 'Summed labels', value: 'i' },
      { label: 'Detected group', value: 'S2' },
    ],
  );
});
