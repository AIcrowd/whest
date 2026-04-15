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

test('EXPLORER_ACTS defines the four narrative acts in story order', () => {
  assert.deepEqual(
    EXPLORER_ACTS.map(({ id, navTitle, heading }) => ({ id, navTitle, heading })),
    [
      { id: 'setup', navTitle: 'Set Up', heading: 'Set Up The Contraction' },
      { id: 'structure', navTitle: 'See Structure', heading: 'See The Structure' },
      { id: 'proof', navTitle: 'Prove Symmetry', heading: 'Prove The Symmetry' },
      { id: 'savings', navTitle: 'Price Savings', heading: 'Price The Savings' },
    ],
  );
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

test('SymmetryExplorer keeps the act ids and renders the preset rail after main content', () => {
  const source = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');
  ['id="setup"', 'id="structure"', 'id="proof"', 'id="savings"'].forEach((needle) => {
    assert.match(source, new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  });
  assert.match(source, /activeActId/);
  assert.match(source, /StickyBar/);
  assert.match(source, /ExplorerSectionCard/);
  assert.match(source, /Button/);
  assert.ok(source.indexOf('<main') < source.indexOf('<PresetSidebar'));
});

test('shell contract uses primitive-based chrome instead of legacy header classes', () => {
  const source = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');

  assert.doesNotMatch(source, /font-accent/);
  assert.match(source, /font-heading/);
  assert.match(source, /text-2xl/);
  assert.match(source, /max-w-\[1400px\]/);
  assert.match(source, /gap-8/);
  assert.match(source, /min-h-screen bg-background/);
  assert.match(source, /title="Symmetry Aware Einsum Contractions"/);
  assert.match(source, /Badge className="bg-coral text-white hover:bg-coral">einsum/);
  assert.doesNotMatch(source, /app-header/);
  assert.doesNotMatch(source, /subtitle/);
  assert.doesNotMatch(source, /einsum-banner/);
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
