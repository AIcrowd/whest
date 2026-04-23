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
      { id: 'structure', navTitle: 'See Structure', heading: 'Encode the Structural Candidates' },
      { id: 'proof', navTitle: 'Prove Symmetry', heading: 'Certify the Pointwise Symmetry Group' },
      { id: 'decompose', navTitle: 'Decompose Action', heading: 'Count Product Orbits and Output Projections' },
      { id: 'cost-savings', navTitle: 'Cost Savings', heading: 'Assemble the Direct Cost Model' },
    ],
  );
});

test('Acts 1 through 4 ask algorithmic questions rather than product-tour questions', () => {
  assert.match(EXPLORER_ACTS[0].question, /what exact indexed computation is being counted/i);
  assert.match(EXPLORER_ACTS[1].question, /which relabelings are even possible/i);
  assert.match(EXPLORER_ACTS[2].question, /preserve the summand itself/i);
  assert.match(EXPLORER_ACTS[3].question, /pointwise group become multiplication and accumulation cost/i);
  assert.match(EXPLORER_ACTS[4].question, /final direct-event cost of the symmetry-aware computation/i);
});

test('Acts 1 through 4 expose introParagraphs and no longer expose paired-callout copy fields', () => {
  for (const act of EXPLORER_ACTS.slice(0, 4)) {
    assert.ok(Array.isArray(act.introParagraphs));
    assert.ok(act.introParagraphs.length >= 2);
    assert.ok(act.introParagraphs.every((paragraph) => typeof paragraph === 'string' && paragraph.length > 40));
  }

  for (const act of EXPLORER_ACTS.slice(0, 4)) {
    assert.equal(typeof act.produces, 'string');
    assert.ok(act.produces.length > 10);
  }
});

test('EXPLORER_ACTS no longer carries legacy paired-callout compatibility fields', () => {
  for (const act of EXPLORER_ACTS) {
    for (const legacyField of ['interpretation', 'algorithmTitle', 'algorithm', 'why', 'bridge', 'takeaway']) {
      assert.equal(legacyField in act, false, `did not expect ${legacyField} on ${act.id}`);
    }
  }
  assert.equal('supportingSentence' in EXPLORER_ACTS[4], false);
});

test('approved mathematically safer prose appears in the narrative data', () => {
  const section2 = [...EXPLORER_ACTS[1].introParagraphs, EXPLORER_ACTS[1].produces].join(' ');
  const section3 = EXPLORER_ACTS[2].introParagraphs.join(' ');
  const section4 = EXPLORER_ACTS[3].introParagraphs.join(' ');

  assert.match(section2, /forget numerical entries and keep only incidence/i);
  assert.match(section2, /operand-axis classes/i);
  assert.match(section2, /bipartite graph/i);
  assert.match(section2, /incidence matrix/i);
  assert.match(section2, /Declared per-operand symmetry and repeated operand names define the candidate row moves/i);
  assert.match(section2, /Label-size compatibility is part of the setup/i);
  assert.doesNotMatch(section2, /purely combinatorial encoding/i);

  assert.match(section3, /wreath product/i);
  assert.match(section3, /G_\{\\mathrm\{wreath\}\}/);
  assert.match(section3, /accepted witness/i);
  assert.match(section3, /detected pointwise group/i);
  assert.match(section3, /H_i \\wr S_\{m_i\}/);
  assert.match(section3, /used for product and accumulation counting/i);
  assert.doesNotMatch(section3, /derivePi/);

  assert.match(section4, /full assignment grid/i);
  assert.match(section4, /product orbit|representative products/i);
  assert.match(section4, /output bin/i);
  assert.match(section4, /projects an assignment orbit onto visible\/output labels/i);
  assert.match(section4, /output symmetry and accumulation symmetry are different/i);
});

test('Acts 1 through 4 still distinguish declared and detected symmetry in the new prose', () => {
  const joinedCopy = EXPLORER_ACTS
    .slice(0, 4)
    .flatMap(({ heading, question, introParagraphs, produces }) => [
      heading,
      question,
      ...introParagraphs,
      produces,
    ])
    .join(' ');

  assert.match(joinedCopy, /declared .*symmetr/i);
  assert.match(joinedCopy, /detected pointwise group/i);
  assert.match(joinedCopy, /product orbits/i);
  assert.match(joinedCopy, /projects an assignment orbit onto visible\/output labels|output symmetry and accumulation symmetry are different/i);
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
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );
  const stickyBarIndex = source.indexOf('<StickyBar');
  const firstSectionIndex = source.indexOf('<ExplorerSectionCard');

  assert.notEqual(stickyBarIndex, -1);
  assert.notEqual(firstSectionIndex, -1);
  assert.ok(stickyBarIndex < firstSectionIndex);
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
