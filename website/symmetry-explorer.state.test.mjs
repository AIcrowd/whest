import test from 'node:test';
import assert from 'node:assert/strict';

import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import {
  CUSTOM_IDX,
  getPresetControlSelection,
  getPresetSummary,
  presetToState,
  resolvePresetSelection,
} from './components/symmetry-aware-einsum-contractions/lib/presetSelection.js';

test('resolvePresetSelection keeps custom mode without requesting a preset reload', () => {
  assert.deepEqual(resolvePresetSelection(EXAMPLES, CUSTOM_IDX), {
    kind: 'custom',
    activePresetIdx: CUSTOM_IDX,
    dirtyState: 'preserve',
    presetState: null,
    presetSummary: null,
  });
});

test('resolvePresetSelection returns the exact preset state for a real example', () => {
  const expectedState = presetToState(EXAMPLES[2]);

  assert.deepEqual(resolvePresetSelection(EXAMPLES, 2), {
    kind: 'preset',
    activePresetIdx: 2,
    dirtyState: 'clear',
    presetState: expectedState,
    presetSummary: getPresetSummary(EXAMPLES[2]),
  });
});

test('getPresetSummary includes the expectedGroup used by the preset controls', () => {
  assert.deepEqual(getPresetSummary(EXAMPLES[0]), {
    id: 'matrix-chain',
    name: 'A·A (no symmetry)',
    formula: "einsum('ij,jk→ik', A, A)",
    description: 'Identical operands but different subscript structure → σ-loop finds no valid π',
    caseIds: ['trivial'],
    regimeId: 'trivial',
    expectedGroup: 'trivial',
    color: '#D1D5DB',
  });
});

test('getPresetSummary compresses mixed presets to shape + terminal regime badges', () => {
  assert.deepEqual(getPresetSummary(EXAMPLES.find((example) => example.id === 'young-s3')), {
    id: 'young-s3',
    name: 'Young S₃ (abc → ab)',
    formula: "einsum('abc→ab', T)",
    description: 'Full S₃ with cross-V/W elements and |V|=2 → Young regime closed form applies.',
    caseIds: ['mixed', 'young'],
    regimeId: 'young',
    expectedGroup: 'S3{a,b,c}',
    color: '#23B761',
  });

  assert.deepEqual(getPresetSummary(EXAMPLES.find((example) => example.id === 'direct-s2-c3')).caseIds, ['mixed', 'directProduct']);
  assert.deepEqual(getPresetSummary(EXAMPLES.find((example) => example.id === 'cross-s3')).caseIds, ['mixed', 'singleton']);
});

test('getPresetControlSelection projects dirty preset edits into custom mode', () => {
  assert.equal(getPresetControlSelection(0, true), CUSTOM_IDX);
  assert.equal(getPresetControlSelection(0, false), 0);
  assert.equal(getPresetControlSelection(CUSTOM_IDX, false), CUSTOM_IDX);
});
