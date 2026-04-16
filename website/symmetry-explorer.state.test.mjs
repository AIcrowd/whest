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
import { reduceMentalModelVisibility } from './components/symmetry-aware-einsum-contractions/lib/mentalModelState.js';

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
    caseType: 'trivial',
    expectedGroup: 'trivial',
    color: '#D1D5DB',
  });
});

test('getPresetControlSelection projects dirty preset edits into custom mode', () => {
  assert.equal(getPresetControlSelection(0, true), CUSTOM_IDX);
  assert.equal(getPresetControlSelection(0, false), 0);
  assert.equal(getPresetControlSelection(CUSTOM_IDX, false), CUSTOM_IDX);
});

test('reduceMentalModelVisibility closes the modal on context changes and preserves other states', () => {
  assert.equal(reduceMentalModelVisibility(true, 'selectPreset'), false);
  assert.equal(reduceMentalModelVisibility(true, 'customMode'), false);
  assert.equal(reduceMentalModelVisibility(true, 'customExample'), false);
  assert.equal(reduceMentalModelVisibility(true, 'noop'), true);
  assert.equal(reduceMentalModelVisibility(false, 'noop'), false);
});
