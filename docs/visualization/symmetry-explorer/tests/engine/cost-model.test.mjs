import test from 'node:test';
import assert from 'node:assert/strict';

import * as engine from '../../src/engine/algorithm.js';
import { CASES } from './fixtures.mjs';

test('analyzeExample matches the cross S2 cost-model expectations', () => {
  const { example, dimensionN, expected } = CASES.crossS2;
  const result = engine.analyzeExample(example, dimensionN);

  assert.equal(result.symmetry.fullGroupName, expected.fullGroupName);
  assert.equal(result.costModel.orbitCount, expected.orbitCount);
  assert.equal(result.costModel.evaluationCost, expected.evaluationCost);
  assert.equal(result.costModel.reductionCost, expected.reductionCost);
});
