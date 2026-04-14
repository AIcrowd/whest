import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CASES,
} from './fixtures.mjs';
import {
  buildBipartite,
  buildGroup,
  buildIncidenceMatrix,
  runSigmaLoop,
} from '../../src/engine/algorithm.js';

test('runSigmaLoop keeps cross V/W pi mappings for cross S2', () => {
  const { example, expected } = CASES.crossS2;
  const graph = buildBipartite(example);
  const matrixData = buildIncidenceMatrix(graph);
  const sigmaResults = runSigmaLoop(graph, matrixData, example);
  const validKinds = sigmaResults
    .filter(r => r.isValid && !r.skipped && !r.piIsIdentity)
    .map(r => r.piKind)
    .sort();

  assert.deepEqual(validKinds, expected.piKinds);
});

test('buildGroup exposes the full group and action summary for mixed S3', () => {
  const { example, expected } = CASES.mixedS3;
  const graph = buildBipartite(example);
  const matrixData = buildIncidenceMatrix(graph);
  const sigmaResults = runSigmaLoop(graph, matrixData, example);
  const group = buildGroup(sigmaResults, graph, example);

  assert.equal(group.fullGroupName, expected.fullGroupName);
  assert.deepEqual(
    {
      hasCross: group.actionSummary?.hasCross,
      hasVOnly: group.actionSummary?.hasVOnly,
      hasWOnly: group.actionSummary?.hasWOnly,
      hasCorrelated: group.actionSummary?.hasCorrelated,
    },
    {
      hasCross: true,
      hasVOnly: false,
      hasWOnly: true,
      hasCorrelated: false,
    },
    'actionSummary should be a structured summary, not a placeholder or garbage string'
  );
});
