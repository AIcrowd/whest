// website/symmetry-explorer.label-clusters.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  computeLabelClusters,
  validateClusterSizes,
} from './components/symmetry-aware-einsum-contractions/engine/sizeAware/labelClusters.js';

test('single cluster when one generator swaps all labels transitively', () => {
  const labels = ['i', 'j', 'k'];
  // (i j k) as a 3-cycle: arr = [1, 2, 0]
  const cyc = new Permutation([1, 2, 0]);
  const clusters = computeLabelClusters(labels, [cyc]);
  assert.equal(clusters.length, 1);
  assert.deepEqual(clusters[0].labels.sort(), ['i', 'j', 'k']);
});

test('two clusters for two disjoint generators', () => {
  const labels = ['a', 'b', 'c', 'd'];
  const swapAB = new Permutation([1, 0, 2, 3]);
  const swapCD = new Permutation([0, 1, 3, 2]);
  const clusters = computeLabelClusters(labels, [swapAB, swapCD]);
  assert.equal(clusters.length, 2);
  const cSets = clusters.map(c => c.labels.sort().join(','));
  assert.ok(cSets.includes('a,b'));
  assert.ok(cSets.includes('c,d'));
});

test('trivial group yields one cluster per label', () => {
  const labels = ['i', 'j', 'k'];
  const clusters = computeLabelClusters(labels, []);
  assert.equal(clusters.length, 3);
  for (const c of clusters) assert.equal(c.labels.length, 1);
});

test('validateClusterSizes accepts matching sizes within each cluster', () => {
  const clusters = [
    { id: 'c0', labels: ['i', 'j'], size: 3 },
    { id: 'c1', labels: ['k'], size: 5 },
  ];
  assert.doesNotThrow(() => validateClusterSizes(clusters));
});

test('validateClusterSizes throws when cluster.size is not a positive integer', () => {
  const clusters = [{ id: 'c0', labels: ['i'], size: 0 }];
  assert.throws(() => validateClusterSizes(clusters), /positive integer/);
});

test('computeLabelClusters assigns a stable id derived from the sorted label list', () => {
  const labels = ['i', 'j', 'k'];
  const clusters = computeLabelClusters(labels, [new Permutation([1, 0, 2])]);
  const cAB = clusters.find(c => c.labels.includes('i'));
  assert.equal(cAB.id, 'i,j');
});
