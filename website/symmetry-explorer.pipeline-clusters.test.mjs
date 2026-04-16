import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

test('analyzeExample exposes clusters array with id, labels, size', () => {
  const gram = EXAMPLES.find((e) => e.id === 'gram');
  const analysis = analyzeExample(gram, 5);
  assert.ok(Array.isArray(analysis.clusters));
  for (const c of analysis.clusters) {
    assert.equal(typeof c.id, 'string');
    assert.ok(Array.isArray(c.labels));
    assert.equal(typeof c.size, 'number');
  }
});

test('clusters reflect G-orbit partition — gram has a {a,b} cluster after symmetry detection', () => {
  const gram = EXAMPLES.find((e) => e.id === 'gram');
  const analysis = analyzeExample(gram, 5);
  const cAB = analysis.clusters.find((c) => c.labels.sort().join(',') === 'a,b');
  assert.ok(cAB, `expected a cluster {a,b}, got ${JSON.stringify(analysis.clusters)}`);
});

test('example labelSizes map overrides the global default', () => {
  const gram = EXAMPLES.find((e) => e.id === 'gram');
  const ex = { ...gram, labelSizes: { 'a,b': 3, 'i': 7 } };
  const analysis = analyzeExample(ex, 5);
  const cAB = analysis.clusters.find((c) => c.id === 'a,b');
  const cI = analysis.clusters.find((c) => c.id === 'i');
  if (cAB) assert.equal(cAB.size, 3);
  if (cI) assert.equal(cI.size, 7);
});
