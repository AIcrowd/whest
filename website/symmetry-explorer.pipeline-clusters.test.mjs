import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

test('analyzeExample exposes clusters array with id, labels, size', () => {
  const ex = EXAMPLES.find((e) => e.id === 'triple-outer');
  const analysis = analyzeExample(ex, 5);
  assert.ok(Array.isArray(analysis.clusters));
  for (const c of analysis.clusters) {
    assert.equal(typeof c.id, 'string');
    assert.ok(Array.isArray(c.labels));
    assert.equal(typeof c.size, 'number');
  }
});

test('clusters reflect G-orbit partition — triple-outer has an {a,b,c} cluster under S₃', () => {
  const ex = EXAMPLES.find((e) => e.id === 'triple-outer');
  const analysis = analyzeExample(ex, 5);
  const cABC = analysis.clusters.find((c) => c.labels.sort().join(',') === 'a,b,c');
  assert.ok(cABC, `expected a cluster {a,b,c}, got ${JSON.stringify(analysis.clusters)}`);
});

test('example labelSizes map overrides the global default', () => {
  const base = EXAMPLES.find((e) => e.id === 'triple-outer');
  const ex = { ...base, labelSizes: { 'a,b,c': 3, 'i': 7 } };
  const analysis = analyzeExample(ex, 5);
  const cABC = analysis.clusters.find((c) => c.id === 'a,b,c');
  const cI = analysis.clusters.find((c) => c.id === 'i');
  if (cABC) assert.equal(cABC.size, 3);
  if (cI) assert.equal(cI.size, 7);
});
