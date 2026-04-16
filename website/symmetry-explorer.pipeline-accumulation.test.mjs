import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

test('analyzeExample attaches comp.shape + comp.accumulation for every component', () => {
  const ex = EXAMPLES.find((e) => e.id === 'gram');
  const analysis = analyzeExample(ex, 3);
  for (const comp of analysis.componentData.components) {
    assert.ok(comp.shape, `component missing shape: ${JSON.stringify(comp.labels)}`);
    assert.ok(comp.accumulation, `component missing accumulation`);
    assert.ok(typeof comp.accumulation.count === 'number' || comp.accumulation.count === null);
    assert.ok(Array.isArray(comp.accumulation.trace));
  }
});

test('analyzeExample produces accumulation across multiple preset examples', () => {
  const sampleIds = ['gram', 'triple-outer', 'outer'];
  for (const id of sampleIds) {
    const ex = EXAMPLES.find((e) => e.id === id);
    if (!ex) continue;
    const analysis = analyzeExample(ex, 3);
    assert.ok(analysis?.componentData?.components?.length > 0, `no components for ${id}`);
    for (const comp of analysis.componentData.components) {
      assert.ok(comp.shape, `${id}: component missing shape`);
      assert.ok(comp.accumulation, `${id}: component missing accumulation`);
    }
  }
});
