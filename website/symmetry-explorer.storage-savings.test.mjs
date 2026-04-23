import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeStorageSavingsRow,
  buildStorageSavingsRows,
} from './components/symmetry-aware-einsum-contractions/engine/storageSavings.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

function preset(id) {
  const found = EXAMPLES.find((example) => example.id === id);
  assert.ok(found, `missing preset ${id}`);
  return found;
}

test('storage table rows use current preset label sizes', () => {
  const triple = computeStorageSavingsRow(preset('triple-outer'), 3);
  const outer = computeStorageSavingsRow(preset('outer'), 3);

  assert.equal(triple.alphaEngine, 1296);
  assert.equal(outer.alphaEngine, 256);
  assert.notEqual(triple.alphaEngine, 162, 'stale triple-outer value must not remain');
  assert.notEqual(outer.alphaEngine, 144, 'stale outer value must not remain');
});

test('storage rows are sorted by savings percentage descending', () => {
  const rows = buildStorageSavingsRows(EXAMPLES, 3);
  assert.equal(rows.length, EXAMPLES.length);
  for (let i = 1; i < rows.length; i += 1) {
    assert.ok(rows[i - 1].savingPctNumber >= rows[i].savingPctNumber);
  }
});

test('scalar-output presets keep α_storage equal to α_engine', () => {
  const frobenius = computeStorageSavingsRow(preset('frobenius'), 3);
  assert.equal(frobenius.vLatex, String.raw`\varnothing`);
  assert.equal(frobenius.alphaStorage, frobenius.alphaEngine);
});
