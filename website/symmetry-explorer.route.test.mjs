import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('symmetry explorer acts use interpretation, algorithm, and output framing', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );

  assert.match(appSource, /label="Interpretation"/);
  assert.match(appSource, /label="Algorithm"/);
  assert.match(appSource, /label="What this produces"/);
});
