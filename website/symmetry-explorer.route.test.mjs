import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

function countMatches(source, pattern) {
  const matches = source.match(pattern);
  return matches ? matches.length : 0;
}

test('symmetry explorer acts use interpretation, algorithm, and output framing', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );
  const act5Start = appSource.indexOf('<section id={EXPLORER_ACTS[4].id}');
  const shellActsSource = act5Start >= 0 ? appSource.slice(0, act5Start) : appSource;

  assert.equal(countMatches(shellActsSource, /label="Interpretation"/g), 4);
  assert.equal(countMatches(shellActsSource, /label="Algorithm"/g), 4);
  assert.equal(countMatches(shellActsSource, /label="What this produces"/g), 4);
  assert.doesNotMatch(shellActsSource, /label="Why this matters"|label="Takeaway"/);
});
