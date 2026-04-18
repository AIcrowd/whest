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
  // Act 3 (two-kinds) is rendered via <TwoKindsSection> and carries its own framing;
  // count the 4 acts whose framing props appear directly in the App source (setup,
  // structure, proof, decompose) by slicing up to the price-savings section.
  const priceStart = appSource.indexOf('<section id={EXPLORER_ACTS[5].id}');
  const shellActsSource = priceStart >= 0 ? appSource.slice(0, priceStart) : appSource;

  assert.equal(countMatches(shellActsSource, /label="Interpretation"/g), 4);
  assert.equal(countMatches(shellActsSource, /label="Approach"/g), 4);
  assert.equal(countMatches(shellActsSource, /label="What this produces"/g), 4);
  assert.doesNotMatch(shellActsSource, /label="Why this matters"|label="Takeaway"/);
});
