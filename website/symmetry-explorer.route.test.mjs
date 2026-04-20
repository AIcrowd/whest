import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

function countMatches(source, pattern) {
  const matches = source.match(pattern);
  return matches ? matches.length : 0;
}

test('symmetry explorer acts use prose-first intros and output framing', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );
  const introSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/SectionIntroProse.jsx', import.meta.url),
    'utf8',
  );
  assert.match(appSource, /import SectionIntroProse from '\.\/components\/SectionIntroProse\.jsx';/);
  assert.equal(countMatches(appSource, /EXPLORER_ACTS\[[0-4]\]\.introParagraphs/g), 4);
  assert.match(appSource, /title={EXPLORER_ACTS\[4\]\.heading}/);
  assert.match(appSource, /description={<InlineMathText>{EXPLORER_ACTS\[4\]\.question}<\/InlineMathText>}/);
  assert.match(introSource, /md:grid-cols-2/);
  assert.match(introSource, /textAlign:\s*'justify'/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.supportingSentence/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.introParagraphs/);
  assert.equal(countMatches(appSource, /label="Interpretation"/g), 0);
  assert.equal(countMatches(appSource, /label="Approach"/g), 0);
  assert.equal(countMatches(appSource, /label="What this produces"/g), 4);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.why/);
  assert.doesNotMatch(appSource, /onOpenModalSection=\{/);
});
