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
  assert.match(appSource, /Scope of the calculation/);
  assert.match(appSource, /Candidate, not proof/);
  assert.match(appSource, /What the model accepts/);
  assert.equal(countMatches(appSource, /tone="preamble"/g), 2);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.why/);
  assert.doesNotMatch(appSource, /onOpenModalSection=\{/);
});

test('"What this produces" callout uses shared vertical centering on desktop', () => {
  const calloutSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/NarrativeCallout.jsx', import.meta.url),
    'utf8',
  );

  assert.match(calloutSource, /sm:items-center/);
  assert.match(calloutSource, /sm:flex sm:items-center/);
  assert.doesNotMatch(calloutSource, /sm:items-start/);
});

test('article route still wires the appendix modal while the main page stays appendix-free', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );

  assert.match(appSource, /<ExpressionLevelModal/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*example=\{example\}/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*onSelectPreset=\{handleSelect\}/);
  assert.doesNotMatch(appSource, /VERBATIM, AUDIT-VERIFIED/);
  assert.doesNotMatch(appSource, /REVIEW_RESPONSE\.md §5/);
  assert.doesNotMatch(appSource, /AUDIT\.md/);
  assert.doesNotMatch(appSource, /empirically verified on 22 presets \+ 543 σ-checks/);
});
