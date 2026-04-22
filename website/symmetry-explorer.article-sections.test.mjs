import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { ARTICLE_SECTIONS, EXPLORER_ACTS } from './components/symmetry-aware-einsum-contractions/components/explorerNarrative.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

test('ARTICLE_SECTIONS exports 8 Distill-style sections with id, heading, lede', () => {
  assert.ok(Array.isArray(ARTICLE_SECTIONS));
  assert.equal(ARTICLE_SECTIONS.length, 8);
  for (const s of ARTICLE_SECTIONS) {
    assert.equal(typeof s.id, 'string');
    assert.equal(typeof s.heading, 'string');
    assert.equal(typeof s.lede, 'string');
  }
});

test('EXPLORER_ACTS still exported (backward compat)', () => {
  assert.ok(Array.isArray(EXPLORER_ACTS));
  assert.ok(EXPLORER_ACTS.length >= 4);
});

test('EXPLORER_ACTS sources main-page copy from registry modules', () => {
  const src = readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/explorerNarrative.js'),
    'utf-8',
  );

  assert.match(src, /import\s+\{[\s\S]*mainSection1[\s\S]*mainSection5[\s\S]*\}\s+from\s+'\.\.\/content\/main\/index\.ts'/);
  assert.match(src, /heading:\s*mainSection1\.title/);
  assert.match(src, /question:\s*mainSection1\.deck/);
  assert.match(src, /introParagraphs:\s*mainSection1\.slots\.intro\.map\(\(\{ text \}\) => text\)/);
  assert.match(src, /produces:\s*getFirstParagraphText\(mainSection1\.slots\.produces\)/);
  assert.doesNotMatch(src, /The first step is to fix the mathematical object/);
  assert.doesNotMatch(src, /Once the contraction is fixed, we forget the numerical entries/);
});

test('ARTICLE_SECTIONS and EXPLORER_ACTS have unique ids', () => {
  const articleIds = ARTICLE_SECTIONS.map((s) => s.id);
  const actIds = EXPLORER_ACTS.map((a) => a.id);
  assert.equal(new Set(articleIds).size, articleIds.length);
  assert.equal(new Set(actIds).size, actIds.length);
});

test('ExplorerModal component exists and exports a default function', () => {
  const src = readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/ExplorerModal.jsx'),
    'utf-8',
  );
  assert.match(src, /export default function ExplorerModal/);
  assert.match(src, /aria-modal="true"/);
});

test('App mounts the Mental Framework inline via the AlgorithmAtAGlance preamble, not a modal', () => {
  const src = readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx'),
    'utf-8',
  );
  // The preamble component is imported and rendered in the app.
  assert.match(src, /import AlgorithmAtAGlance from '\.\/components\/AlgorithmAtAGlance\.jsx'/);
  assert.match(src, /<AlgorithmAtAGlance[^>]*example=\{/);
  // The old modal pattern is gone from this file.
  assert.doesNotMatch(src, /<ExplorerModal[^>]*title="Mental Framework"/);
});

test('ComponentCostView uses ExplorerModal for Orbit Enumeration', () => {
  const src = readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx'),
    'utf-8',
  );
  assert.match(src, /import ExplorerModal/);
  assert.match(src, /<ExplorerModal[^>]*title="Orbit Enumeration"/);
});
