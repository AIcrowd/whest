import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('GlossaryList exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/GlossaryList.jsx');
  assert.match(src, /export default function GlossaryList/);
});

test('GlossaryList renders a <dl> with term/definition pairs', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/GlossaryList.jsx');
  assert.match(src, /<dl /);
  assert.match(src, /<dt /);
  assert.match(src, /<dd /);
});

test('GlossaryList uses Latex for term + GlossaryProse for definition', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/GlossaryList.jsx');
  assert.match(src, /<Latex math=\{term\}/);
  assert.match(src, /<GlossaryProse text=\{definition\}/);
});

test('tooltip prose renderers style backticked example ids as quiet mono chips', () => {
  const inlineMathSrc = read('components/symmetry-aware-einsum-contractions/components/InlineMathText.jsx');
  const glossaryProseSrc = read('components/symmetry-aware-einsum-contractions/components/GlossaryProse.jsx');
  assert.match(inlineMathSrc, /renderTooltipInlineText/);
  assert.match(inlineMathSrc, /segment\.startsWith\('`'\)/);
  assert.match(inlineMathSrc, /rounded bg-gray-100 px-1\.5 py-\[1px\] font-mono text-\[0\.92em\] text-gray-700/);
  assert.match(glossaryProseSrc, /renderTooltipInlineText/);
});

test('SHAPE_SPEC glossary entries are arrays of {term, definition}', async () => {
  const { SHAPE_SPEC } = await import(
    './components/symmetry-aware-einsum-contractions/engine/shapeSpec.js'
  );
  for (const [id, spec] of Object.entries(SHAPE_SPEC)) {
    assert.ok(Array.isArray(spec.glossary), `${id}.glossary must be an array`);
    for (const entry of spec.glossary) {
      assert.equal(typeof entry.term, 'string', `${id}: term must be a string`);
      assert.equal(typeof entry.definition, 'string', `${id}: definition must be a string`);
    }
  }
});

test('REGIME_SPEC glossary entries are arrays of {term, definition}', async () => {
  const { REGIME_SPEC } = await import(
    './components/symmetry-aware-einsum-contractions/engine/regimeSpec.js'
  );
  for (const [id, spec] of Object.entries(REGIME_SPEC)) {
    assert.ok(Array.isArray(spec.glossary), `${id}.glossary must be an array`);
    for (const entry of spec.glossary) {
      assert.equal(typeof entry.term, 'string', `${id}: term must be a string`);
      assert.equal(typeof entry.definition, 'string', `${id}: definition must be a string`);
    }
  }
});
