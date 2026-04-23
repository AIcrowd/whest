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

test('glossary copy includes the updated pointwise/formal/direct-cost terminology', async () => {
  const { GLOSSARY } = await import(
    './components/symmetry-aware-einsum-contractions/engine/glossary.js'
  );
  const byTerm = new Map(GLOSSARY.map((entry) => [entry.term, entry.definition]));

  assert.equal(
    byTerm.get('pointwise symmetry'),
    'A label relabeling π that preserves every pre-summation scalar product under the declared operand equality symmetries and repeated-operand identities. This is the group used for direct computation: one representative per orbit is valid only when every assignment in the orbit has the same summand product.',
  );
  assert.equal(
    byTerm.get('formal symmetry'),
    'A symmetry of the completed expression after summed labels have become bound variables. Formal symmetry may include output relabelings inherited from pointwise symmetry and same-domain dummy renamings of summed labels. It explains expression equality after summation; it is not generally valid for reducing pre-summation products or accumulation updates.',
  );
  assert.equal(
    byTerm.get('$S(W)$'),
    'The symmetric group on a same-domain block of summed labels. With heterogeneous sizes, the valid dummy-renaming factor is ∏_d S(W_d), where each W_d contains summed labels with the same domain/size. Full S(W) is valid only when all summed labels share a common domain.',
  );
  assert.equal(
    byTerm.get('representative products M'),
    'M is the number of product orbits under G_pt. In components, M = ∏_a M_a. It counts how many distinct product values the direct symmetry-aware evaluator must form before accounting for the k-operand multiplication chain length.',
  );
  assert.equal(
    byTerm.get('multiplication cost μ'),
    'μ is the multiplication-chain event count derived from representative products: μ = (k - 1)M for k operand tensors. μ is not the product-orbit count itself.',
  );
  assert.equal(
    byTerm.get('accumulation cost α'),
    'α is the direct output-bin update count. It is an orbit-projection count: sum over product orbits O of the number of distinct visible/output projections touched by O. It is not output storage and not generally equal to M.',
  );
  assert.equal(
    byTerm.get('component'),
    "A support-connected block of labels induced by the detected generators. Each component has labels L_a, output labels V_a, summed labels W_a, and restricted group G_a. The decomposition is safe for the displayed product formula; algebraically independent factors that remain inside a support-connected block are handled by the regime ladder.",
  );
  assert.equal(
    byTerm.get('Factorization check'),
    'The direct-product recognizer checks that no group element crosses V/W and that |G| = |G_V| · |G_W|. Passing means the action factors over visible and summed labels, so the direct-product α formula is exact.',
  );
});

test('shape tooltip copy reflects the updated M and α semantics', async () => {
  const { SHAPE_SPEC } = await import(
    './components/symmetry-aware-einsum-contractions/engine/shapeSpec.js'
  );

  assert.equal(
    SHAPE_SPEC.trivial.description,
    'No detected pointwise symmetry in this component, so every full assignment remains its own product/update representative.',
  );
  assert.equal(
    SHAPE_SPEC.allVisible.description,
    'No summed labels in this component. Product symmetry can reduce M, but the dense output still has one bin for every visible-label tuple.',
  );
  assert.equal(
    SHAPE_SPEC.allVisible.glossary.find((entry) => entry.term.includes('\\text{-symmetry}')).definition,
    'Symmetry on $V_{\\mathrm{free}}$ can reduce representative products, but it does not reduce direct output-bin updates in an all-visible dense output. Every visible tuple is still an output bin.',
  );
  assert.equal(
    SHAPE_SPEC.allSummed.description,
    'No visible labels in this component. Every product orbit updates the single scalar output once, so α equals the product-orbit count.',
  );
  assert.equal(
    SHAPE_SPEC.mixed.description,
    'Both $V_{\\mathrm{free}}$ and $W_{\\mathrm{summed}}$ are present. α must count output projections of product orbits, so the component dispatches to the regime ladder.',
  );
});
