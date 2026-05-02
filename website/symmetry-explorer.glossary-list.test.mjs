import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import katex from 'katex';
import { colorizeNotationLatex } from './components/symmetry-aware-einsum-contractions/lib/notationSystem.js';

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

test('GlossaryList keeps long math terms visible without internal term scrolling', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/GlossaryList.jsx');
  assert.match(src, /<dt className="inline/);
  assert.match(src, /<dd className="inline/);
  assert.match(src, /min-w-0/);
  assert.doesNotMatch(src, /overflow-x-auto/);
  assert.doesNotMatch(src, /shrink-0 whitespace-nowrap/);
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

test('shape and regime tooltip glossary terms render as valid KaTeX', async () => {
  const [{ SHAPE_SPEC }, { REGIME_SPEC }] = await Promise.all([
    import('./components/symmetry-aware-einsum-contractions/engine/shapeSpec.js'),
    import('./components/symmetry-aware-einsum-contractions/engine/regimeSpec.js'),
  ]);
  const specs = [
    ...Object.entries(SHAPE_SPEC).map(([id, spec]) => [`shape:${id}`, spec]),
    ...Object.entries(REGIME_SPEC).map(([id, spec]) => [`regime:${id}`, spec]),
  ];

  for (const [id, spec] of specs) {
    for (const entry of spec.glossary ?? []) {
      const html = katex.renderToString(colorizeNotationLatex(entry.term), {
        throwOnError: false,
        trust: true,
      });
      assert.equal(html.includes('katex-error'), false, `${id} glossary term failed KaTeX: ${entry.term}`);
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
    byTerm.get('accumulation count A or alpha'),
    'The number of updates from product-orbit representatives into stored output representatives. Formally, the number of pairs (O, Q) where O is a product orbit in X/G_pt and Q is a stored output representative orbit in Y/H hit by projecting O to the visible labels.',
  );
  assert.equal(
    byTerm.get('component'),
    "A support-connected block of labels induced by the detected generators. Each component has labels L_a, output labels V_a, summed labels W_a, and restricted group G_a. The decomposition is safe for the displayed product equation; algebraically independent factors that remain inside a support-connected block are handled by the regime ladder.",
  );
  // The legacy "Factorization check" entry was retired with the directProduct
  // regime; its successor is "functional projection" (G preserves V as a set).
  assert.equal(
    byTerm.get('functional projection'),
    'A regime that fires when every g ∈ G preserves V as a set: projection π_V then descends from product orbits to stored output representatives functionally, and α = M = |X/G|.',
  );
});

test('shape tooltip copy reflects the updated M and α semantics', async () => {
  const { SHAPE_SPEC } = await import(
    './components/symmetry-aware-einsum-contractions/engine/shapeSpec.js'
  );

  // Shape descriptions are now semantic rather than formula-owning under the
  // unified output-orbit metric — the regime ladder picks the cheapest exact
  // counter for one universal α. Each shape says when it routes through which
  // branch of the ladder, not a standalone formula.
  assert.equal(
    SHAPE_SPEC.trivial.description,
    'Each full assignment is its own product orbit, so each update goes to exactly one stored output representative.',
  );
  assert.equal(
    SHAPE_SPEC.allVisible.description,
    'There is no summation. Product representatives and stored output representatives are the same quotient.',
  );
  assert.equal(
    SHAPE_SPEC.allVisible.glossary.find((entry) => entry.term.includes('H = G')).definition,
    'with $V = L$ every $g \\in G$ trivially preserves $V$, so $H = G$ and product orbits and stored output representatives coincide. $\\alpha = M$.',
  );
  assert.equal(
    SHAPE_SPEC.allSummed.description,
    'The output is one scalar representative. Each product orbit updates it once.',
  );
  assert.match(SHAPE_SPEC.mixed.description, /Projection may have one destination per product orbit, or it may branch/);
});
