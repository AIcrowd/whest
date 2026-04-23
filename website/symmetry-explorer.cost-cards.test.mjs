import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import katex from 'katex';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('MultiplicationCostCard exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx');
  assert.match(src, /export default function MultiplicationCostCard/);
});

test('MultiplicationCostCard uses the aligned M/μ/α event labels and caption', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx');
  assert.match(src, /Multiplication Events \(μ\)/);
  assert.match(src, /M counts representative product values\. μ=\(k−1\)M counts multiplication-chain events\. α counts direct output-bin updates induced by product orbits\./);
  assert.match(src, /Live for this example/i);
  assert.match(src, /\\mu\s*\\;=\\;\s*\(/);
  assert.match(src, /num_terms/);
  assert.match(src, /\\mathrm\{cycles\}\(g\)/);
  assert.match(src, /multiplicationCount/);
  assert.match(src, /InlineMathText/);
  assert.match(src, /className="explorer-support-prose mt-2"/);
  assert.match(src, /V_\{\\mathrm\{free\}\}/);
  assert.match(src, /W_\{\\mathrm\{summed\}\}/);
  assert.doesNotMatch(src, /Calculating Multiplication Cost \(μ\)/);
});

test('AccumulationHardCard exports a default React component with shared support prose and no extra pointer footer', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/AccumulationHardCard.jsx');
  assert.match(src, /export default function AccumulationHardCard/);
  // Title now uses α to match the Counting Convention band.
  assert.match(src, /Why Accumulation Cost \(α\) is Hard/);
  assert.match(src, /notationColor\('alpha_total'\)/);
  assert.match(src, /InlineMathText/);
  assert.match(src, /className="explorer-support-prose mt-2"/);
  assert.match(src, /className="explorer-support-prose mt-3"/);
  assert.doesNotMatch(src, /bg-amber-400/);
  assert.doesNotMatch(src, /See the Classification Tree below/);
  assert.match(src, /V_\{\\mathrm\{free\}\}/);
  assert.match(src, /W_\{\\mathrm\{summed\}\}/);
});

test('ComponentCostView uses the shared support prose tier under subsection headers', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /className="explorer-support-prose mt-2"/);
  assert.match(src, /editorial-two-col-divider-lg editorial-two-col-divider-lg-inset border-y border-gray-100 py-6 grid gap-6 lg:grid-cols-2/);
});

test('ComponentCostView imports the two new cards + renders the CLASSIFICATION TREE section', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /import MultiplicationCostCard/);
  assert.match(src, /import AccumulationHardCard/);
  assert.match(src, /Classification Tree/);
  assert.match(src, /Product Orbits \(/);
  assert.match(src, /Accumulation Updates \(/);
  assert.match(src, /NotationSymbol id="m_component"/);
  assert.match(src, /NotationSymbol id="alpha_component"/);
  assert.match(src, /The dense baseline is the same direct-event convention without symmetry: one product chain and one output update for every full label assignment\./);
  assert.doesNotMatch(src, /Product orbits\s*</);
  assert.doesNotMatch(src, /Output updates\s*</);
  assert.doesNotMatch(src, /NotationSymbol id="m_total"/);
  assert.doesNotMatch(src, /NotationSymbol id="alpha_total"/);
});

test('Decision surfaces rename the scalar-output leaf to Direct Scalar Events', () => {
  const ladderSrc = read('components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx');
  const traceSrc = read('components/symmetry-aware-einsum-contractions/components/RegimeTrace.jsx');

  assert.match(ladderSrc, /Direct Scalar Events/);
  assert.match(traceSrc, /Direct Scalar Events/);
});

test('ComponentCostView passes activeLeafIds (all detected) to the DecisionLadder', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /activeLeafIds=/);
  // Should flatmap over components for union of regimeId + shape.
  assert.match(src, /c\.accumulation\?\.regimeId/);
  assert.match(src, /c\.shape/);
});

test('Section 5 piecewise brace renders as valid KaTeX', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');
  const match = src.match(/const PIECEWISE_BRACE = String\.raw`([^`]+)`;/);

  assert.ok(match, 'expected PIECEWISE_BRACE constant in TotalCostView');

  const html = katex.renderToString(match[1], {
    throwOnError: false,
    trust: true,
  });

  assert.doesNotMatch(html, /katex-error/);
});

test('TotalCostView compares dense and symmetry-aware direct event counts with product-size formulas', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');

  assert.match(src, /denseDirectEventCostFromComponents/);
  assert.match(src, /denseTupleCountFromComponents/);
  assert.match(src, /label:\s*'Dense Direct Events'/);
  assert.match(src, /label:\s*'Symmetry-Aware Direct Events'/);
  assert.match(src, /formula:\s*String\.raw`\((k-1|k-1)\)\\prod_\{\\ell\\in L\} n_\\ell \+ \\prod_\{\\ell\\in L\} n_\\ell`/);
  assert.doesNotMatch(src, /label:\s*'Dense Cost'/);
  assert.doesNotMatch(src, /label:\s*'Symmetry-Aware Cost'/);
  assert.doesNotMatch(src, /n\^\{\|L\|\}/);
  assert.match(src, /support-connected component decomposition/);
  assert.match(src, /M_a be the number of product orbits/);
  assert.match(src, /Under the independent-component factorization, M = ∏_a M_a and α = ∏_a α_a/);
  assert.match(src, /reported unavailable instead of being guessed/);
});
