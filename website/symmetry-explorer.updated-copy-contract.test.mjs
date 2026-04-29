import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = (...parts) => resolve(__dirname, ...parts);
const read = (...parts) => readFileSync(root(...parts), 'utf-8');

test('main copy distinguishes M, μ, and α', () => {
  const preamble = read('components/symmetry-aware-einsum-contractions/content/main/preamble.js');
  const section4 = read('components/symmetry-aware-einsum-contractions/content/main/section4.js');
  const total = read('components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');

  assert.match(preamble, /\$M\$ \(representative products\)/);
  assert.match(preamble, /\\mu = \(k-1\)M/);
  assert.match(section4, /\\\\alpha = \\\\sum_\{O \\\\in X\/G_\{\\\\text\{pt\}\}\}/);
  assert.match(total, /let \$M_a\$ be the number of product orbits/);
});

test('main copy does not overclaim n^5 divided by group order', () => {
  const preamble = read('components/symmetry-aware-einsum-contractions/content/main/preamble.js');
  const glance = read('components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx');

  assert.doesNotMatch(preamble, /n\^\{?5\}?\/\|G\|/);
  assert.doesNotMatch(glance, /DENSE_SCALING/);
  assert.match(preamble, /If the action were free/);
  assert.match(preamble, /Burnside/);
});

test('scope copy names direct event model and unsupported features', () => {
  const app = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  assert.match(app, /direct indexed scalar-event count/);
  assert.match(app, /not flopscope’s general FMA FLOP convention/);
  assert.match(app, /repeated operand names denote the same tensor object/);
  assert.match(app, /forbids[\s\S]*ellipsis, broadcasting, repeated labels within one input, and[\s\S]*duplicate output labels/);
});

test('appendix copy uses domain-compatible formal dummy renaming', () => {
  const section2 = read('components/symmetry-aware-einsum-contractions/content/appendix/section2.ts');
  const section4 = read('components/symmetry-aware-einsum-contractions/content/appendix/section4.ts');
  const section5 = read('components/symmetry-aware-einsum-contractions/content/appendix/section5.ts');

  assert.match(section2, /same-domain blocks/);
  assert.match(section4, /\\\\prod_d S\(W_d\)/);
  assert.match(section5, /domain-compatible dummy-label factor/);
  assert.doesNotMatch(section4, /G_\{\\text\{f\}\} = G_\{\\mathrm\{out\}\} \\times S\(W_\{\\mathrm\{summed\}\}\)/);
});
