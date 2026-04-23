import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = (...parts) => resolve(__dirname, ...parts);
const read = (...parts) => readFileSync(root(...parts), 'utf-8');

test('main copy distinguishes M, μ, and α', () => {
  const preamble = read('components/symmetry-aware-einsum-contractions/content/main/preamble.ts');
  const section4 = read('components/symmetry-aware-einsum-contractions/content/main/section4.ts');
  const total = read('components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');

  assert.match(preamble, /M \(representative products\)/);
  assert.match(preamble, /\\mu=\(k-1\)M/);
  assert.match(section4, /\\\\alpha = \\\\sum_\{O \\\\in X\/G_\{\\\\mathrm\{pt\}\}\}/);
  assert.match(total, /M_a be the number of product orbits/);
});

test('main copy does not overclaim n^5 divided by group order', () => {
  const preamble = read('components/symmetry-aware-einsum-contractions/content/main/preamble.ts');
  const glance = read('components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx');

  assert.doesNotMatch(preamble, /n\^\{?5\}?\/\|G\|/);
  assert.doesNotMatch(glance, /DENSE_SCALING/);
  assert.match(preamble, /If the action were free/);
  assert.match(preamble, /Burnside/);
});

test('scope copy names direct event model and unsupported features', () => {
  const app = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  assert.match(app, /direct indexed scalar-event count/);
  assert.match(app, /not whest’s general FMA FLOP convention/);
  assert.match(app, /repeated operand names denote the same tensor object/);
  assert.match(app, /no ellipsis, broadcasting, repeated labels within one input, duplicate output labels/);
});
