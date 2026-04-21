import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const SIGMA_LOOP_SRC = readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx', import.meta.url),
  'utf8',
);
const APP_SRC = readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
  'utf8',
);

test('SigmaLoop renders its intro through the shared inline-math path so sigma and pi colorize', () => {
  assert.match(SIGMA_LOOP_SRC, /InlineMathText/);
  assert.match(SIGMA_LOOP_SRC, /notationLatex\('sigma_row_move'\)/);
  assert.match(SIGMA_LOOP_SRC, /notationLatex\('pi_relabeling'\)/);
  assert.match(SIGMA_LOOP_SRC, /className="explorer-support-prose mb-2"/);
  assert.doesNotMatch(SIGMA_LOOP_SRC, /Each σ is a wreath element; each accepted pair shows a row move together with its matching relabeling π\./);
});

test('sigma-loop panel title uses the shared inline-math path so sigma and pi colorize in the heading', () => {
  assert.match(APP_SRC, /InlineMathText/);
  assert.match(APP_SRC, /notationLatex\('sigma_row_move'\)/);
  assert.match(APP_SRC, /notationLatex\('pi_relabeling'\)/);
  assert.doesNotMatch(APP_SRC, />\s*σ-Loop &amp; π Detection\s*</);
});
