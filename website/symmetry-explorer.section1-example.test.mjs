import test from 'node:test';
import assert from 'node:assert/strict';

import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { buildSection1ExampleView } from './components/symmetry-aware-einsum-contractions/lib/section1ExampleView.js';

test('buildSection1ExampleView derives exact einsum and structural summaries from a preset', () => {
  const view = buildSection1ExampleView(EXAMPLES.find((example) => example.id === 'triple-outer'));

  assert.equal(view.exactEinsumText, "einsum('ia,ib,ic->abc', X, X, X)");
  assert.equal(view.operandSummary, 'X, X, X');
  assert.equal(view.outputSummary, 'abc');
  assert.equal(view.vFreeSummary, 'a, b, c');
  assert.equal(view.wSummedSummary, 'i');
  assert.match(view.expandedEquationLatex, /R\[/);
  assert.match(view.expandedEquationLatex, /\\sum_/);
  assert.match(view.declaredSymmetrySummary, /X: dense/);
});

test('buildSection1ExampleView follows custom edited subscripts and output', () => {
  const view = buildSection1ExampleView({
    id: 'custom',
    name: 'Custom',
    variables: [
      { name: 'T', rank: 3, symmetry: 'symmetric', symAxes: [0, 1, 2], generators: '' },
    ],
    expression: {
      subscripts: 'ijk',
      output: 'i',
      operandNames: 'T',
    },
  });

  assert.equal(view.exactEinsumText, "einsum('ijk->i', T)");
  assert.equal(view.outputSummary, 'i');
  assert.equal(view.vFreeSummary, 'i');
  assert.equal(view.wSummedSummary, 'j, k');
  assert.match(view.declaredSymmetrySummary, /T: S3/);
  assert.match(view.expandedEquationLatex, /\\sum_/);
});

test('buildSection1ExampleView accepts local free/summed chrome colors for the expanded equation', () => {
  const view = buildSection1ExampleView(
    EXAMPLES.find((example) => example.id === 'triple-outer'),
    {
      freeLabelColor: '#F0524D',
      summedLabelColor: '#64748B',
    },
  );

  assert.match(view.expandedEquationLatex, /\\textcolor\{#F0524D\}\{a\}/);
  assert.match(view.expandedEquationLatex, /\\textcolor\{#F0524D\}\{b\}/);
  assert.match(view.expandedEquationLatex, /\\textcolor\{#F0524D\}\{c\}/);
  assert.match(view.expandedEquationLatex, /\\textcolor\{#64748B\}\{i\}/);
});
