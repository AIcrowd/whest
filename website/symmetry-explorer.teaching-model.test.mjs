import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildMentalModelCode,
  buildMentalModelLines,
  PSEUDOCODE_LINES,
  getFocusedLines,
  pickDefaultOrbitRow,
  tokenizePseudocodeLine,
} from './components/symmetry-explorer/engine/teachingModel.js';

test('PSEUDOCODE_LINES defines a stable teaching scaffold', () => {
  assert.equal(Array.isArray(PSEUDOCODE_LINES), true);
  assert.equal(PSEUDOCODE_LINES.length, 20);
  assert.deepEqual(
    PSEUDOCODE_LINES.map((line) => line.code),
    [
      '# sigma row moves induce valid pi relabelings on the active labels.',
      '# Those pi relabelings generate the full symmetry group G for this step.',
      '# multiplication_cost counts one product evaluation per G-orbit representative.',
      'multiplication_cost = 0',
      '# accumulation_cost counts one accumulation per distinct projected output bin.',
      'accumulation_cost = 0',
      '',
      '# RepSet = one representative full tuple from each orbit of G.',
      '# Burnside counts RepSet without enumerating every dense tuple.',
      '# Example rep = (...)',
      'for rep in RepSet:',
      '    base_val = product_of_operand_entries_at(rep)',
      '    multiplication_cost += max(num_terms - 1, 0)',
      '',
      '    # project_V keeps only the output labels V from a full tuple.',
      '    # Example Outs(rep) = [...]',
      '    # Example coeff(rep, (...)) = ?',
      '    for out in Outs(rep):',
      '        R[out] += coeff(rep, out) * base_val',
      '        accumulation_cost += 1',
    ],
  );
});

test('getFocusedLines maps each walkthrough step to the relevant pseudocode lines', () => {
  assert.deepEqual(getFocusedLines('framework'), [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]);
  assert.deepEqual(getFocusedLines('component-cost'), [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]);
  assert.deepEqual(getFocusedLines('total-cost'), [4, 6, 13, 20]);
  assert.deepEqual(
    getFocusedLines('unknown-step'),
    Array.from({ length: 20 }, (_, idx) => idx + 1),
  );
});

test('pickDefaultOrbitRow prefers the most informative orbit row', () => {
  const orbitRows = [
    { orbitSize: 1, outputCount: 1 },
    { orbitSize: 4, outputCount: 2 },
    { orbitSize: 4, outputCount: 3 },
  ];

  assert.equal(pickDefaultOrbitRow([]), -1);
  assert.equal(pickDefaultOrbitRow(null), -1);
  assert.equal(pickDefaultOrbitRow(orbitRows), 2);
});

test('tokenizePseudocodeLine marks keywords, state variables, functions, and numbers', () => {
  assert.deepEqual(
    tokenizePseudocodeLine('# coeff(rep, out) counts dense tuples in one output bin.'),
    [{ text: '# coeff(rep, out) counts dense tuples in one output bin.', kind: 'comment' }],
  );

  assert.deepEqual(tokenizePseudocodeLine('multiplication_cost = 0'), [
    { text: 'multiplication_cost', kind: 'state' },
    { text: ' ', kind: 'plain' },
    { text: '=', kind: 'plain' },
    { text: ' ', kind: 'plain' },
    { text: '0', kind: 'number' },
  ]);

  assert.deepEqual(tokenizePseudocodeLine('    for out in Outs(rep):'), [
    { text: '    ', kind: 'plain' },
    { text: 'for', kind: 'keyword' },
    { text: ' ', kind: 'plain' },
    { text: 'out', kind: 'state' },
    { text: ' ', kind: 'plain' },
    { text: 'in', kind: 'keyword' },
    { text: ' ', kind: 'plain' },
    { text: 'Outs', kind: 'function' },
    { text: '(', kind: 'plain' },
    { text: 'rep', kind: 'state' },
    { text: ')', kind: 'plain' },
    { text: ':', kind: 'plain' },
  ]);
});

test('buildMentalModelLines injects a concrete selected orbit example into the comments', () => {
  const lines = buildMentalModelLines({
    repTuple: { i: 0, j: 1, k: 2 },
    outputs: [
      { outTuple: { i: 0, k: 2 }, coeff: 1 },
      { outTuple: { i: 1, k: 2 }, coeff: 1 },
    ],
  });

  assert.equal(lines[9].code, '# Example rep = (i=0, j=1, k=2)');
  assert.equal(lines[15].code, '    # Example Outs(rep) = [(i=0, k=2), (i=1, k=2)]');
  assert.equal(lines[16].code, '    # Example coeff(rep, (i=0, k=2)) = 1');
});

test('buildMentalModelCode joins the selected orbit example into a single code block', () => {
  const selectedOrbitRow = {
    repTuple: { i: 0, j: 1, k: 2 },
    outputs: [
      { outTuple: { i: 0, k: 2 }, coeff: 1 },
      { outTuple: { i: 1, k: 2 }, coeff: 1 },
    ],
  };

  assert.equal(
    buildMentalModelCode(selectedOrbitRow),
    buildMentalModelLines(selectedOrbitRow).map(({ code }) => code).join('\n'),
  );
});
