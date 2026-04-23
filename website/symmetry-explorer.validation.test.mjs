import test from 'node:test';
import assert from 'node:assert/strict';

import { validateAll } from './components/symmetry-aware-einsum-contractions/engine/validation.js';
import { ERROR_CODES } from './components/symmetry-aware-einsum-contractions/engine/validationMessages.js';

// ── Helpers ──────────────────────────────────────────────────────────

function v(partial) {
  return {
    name: 'T',
    rank: 3,
    symmetry: 'none',
    symAxes: null,
    generators: '',
    ...partial,
  };
}

function stateFor(variables, subscriptsStr, outputStr, operandNamesStr) {
  return { variables, subscriptsStr, outputStr, operandNamesStr };
}

function run(state) {
  return validateAll(
    state.variables,
    state.subscriptsStr,
    state.outputStr,
    state.operandNamesStr,
  );
}

function findError(result, code) {
  return result.errors.find((e) => e.code === code);
}

function applyFix(state, error) {
  assert.ok(error, 'expected an error to apply a fix on');
  assert.ok(error.fix, `expected error ${error.code} to have a fix`);
  return error.fix.apply(state);
}

// ── Regression + original custom-generator coverage ─────────────────

test('does not throw when custom symmetry has non-empty generators', () => {
  assert.doesNotThrow(() =>
    validateAll(
      [v({ symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1)(1 2)' })],
      'ijk',
      'ijk',
      'T',
    ),
  );
});

test('valid custom cycle generator validates clean', () => {
  const r = validateAll(
    [v({ symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1)' })],
    'ijk',
    'ijk',
    'T',
  );
  assert.equal(r.valid, true);
  assert.deepEqual(r.errors, []);
});

test('every error has a structured shape', () => {
  const r = validateAll([v({ name: '' })], '', '', '');
  for (const err of r.errors) {
    assert.equal(typeof err.code, 'string');
    assert.equal(typeof err.field, 'string');
    assert.equal(typeof err.message, 'string');
    if (err.fix) {
      assert.equal(typeof err.fix.label, 'string');
      assert.equal(typeof err.fix.apply, 'function');
    }
  }
});

// ── Per-rule tests ───────────────────────────────────────────────────

test('#1 name-empty: message + field', () => {
  const r = run(stateFor([v({ name: '' })], 'ij', 'ij', 'T'));
  const err = findError(r, ERROR_CODES.NAME_EMPTY);
  assert.ok(err);
  assert.match(err.message, /needs a name/i);
  assert.equal(err.field, 'var-0-name');
});

test('#2 rank-too-small: fix sets rank to 1', () => {
  const state = stateFor([v({ rank: 0 })], 'ij', 'ij', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.RANK_TOO_SMALL);
  assert.ok(err);
  assert.match(err.message, /rank ≥ 1/);
  assert.equal(err.fix.label, 'Set rank to 1');
  const next = applyFix(state, err);
  const r2 = run(next);
  assert.equal(findError(r2, ERROR_CODES.RANK_TOO_SMALL), undefined);
  assert.equal(next.variables[0].rank, 1);
});

test('#3 named-sym-axes-too-few: fix selects all axes', () => {
  const state = stateFor([v({ symmetry: 'cyclic', symAxes: [0] })], 'ijk', 'ijk', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.NAMED_SYM_AXES_TOO_FEW);
  assert.ok(err);
  assert.match(err.message, /cyclic symmetry needs at least 2 axes/);
  assert.equal(err.fix.label, 'Select all axes');
  const next = applyFix(state, err);
  const r2 = run(next);
  assert.equal(findError(r2, ERROR_CODES.NAMED_SYM_AXES_TOO_FEW), undefined);
});

test('#4 named-sym-axis-oor: fix removes the bad axis', () => {
  const state = stateFor([v({ rank: 2, symmetry: 'symmetric', symAxes: [0, 1, 2] })], 'ij', 'ij', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.NAMED_SYM_AXIS_OOR);
  assert.ok(err);
  assert.match(err.message, /axis 2 doesn't exist/);
  assert.match(err.message, /valid axes are 0–1/);
  assert.equal(err.fix.label, 'Remove axis 2');
  const next = applyFix(state, err);
  assert.deepEqual(next.variables[0].symAxes, [0, 1]);
});

test('#5 custom-generators-empty: tailored 2-axis message', () => {
  const r = run(stateFor(
    [v({ rank: 2, symmetry: 'custom', symAxes: [0, 1] })],
    'ij', 'ij', 'T',
  ));
  const err = findError(r, ERROR_CODES.CUSTOM_GENERATORS_EMPTY);
  assert.ok(err);
  assert.match(err.message, /swap the two selected axes/);
  assert.equal(err.field, 'var-0-generators');
});

test('#6 custom-generators-parse: delegates to friendly prettifier', () => {
  const r = run(stateFor(
    [v({ symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1' })],
    'ijk', 'ijk', 'T',
  ));
  const err = findError(r, ERROR_CODES.CUSTOM_GENERATORS_PARSE);
  assert.ok(err);
  assert.match(err.message, /outside any cycle/);
});

test('#7 custom-generator-axis-oor: message names the valid range', () => {
  const r = run(stateFor(
    [v({ rank: 2, symmetry: 'custom', symAxes: [0, 1], generators: '(0 1)(2 3)' })],
    'ij', 'ij', 'T',
  ));
  const err = findError(r, ERROR_CODES.CUSTOM_GENERATOR_AXIS_OOR);
  assert.ok(err);
  assert.match(err.message, /valid indices are 0–1/);
});

test('custom generator indices are local to the selected-axis list', () => {
  const local = run(stateFor(
    [v({ rank: 4, symmetry: 'custom', symAxes: [2, 3], generators: '(0 1)' })],
    'abcd',
    'abcd',
    'T',
  ));
  assert.equal(local.valid, true);

  const absolute = run(stateFor(
    [v({ rank: 4, symmetry: 'custom', symAxes: [2, 3], generators: '(2 3)' })],
    'abcd',
    'abcd',
    'T',
  ));
  const err = findError(absolute, ERROR_CODES.CUSTOM_GENERATOR_AXIS_OOR);
  assert.ok(err);
  assert.match(err.message, /valid indices are 0–1/);
});

test('#8 no-operands: helpful example', () => {
  const r = run(stateFor([v()], '', '', ''));
  const err = findError(r, ERROR_CODES.NO_OPERANDS);
  assert.ok(err);
  assert.match(err.message, /at least one operand/);
  assert.match(err.message, /einsum/);
});

test('#9 subscripts-operands-count-mismatch: names the two counts', () => {
  const r = run(stateFor([v()], 'ij, jk', 'ik', 'T'));
  const err = findError(r, ERROR_CODES.SUBSCRIPTS_OPERANDS_COUNT_MISMATCH);
  assert.ok(err);
  assert.match(err.message, /2 subscripts/);
  assert.match(err.message, /1 operand/);
});

test('#10 operand-undefined: offers both options in the message', () => {
  const r = run(stateFor([v()], 'ijk', 'ijk', 'Q'));
  const err = findError(r, ERROR_CODES.OPERAND_UNDEFINED);
  assert.ok(err);
  assert.match(err.message, /rename the operand/);
  assert.match(err.message, /add a variable called "Q"/);
});

test('#11 subscript-non-lowercase: fix cleans the subscript', () => {
  const state = stateFor([v()], 'Ij9', 'ij', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.SUBSCRIPT_NON_LOWERCASE);
  assert.ok(err);
  assert.match(err.message, /lowercase letters a–z only/);
  assert.equal(err.fix.label, 'Clean to "ij"');
  const next = applyFix(state, err);
  assert.equal(next.subscriptsStr, 'ij');
});

test('#12 subscript-duplicate-label: pinpoints the repeated letter', () => {
  const r = run(stateFor([v({ rank: 3 })], 'iij', 'ij', 'T'));
  const err = findError(r, ERROR_CODES.SUBSCRIPT_DUPLICATE_LABEL);
  assert.ok(err);
  assert.match(err.message, /"i" twice/);
});

test('#13 subscript-length-mismatch (too short): fix pads with a fresh label', () => {
  const state = stateFor([v({ rank: 3 })], 'ia', 'ia', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.SUBSCRIPT_LENGTH_MISMATCH);
  assert.ok(err);
  assert.match(err.message, /rank 3/);
  assert.match(err.message, /"ia" has 2 labels/);
  assert.match(err.message, /add 1 more/);
  assert.equal(err.fix.label, 'Pad to "iab"');
  const next = applyFix(state, err);
  assert.equal(next.subscriptsStr, 'iab');
  const r2 = run(next);
  assert.equal(findError(r2, ERROR_CODES.SUBSCRIPT_LENGTH_MISMATCH), undefined);
});

test('#13 subscript-length-mismatch (too long): message, no fix', () => {
  const state = stateFor([v({ rank: 2 })], 'ijk', 'ij', 'T');
  const err = findError(run(state), ERROR_CODES.SUBSCRIPT_LENGTH_MISMATCH);
  assert.ok(err);
  assert.match(err.message, /drop 1 label/);
  assert.equal(err.fix, undefined);
});

test('#14 output-non-lowercase: fix cleans the output', () => {
  const state = stateFor([v({ rank: 2 })], 'ij', 'IJ9', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.OUTPUT_NON_LOWERCASE);
  assert.ok(err);
  assert.equal(err.fix.label, 'Clean to "ij"');
  const next = applyFix(state, err);
  assert.equal(next.outputStr, 'ij');
});

test('#15 output-label-missing: fix removes the offending label', () => {
  const state = stateFor([v({ rank: 2 })], 'ij', 'ijc', 'T');
  const r = run(state);
  const err = findError(r, ERROR_CODES.OUTPUT_LABEL_MISSING);
  assert.ok(err);
  assert.match(err.message, /"c" doesn't appear in any input/);
  assert.equal(err.fix.label, 'Remove "c"');
  const next = applyFix(state, err);
  assert.equal(next.outputStr, 'ij');
});

// ── Field plumbing ───────────────────────────────────────────────────

test('fields are stable per rule so the UI can suppress by touch', () => {
  // Trigger a variety of errors across different fields.
  const r = validateAll(
    [
      v({ name: '' }),                                               // var-0-name
      v({ name: 'T', rank: 2, symmetry: 'cyclic', symAxes: [0, 5] }), // var-1-axes
      v({ name: 'X', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '' }), // var-2-generators
    ],
    'ia, jk, ijk',
    'xyz',
    'Q',
  );
  const fields = new Set(r.errors.map((e) => e.field));
  for (const expected of ['var-0-name', 'var-1-axes', 'var-2-generators', 'subscripts', 'operands', 'output']) {
    assert.ok(fields.has(expected), `expected field "${expected}" among ${[...fields].join(', ')}`);
  }
});
