import test from 'node:test';
import assert from 'node:assert/strict';

import { validateAll } from './components/symmetry-aware-einsum-contractions/engine/validation.js';

function validate(variable, subs = 'ijk', output = 'ijk', operands = variable.name) {
  return validateAll([variable], subs, output, operands);
}

test('validateAll accepts a valid custom cycle generator', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1)',
  });
  assert.deepEqual(result, { valid: true, errors: [] });
});

test('validateAll does not throw when custom symmetry has non-empty generators', () => {
  // Regression: previously passing {generators, error} to generatorIndices()
  // raised "TypeError: generators is not iterable" and crashed the explorer.
  assert.doesNotThrow(() =>
    validate({
      name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1)(1 2)',
    }),
  );
});

test('empty-generator hint is tailored to a 2-axis variable', () => {
  const result = validate({
    name: 'A', rank: 2, symmetry: 'custom', symAxes: [0, 1], generators: '',
  }, 'ij', 'ij');
  assert.equal(result.valid, false);
  assert.equal(result.errors.length, 1);
  assert.match(result.errors[0], /Variable "A":/);
  assert.match(result.errors[0], /\(0 1\)/);
  assert.match(result.errors[0], /two selected axes/);
});

test('empty-generator hint is tailored to a 3-axis variable', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /\(0 1 2\)/);
  assert.match(result.errors[0], /rotate all three/);
});

test('empty-generator hint for ≥4 axes shows a double-swap example', () => {
  const result = validate({
    name: 'T', rank: 4, symmetry: 'custom', symAxes: [0, 1, 2, 3], generators: '',
  }, 'ijkl', 'ijkl');
  assert.match(result.errors[0], /\(0 1\)\(2 3\)/);
});

test('out-of-range hint names the axes count and valid range', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1], generators: '(0 1)(2 3)',
  }, 'ij', 'ij', 'T');
  assert.equal(result.valid, false);
  const rangeErrors = result.errors.filter((e) => /is outside your/.test(e));
  assert.equal(rangeErrors.length, 2);
  for (const err of rangeErrors) {
    assert.match(err, /2 selected axes/);
    assert.match(err, /valid indices are 0–1/);
  }
});

test('parser "Unexpected characters" error becomes an in-parentheses hint', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /outside any cycle/);
  assert.match(result.errors[0], /\(0 1 2\)/);
});

test('parser "Empty cycle" error becomes a friendly hint', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '()',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /empty cycle/);
  assert.match(result.errors[0], /\(0 1 2\)/);
});

test('parser "Cycle too short" error keeps the offending cycle visible', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0)',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /\(0\) is too short/);
  assert.match(result.errors[0], /≥ 2 elements/);
});

test('parser "Invalid element" error points at the offending token', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(a b)',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /"a" isn't a valid index/);
  assert.match(result.errors[0], /0–2/);
});

test('parser "Duplicate within cycle" error names the repeated axis and cycle', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1 0)',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /axis 0 appears twice in \(0 1 0\)/);
});

test('parser "Duplicate across cycles" error suggests disjoint cycles', () => {
  const result = validate({
    name: 'T', rank: 3, symmetry: 'custom', symAxes: [0, 1, 2], generators: '(0 1)(2 0)',
  });
  assert.equal(result.valid, false);
  assert.match(result.errors[0], /axis 0 appears in more than one cycle/);
  assert.match(result.errors[0], /disjoint/);
});
