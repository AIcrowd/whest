// website/symmetry-explorer.wreath-equivalence.test.mjs
//
// Regression: |sigmaResults| must equal the wreath product order
// ∏_i |H_i|^{m_i} · m_i! for every preset. Guards against future drift
// if σ-loop emission ever deviates from the wreath.

import test from 'node:test';
import assert from 'node:assert/strict';

import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { parseCycleNotation } from './components/symmetry-aware-einsum-contractions/engine/cycleParser.js';

function factorial(n) { let f = 1; for (let i = 2; i <= n; i++) f *= i; return f; }

function declaredGroupSize(variable, rank) {
  if (!variable || variable.symmetry === 'none') return 1;
  const { symmetry, symAxes, generators } = variable;
  const axes = symAxes || Array.from({ length: rank }, (_, i) => i);
  const nAxes = axes.length;
  if (symmetry === 'symmetric') return factorial(nAxes);
  if (symmetry === 'cyclic') return nAxes;
  if (symmetry === 'dihedral') return nAxes >= 3 ? 2 * nAxes : nAxes;
  if (symmetry === 'custom') {
    const parsed = parseCycleNotation(generators || '');
    const gens = (parsed.generators || []).map((perm) => {
      const arr = Array.from({ length: rank }, (_, i) => i);
      for (const cycle of perm) {
        for (let k = 0; k < cycle.length; k++) arr[cycle[k]] = cycle[(k + 1) % cycle.length];
      }
      return new Permutation(arr);
    });
    if (gens.length === 0) return 1;
    return dimino(gens).length;
  }
  return 1;
}

function predictedWreathOrder(preset) {
  const { expression, variables = [] } = preset;
  const opNames = expression.operandNames.split(',').map((s) => s.trim());
  const subscripts = expression.subscripts.split(',').map((s) => s.trim());
  const nameToPositions = new Map();
  for (let i = 0; i < opNames.length; i += 1) {
    const list = nameToPositions.get(opNames[i]) || [];
    list.push(i);
    nameToPositions.set(opNames[i], list);
  }
  let product = 1;
  for (const [name, positions] of nameToPositions) {
    const m = positions.length;
    const rank = subscripts[positions[0]].length;
    const variable = variables.find((v) => v.name === name);
    const hSize = declaredGroupSize(variable, rank);
    product *= Math.pow(hSize, m) * factorial(m);
  }
  return product;
}

test('|sigmaResults| equals the wreath order ∏_i |H_i|^{m_i} · m_i! on every preset at n=3', () => {
  for (const preset of EXAMPLES) {
    const predicted = predictedWreathOrder(preset);
    const r = analyzeExample(preset, 3);
    const actual = r.sigmaResults.length;
    assert.equal(
      actual,
      predicted,
      `${preset.id}: expected ${predicted} row-perms, got ${actual}`,
    );
  }
});

import { enumerateWreath } from './components/symmetry-aware-einsum-contractions/engine/wreath.js';

test('enumerateWreath emits the correct count on representative structures', () => {
  // Single copy, no declared symmetry: wreath = trivial
  const trivial = [...enumerateWreath({
    identicalGroups: [[0]],
    perOpSymmetry: [null],
    axisRanks: [2],
  })];
  assert.equal(trivial.length, 1, 'single-copy trivial-sym wreath has order 1');

  // Two identical copies, no declared symmetry: wreath = {e} ≀ S_2 = S_2 (order 2)
  const identicalPair = [...enumerateWreath({
    identicalGroups: [[0, 1]],
    perOpSymmetry: [null, null],
    axisRanks: [2, 2],
  })];
  assert.equal(identicalPair.length, 2, '{e} ≀ S_2 has order 2');

  // One rank-2 symmetric operand, single copy: wreath = S_2 ≀ S_1 = S_2 (order 2)
  const oneSym = [...enumerateWreath({
    identicalGroups: [[0]],
    perOpSymmetry: ['symmetric'],
    axisRanks: [2],
  })];
  assert.equal(oneSym.length, 2, 'S_2 ≀ S_1 has order 2');

  // Four rank-2 symmetric identical copies: wreath = S_2 ≀ S_4 = 2^4 · 24 = 384
  const fourCycleLike = [...enumerateWreath({
    identicalGroups: [[0, 1, 2, 3]],
    perOpSymmetry: ['symmetric', 'symmetric', 'symmetric', 'symmetric'],
    axisRanks: [2, 2, 2, 2],
  })];
  assert.equal(fourCycleLike.length, 384, 'S_2 ≀ S_4 has order 384');
});

test('enumerateWreath first emitted element is the identity', () => {
  const [first] = [...enumerateWreath({
    identicalGroups: [[0, 1]],
    perOpSymmetry: [null, null],
    axisRanks: [2, 2],
  })];
  // Identity on a 4-vertex universe: [0,1,2,3]
  assert.deepEqual(first.rowPerm.arr, [0, 1, 2, 3], 'first element must be identity');
});

test('enumerateH accepts both raw-string and pre-parsed custom.generators', () => {
  // Raw string form — the shape the wreath-equivalence loop test builds
  const fromString = [...enumerateWreath({
    identicalGroups: [[0]],
    perOpSymmetry: [{ type: 'custom', axes: [0, 1, 2, 3], generators: '(0 1), (2 3)' }],
    axisRanks: [4],
  })];

  // Pre-parsed array form — the shape `pipeline.js::analyzeExample` produces
  // after `parseCycleNotation`. Same group ⟨(0 1), (2 3)⟩ ≅ S_2 × S_2, |G| = 4.
  const fromArray = [...enumerateWreath({
    identicalGroups: [[0]],
    perOpSymmetry: [{ type: 'custom', axes: [0, 1, 2, 3], generators: [[[0, 1]], [[2, 3]]] }],
    axisRanks: [4],
  })];

  assert.equal(fromString.length, 4, '⟨(0 1), (2 3)⟩ has order 4 (raw string)');
  assert.equal(fromArray.length, 4, '⟨(0 1), (2 3)⟩ has order 4 (pre-parsed)');
  assert.equal(fromString.length, fromArray.length, 'both shapes yield the same count');
});
