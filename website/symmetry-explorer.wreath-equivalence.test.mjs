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
