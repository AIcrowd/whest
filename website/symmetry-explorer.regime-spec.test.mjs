import test from 'node:test';
import assert from 'node:assert/strict';
import { REGIME_SPEC, REGIME_PRIORITY } from './components/symmetry-aware-einsum-contractions/engine/regimeSpec.js';
import { MIXED_REGIMES } from './components/symmetry-aware-einsum-contractions/engine/regimes/index.js';
import { SHAPE_SPEC } from './components/symmetry-aware-einsum-contractions/engine/shapeSpec.js';
import { notationLatex } from './components/symmetry-aware-einsum-contractions/lib/notationSystem.js';

test('every regime in MIXED_REGIMES has a REGIME_SPEC entry', () => {
  for (const r of MIXED_REGIMES) {
    assert.ok(REGIME_SPEC[r.id], `missing REGIME_SPEC[${r.id}]`);
  }
});

test('REGIME_PRIORITY order matches MIXED_REGIMES order', () => {
  assert.deepEqual(MIXED_REGIMES.map((r) => r.id), REGIME_PRIORITY);
});

test('each REGIME_SPEC entry has label, shortLabel, when, latex, description', () => {
  for (const [id, spec] of Object.entries(REGIME_SPEC)) {
    for (const field of ['label', 'shortLabel', 'when', 'latex', 'description']) {
      assert.ok(spec[field], `${id}.${field} missing`);
    }
  }
});

test('SHAPE_SPEC has entries for trivial/allVisible/allSummed/mixed with required fields', () => {
  const required = ['label', 'shortLabel', 'description', 'latex', 'when'];
  for (const kind of ['trivial', 'allVisible', 'allSummed', 'mixed']) {
    assert.ok(SHAPE_SPEC[kind], `missing SHAPE_SPEC[${kind}]`);
    for (const f of required) {
      assert.ok(SHAPE_SPEC[kind][f], `SHAPE_SPEC[${kind}].${f} missing`);
    }
  }
});

test('mixed shape spec uses notation-aware inline math in its prose fields', () => {
  assert.ok(SHAPE_SPEC.mixed.description.includes(`$${notationLatex('v_free')}$`));
  assert.ok(SHAPE_SPEC.mixed.description.includes(`$${notationLatex('w_summed')}$`));
  assert.ok(SHAPE_SPEC.mixed.when.includes(`$${notationLatex('v_free')}$`));
  assert.ok(SHAPE_SPEC.mixed.when.includes(`$${notationLatex('w_summed')}$`));
});

test('directProduct uses the shorter preset-facing label', () => {
  assert.equal(REGIME_SPEC.directProduct.label, 'Direct Product');
});
