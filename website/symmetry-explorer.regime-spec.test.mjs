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

test('mixed shape tooltip formula is line-broken for narrow panels', () => {
  assert.match(SHAPE_SPEC.mixed.latex, /\\begin\{aligned\}/);
  assert.match(SHAPE_SPEC.mixed.latex, /\\\\/);
});

test('functionalProjection covers the no-cross / setwise-V cases that used to fire directProduct', () => {
  assert.equal(REGIME_SPEC.functionalProjection.label, 'One destination per product orbit');
});

test('partitionCount fires after young in the priority list', () => {
  const youngIdx = REGIME_PRIORITY.indexOf('young');
  const partitionIdx = REGIME_PRIORITY.indexOf('partitionCount');
  assert.ok(youngIdx >= 0);
  assert.ok(partitionIdx > youngIdx);
});

test('singleton spec routes c_Ω(g) through the shared notation registry', () => {
  assert.ok(REGIME_SPEC.singleton.latex.includes(notationLatex('c_omega_cycles')));
  assert.equal(REGIME_SPEC.singleton.glossary.some((entry) => entry.term === notationLatex('c_omega_cycles')), true);
});
