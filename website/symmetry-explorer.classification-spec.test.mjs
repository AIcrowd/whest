import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CLASSIFICATION_LEAVES,
  CLASSIFICATION_QUESTIONS,
  classifyComponent,
} from './components/symmetry-aware-einsum-contractions/engine/classificationSpec.js';

import {
  decomposeAndClassify,
} from './components/symmetry-aware-einsum-contractions/engine/componentDecomposition.js';

import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';

// ─── Spec shape ──────────────────────────────────────────────────

test('CLASSIFICATION_LEAVES defines all six leaves with caseType and label', () => {
  assert.deepEqual(
    Object.keys(CLASSIFICATION_LEAVES).sort(),
    ['a', 'b', 'c', 'd', 'e', 'trivial'],
  );

  for (const leaf of Object.values(CLASSIFICATION_LEAVES)) {
    assert.equal(typeof leaf.id, 'string');
    assert.equal(typeof leaf.caseType, 'string');
    assert.equal(typeof leaf.label, 'string');
  }

  assert.equal(CLASSIFICATION_LEAVES.trivial.caseType, 'trivial');
  assert.equal(CLASSIFICATION_LEAVES.a.caseType, 'A');
  assert.equal(CLASSIFICATION_LEAVES.b.caseType, 'B');
  assert.equal(CLASSIFICATION_LEAVES.c.caseType, 'C');
  assert.equal(CLASSIFICATION_LEAVES.d.caseType, 'D');
  assert.equal(CLASSIFICATION_LEAVES.e.caseType, 'E');
});

test('each leaf\'s latex formula matches the mathematics the engine actually computes', () => {
  // Trivial: |I_a/G_a| = |I_a| when G_a = {e}
  assert.match(CLASSIFICATION_LEAVES.trivial.latex, /I_a\s*\/\s*G_a/);
  assert.match(CLASSIFICATION_LEAVES.trivial.latex, /G_a\s*=\s*\\\{e\\\}/);

  // A: ρ_a = Π_{ℓ∈V_a} n_ℓ
  assert.match(CLASSIFICATION_LEAVES.a.latex, /\\rho_a/);
  assert.match(CLASSIFICATION_LEAVES.a.latex, /\\prod/);
  assert.match(CLASSIFICATION_LEAVES.a.latex, /V_a/);
  assert.match(CLASSIFICATION_LEAVES.a.latex, /n_\\ell/);

  // B: ρ_a = |I_a/G_a| (Burnside)
  assert.match(CLASSIFICATION_LEAVES.b.latex, /\\rho_a/);
  assert.match(CLASSIFICATION_LEAVES.b.latex, /I_a\s*\/\s*G_a/);
  assert.match(CLASSIFICATION_LEAVES.b.latex, /Burnside/);

  // C/E: ρ_a = Σ_{O∈I_a/G_a} |π_{V_a}(O)|
  for (const leafId of ['c', 'e']) {
    const leaf = CLASSIFICATION_LEAVES[leafId];
    assert.match(leaf.latex, /\\rho_a/, `${leafId} must define ρ_a`);
    assert.match(leaf.latex, /\\sum/, `${leafId} must be a sum over orbits`);
    assert.match(leaf.latex, /I_a\s*\/\s*G_a/, `${leafId} must iterate orbits of G_a`);
    assert.match(leaf.latex, /\\pi_\{V_a\}/, `${leafId} must project onto V_a`);
  }

  // D: ρ_a = |I_a/H_a|, H_a = Stab_{G_a}(V_a)
  assert.match(CLASSIFICATION_LEAVES.d.latex, /\\rho_a/);
  assert.match(CLASSIFICATION_LEAVES.d.latex, /I_a\s*\/\s*H_a/);
  assert.match(CLASSIFICATION_LEAVES.d.latex, /H_a\s*=\s*\\mathrm\{Stab\}_\{G_a\}\(V_a\)/);
});

test('every leaf carries latex + glossary (tooltip contract)', () => {
  for (const leaf of Object.values(CLASSIFICATION_LEAVES)) {
    assert.equal(typeof leaf.latex, 'string', `leaf ${leaf.id} must have latex`);
    assert.equal(
      typeof leaf.glossary,
      'string',
      `leaf ${leaf.id} must have glossary (distil-style variable legend)`,
    );
    assert.ok(
      leaf.glossary.length >= 40,
      `leaf ${leaf.id} glossary must actually explain the symbols (got ${leaf.glossary.length} chars)`,
    );
    // Glossary must use $...$ markers for math so symbols render with the
    // same KaTeX font as the displayed formula (distill convention).
    const dollarCount = (leaf.glossary.match(/\$/g) ?? []).length;
    assert.equal(
      dollarCount % 2,
      0,
      `leaf ${leaf.id} glossary has unbalanced $ markers (${dollarCount} total)`,
    );
    assert.ok(
      dollarCount >= 4,
      `leaf ${leaf.id} glossary must include at least two inline math segments (got ${dollarCount / 2})`,
    );
  }
});

test('CLASSIFICATION_QUESTIONS defines q0..q4 with tests and targets', () => {
  assert.deepEqual(
    CLASSIFICATION_QUESTIONS.map((q) => q.id),
    ['q0', 'q1', 'q2', 'q3', 'q4'],
  );

  for (const q of CLASSIFICATION_QUESTIONS) {
    assert.equal(typeof q.short, 'string');
    assert.equal(typeof q.long, 'string');
    assert.equal(typeof q.test, 'function');
    assert.ok(q.onTrue, `${q.id} must have onTrue`);
    assert.ok(q.onFalse, `${q.id} must have onFalse`);
  }
});

// ─── classifyComponent — every leaf is reachable ─────────────────

test('classifyComponent routes trivial (order=1) to the trivial leaf', () => {
  const result = classifyComponent({
    order: 1,
    vCount: 2,
    wCount: 1,
    hasCrossGen: false,
    isFullSym: false,
    labelCount: 3,
  });
  assert.equal(result.caseType, 'trivial');
  assert.equal(result.leaf, 'trivial');
  assert.deepEqual(result.path, ['q0', 'trivial']);
});

test('classifyComponent routes W-empty (V-only) to Case A', () => {
  const result = classifyComponent({
    order: 6,
    vCount: 3,
    wCount: 0,
    hasCrossGen: false,
    isFullSym: true,
    labelCount: 3,
  });
  assert.equal(result.caseType, 'A');
  assert.deepEqual(result.path, ['q0', 'q1', 'a']);
});

test('classifyComponent routes V-empty (W-only) to Case B', () => {
  const result = classifyComponent({
    order: 2,
    vCount: 0,
    wCount: 2,
    hasCrossGen: false,
    isFullSym: true,
    labelCount: 2,
  });
  assert.equal(result.caseType, 'B');
  assert.deepEqual(result.path, ['q0', 'q1', 'q2', 'b']);
});

test('classifyComponent routes mixed + no cross-gens to Case C', () => {
  const result = classifyComponent({
    order: 4,
    vCount: 2,
    wCount: 2,
    hasCrossGen: false,
    isFullSym: false,
    labelCount: 4,
  });
  assert.equal(result.caseType, 'C');
  assert.deepEqual(result.path, ['q0', 'q1', 'q2', 'q3', 'c']);
});

test('classifyComponent routes cross-gens + full Sym to Case D', () => {
  const result = classifyComponent({
    order: 6,
    vCount: 1,
    wCount: 2,
    hasCrossGen: true,
    isFullSym: true,
    labelCount: 3,
  });
  assert.equal(result.caseType, 'D');
  assert.deepEqual(result.path, ['q0', 'q1', 'q2', 'q3', 'q4', 'd']);
});

test('classifyComponent routes cross-gens + partial group to Case E', () => {
  const result = classifyComponent({
    order: 3,
    vCount: 1,
    wCount: 2,
    hasCrossGen: true,
    isFullSym: false,
    labelCount: 3,
  });
  assert.equal(result.caseType, 'E');
  assert.deepEqual(result.path, ['q0', 'q1', 'q2', 'q3', 'q4', 'e']);
});

// ─── Engine integration: decomposeAndClassify uses the spec ─────

function componentFactsFromExample(exampleId) {
  const example = EXAMPLES.find((e) => e.id === exampleId);
  assert.ok(example, `example ${exampleId} must exist`);
  return example;
}

test('engine: trivial example "matrix-chain" yields only trivial components', () => {
  // einsum('ij,jk→ik', A, A) — no detected symmetry, no generators.
  // allLabels = ['i','j','k']; V={i,k}; W={j}
  const allLabels = ['i', 'j', 'k'];
  const vLabels = ['i', 'k'];
  const wLabels = ['j'];
  const { components } = decomposeAndClassify(allLabels, vLabels, wLabels, [], []);
  assert.ok(components.length >= 1);
  for (const comp of components) {
    assert.equal(comp.caseType, 'trivial', `component ${comp.labels.join(',')} should be trivial`);
    assert.ok(Array.isArray(comp.path));
    assert.equal(comp.path[comp.path.length - 1], 'trivial');
  }
});

test('engine: decomposeAndClassify emits a path on every component', () => {
  // Simulate a detected S2 on {a,b} acting only on output labels.
  const allLabels = ['a', 'b', 'i'];
  const vLabels = ['a', 'b'];
  const wLabels = ['i'];
  const swap = new Permutation([1, 0, 2]); // (a b), leaves i fixed
  const { components } = decomposeAndClassify(
    allLabels,
    vLabels,
    wLabels,
    [swap],
    [],
  );
  for (const comp of components) {
    assert.ok(Array.isArray(comp.path), `component ${comp.labels.join(',')} must have path`);
    assert.ok(comp.path.length >= 2, 'path should include at least q0 and a leaf');
    assert.equal(comp.path[0], 'q0', 'path always starts at q0');
  }

  // The {a,b} component (V-only with S2) should land on Case A, path q0 → q1 → a
  const abComp = components.find((c) => c.labels.includes('a') && c.labels.includes('b'));
  assert.ok(abComp, 'component {a,b} expected');
  assert.equal(abComp.caseType, 'A');
  assert.deepEqual(abComp.path, ['q0', 'q1', 'a']);

  // The {i} singleton has trivial group, lands on trivial.
  const iComp = components.find((c) => c.labels.length === 1 && c.labels[0] === 'i');
  assert.ok(iComp, 'singleton component {i} expected');
  assert.equal(iComp.caseType, 'trivial');
  assert.deepEqual(iComp.path, ['q0', 'trivial']);
});

// ─── Drift guard: engine and tree render from the same spec ────

test('drift guard: every caseType emitted by classifyComponent corresponds to a leaf', () => {
  const leafCaseTypes = new Set(Object.values(CLASSIFICATION_LEAVES).map((l) => l.caseType));
  const observedCaseTypes = [
    classifyComponent({ order: 1, vCount: 0, wCount: 0, hasCrossGen: false, isFullSym: false, labelCount: 1 }).caseType,
    classifyComponent({ order: 2, vCount: 2, wCount: 0, hasCrossGen: false, isFullSym: true, labelCount: 2 }).caseType,
    classifyComponent({ order: 2, vCount: 0, wCount: 2, hasCrossGen: false, isFullSym: true, labelCount: 2 }).caseType,
    classifyComponent({ order: 4, vCount: 2, wCount: 2, hasCrossGen: false, isFullSym: false, labelCount: 4 }).caseType,
    classifyComponent({ order: 6, vCount: 1, wCount: 2, hasCrossGen: true, isFullSym: true, labelCount: 3 }).caseType,
    classifyComponent({ order: 3, vCount: 1, wCount: 2, hasCrossGen: true, isFullSym: false, labelCount: 3 }).caseType,
  ];
  for (const ct of observedCaseTypes) {
    assert.ok(leafCaseTypes.has(ct), `caseType ${ct} must be in CLASSIFICATION_LEAVES`);
  }
  // All six distinct caseTypes are represented
  assert.equal(new Set(observedCaseTypes).size, 6);
});

test('drift guard: every question node is reachable from some facts vector', () => {
  const visitedQuestions = new Set();
  const scenarios = [
    { order: 1, vCount: 0, wCount: 0, hasCrossGen: false, isFullSym: false, labelCount: 1 },
    { order: 2, vCount: 2, wCount: 0, hasCrossGen: false, isFullSym: true, labelCount: 2 },
    { order: 2, vCount: 0, wCount: 2, hasCrossGen: false, isFullSym: true, labelCount: 2 },
    { order: 4, vCount: 2, wCount: 2, hasCrossGen: false, isFullSym: false, labelCount: 4 },
    { order: 6, vCount: 1, wCount: 2, hasCrossGen: true, isFullSym: true, labelCount: 3 },
    { order: 3, vCount: 1, wCount: 2, hasCrossGen: true, isFullSym: false, labelCount: 3 },
  ];
  for (const facts of scenarios) {
    const { path } = classifyComponent(facts);
    for (const step of path) {
      if (step.startsWith('q')) visitedQuestions.add(step);
    }
  }
  for (const q of CLASSIFICATION_QUESTIONS) {
    assert.ok(
      visitedQuestions.has(q.id),
      `question ${q.id} must be reachable from the spec (unused => drift)`,
    );
  }
});
