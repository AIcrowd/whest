#!/usr/bin/env node
/**
 * Cross-validation test: runs the JS algorithm engine on preset examples
 * AND comprehensive corner cases, comparing detected groups with expected
 * Python results.
 *
 * Usage:
 *   node test-cross-validate.mjs                  # run all
 *   node test-cross-validate.mjs --verbose         # show details
 *
 * Exit code 0 = all match, 1 = mismatch found.
 */

import { buildBipartite, buildIncidenceMatrix, runSigmaLoop, buildGroup } from './src/engine/algorithm.js';
import { EXAMPLES } from './src/data/examples.js';

const verbose = process.argv.includes('--verbose');

// ── Helpers ──

function normalizeExample(example) {
  if (Array.isArray(example.subscripts)) return example;
  const { expression, variables } = example;
  if (!expression) return example;
  const subsArr = expression.subscripts.split(',').map(s => s.trim());
  const opsArr = expression.operandNames.split(',').map(s => s.trim());
  const perOpSymmetry = opsArr.map(opName => {
    const v = variables.find(v => v.name === opName);
    if (!v || v.symmetry === 'none') return null;
    const axes = v.symAxes || [...Array(v.rank).keys()];
    if (v.symmetry === 'symmetric' && axes.length === v.rank) return 'symmetric';
    return { type: v.symmetry, axes };
  });
  const hasAnySym = perOpSymmetry.some(s => s !== null);
  return {
    ...example,
    subscripts: subsArr,
    output: expression.output,
    operandNames: opsArr,
    perOpSymmetry: hasAnySym ? perOpSymmetry : null,
  };
}

function detectGroup(example) {
  const norm = normalizeExample(example);
  const graph = buildBipartite(norm);
  const matrixData = buildIncidenceMatrix(graph);
  const sigmaResults = runSigmaLoop(graph, matrixData, norm);
  const group = buildGroup(sigmaResults, graph, norm);

  if (group.vGroupName !== 'trivial') return group.vGroupName;
  if (group.wGroupName && group.wGroupName !== 'trivial') return `W: ${group.wGroupName}`;
  return 'trivial';
}

/** Build an inline example object (algorithm-compatible format). */
function makeExample(subscripts, output, operandNames, perOpSymmetry = null) {
  const subsArr = subscripts.split(',');
  const opsArr = operandNames.split(',').map(s => s.trim());
  return {
    subscripts: subsArr,
    output,
    operandNames: opsArr,
    perOpSymmetry,
  };
}

// ── Test runner ──

let passed = 0;
let failed = 0;
const failures = [];

function test(name, example, expected) {
  try {
    const detected = detectGroup(example);
    if (detected === expected) {
      passed++;
      if (verbose) console.log(`  PASS  ${name}: ${detected}`);
    } else {
      failed++;
      failures.push({ name, expected, detected });
      console.log(`  FAIL  ${name}: expected "${expected}", got "${detected}"`);
    }
  } catch (err) {
    failed++;
    failures.push({ name, expected, detected: `ERROR: ${err.message}` });
    console.log(`  ERROR ${name}: ${err.message}`);
    if (verbose) console.log(`        ${err.stack}`);
  }
}

// ══════════════════════════════════════════════════════════════
// Part 1: Preset examples from examples.js
// ══════════════════════════════════════════════════════════════

console.log('=== Preset Examples ===');

const PRESET_EXPECTED = {
  'gram':         'S2{a,b}',
  'triple-outer': 'S3{a,b,c}',
  'outer':        'S2{a,c}\u00d7S2{b,d}',
  'triangle':     'C3{i,j,k}',
  'four-cycle':   'D4{i,j,k,l}',
  'trace-product':'W: S2{i,j}',
  'declared-c3':  'C3{i,j,k}',
  'declared-d4':  'D4{i,j,k,l}',
  'frobenius':    'W: S2{i,j}',
  'matrix-chain': 'trivial',
  'mixed-chain':  'trivial',
};

for (const ex of EXAMPLES) {
  const expected = PRESET_EXPECTED[ex.id];
  if (expected === undefined) {
    if (verbose) console.log(`  SKIP  ${ex.name} (no expected value)`);
    continue;
  }
  test(`[preset] ${ex.name}`, ex, expected);
}

// ══════════════════════════════════════════════════════════════
// Part 2: Corner cases — identical operands
// ══════════════════════════════════════════════════════════════

console.log('\n=== Category 1: Identical operands ===');

test('A*A matmul (trivial)',
  makeExample('ij,jk', 'ik', 'A, A'),
  'trivial');

test('Gram matrix X^T X (S2)',
  makeExample('ia,ib', 'ab', 'X, X'),
  'S2{a,b}');

test('Vector outer v⊗v (S2)',
  makeExample('i,j', 'ij', 'v, v'),
  'S2{i,j}');

test('Triple outer v⊗v⊗v (S3)',
  makeExample('i,j,k', 'ijk', 'v, v, v'),
  'S3{i,j,k}');

test('Quad outer v⊗v⊗v⊗v (S4)',
  makeExample('i,j,k,l', 'ijkl', 'v, v, v, v'),
  'S4{i,j,k,l}');

test('Directed triangle (C3)',
  makeExample('ij,jk,ki', 'ijk', 'A, A, A'),
  'C3{i,j,k}');

test('Directed 4-cycle (C4)',
  makeExample('ij,jk,kl,li', 'ijkl', 'A, A, A, A'),
  'C4{i,j,k,l}');

test('Block outer A⊗A (order 2)',
  makeExample('ab,cd', 'abcd', 'X, X'),
  'S2{a,c}\u00d7S2{b,d}');

test('Hadamard A⊙A (trivial)',
  makeExample('ij,ij', 'ij', 'A, A'),
  'trivial');

// ══════════════════════════════════════════════════════════════
// Part 3: Declared per-operand symmetry (non-identical operands)
// ══════════════════════════════════════════════════════════════

console.log('\n=== Category 2: Declared symmetry (non-identical) ===');

test('S*W contraction (trivial)',
  makeExample('ij,jk', 'ik', 'S, W', ['symmetric', null]),
  'trivial');

test('C3 tensor contraction (C3)',
  makeExample('aijk,ab', 'ijkb', 'T, W',
    [{ type: 'cyclic', axes: [1, 2, 3] }, null]),
  'C3{i,j,k}');

test('D4 tensor contraction (D4)',
  makeExample('aijkl,ab', 'ijklb', 'T, W',
    [{ type: 'dihedral', axes: [1, 2, 3, 4] }, null]),
  'D4{i,j,k,l}');

// ══════════════════════════════════════════════════════════════
// Part 4: Declared symmetry + identical operands
// ══════════════════════════════════════════════════════════════

console.log('\n=== Category 3: Declared + identical ===');

test('S*S matmul (S2)',
  makeExample('ij,jk', 'ik', 'S, S', ['symmetric', 'symmetric']),
  'S2{i,k}');

test('Undirected 4-cycle D4',
  makeExample('ij,jk,kl,li', 'ijkl', 'S, S, S, S',
    ['symmetric', 'symmetric', 'symmetric', 'symmetric']),
  'D4{i,j,k,l}');

test('C3 self-contract (BUG CASE — must be trivial)',
  makeExample('ijk,jki', 'ik', 'T, T',
    [{ type: 'cyclic', axes: [0, 1, 2] }, { type: 'cyclic', axes: [0, 1, 2] }]),
  'trivial');

test('Sym matrix triangle (S3)',
  makeExample('ij,jk,ki', 'ijk', 'S, S, S',
    ['symmetric', 'symmetric', 'symmetric']),
  'S3{i,j,k}');

// ══════════════════════════════════════════════════════════════
// Part 5: W-side (summed index) symmetry
// ══════════════════════════════════════════════════════════════

console.log('\n=== Category 4: W-side symmetry ===');

test('Tr(A*A) — W: S2',
  makeExample('ij,ji', '', 'A, A'),
  'W: S2{i,j}');

test('Tr(A^3) — W: C3',
  makeExample('ij,jk,ki', '', 'A, A, A'),
  'W: C3{i,j,k}');

test('Tr(A^4) — W: C4',
  makeExample('ij,jk,kl,li', '', 'A, A, A, A'),
  'W: C4{i,j,k,l}');

test('Partial trace triangle (trivial)',
  makeExample('ij,jk,ki', 'i', 'A, A, A'),
  'trivial');

test('Frobenius inner product — W: S2 (Source C)',
  makeExample('ij,ij', '', 'A, A'),
  'W: S2{i,j}');

// ══════════════════════════════════════════════════════════════
// Part 6: Edge cases
// ══════════════════════════════════════════════════════════════

console.log('\n=== Category 5: Edge cases ===');

test('A*B*A mixed chain (trivial)',
  makeExample('ij,jk,kl', 'il', 'A, B, A'),
  'trivial');

test('A*B*A*B alternating (trivial)',
  makeExample('ij,jk,kl,lm', 'im', 'A, B, A, B'),
  'trivial');

test('Diagonal extraction (trivial)',
  makeExample('iij', 'ij', 'D'),
  'trivial');

// ══════════════════════════════════════════════════════════════
// Summary
// ══════════════════════════════════════════════════════════════

console.log(`\n${'='.repeat(50)}`);
console.log(`${passed} passed, ${failed} failed out of ${passed + failed} tests`);

if (failures.length > 0) {
  console.log('\nFailures:');
  for (const f of failures) {
    console.log(`  ${f.name}: expected "${f.expected}", got "${f.detected}"`);
  }
  process.exit(1);
} else {
  console.log('All JS results match Python expected values.');
  process.exit(0);
}
