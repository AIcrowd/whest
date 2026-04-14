#!/usr/bin/env node
/**
 * Cross-validation test: runs the JS algorithm engine on all preset examples
 * and compares detected groups with expected Python results.
 *
 * Usage:
 *   node test-cross-validate.mjs                  # run all
 *   node test-cross-validate.mjs --verbose         # show details
 *   python -m pytest ... && node test-cross-validate.mjs  # run after Python tests
 *
 * Exit code 0 = all match, 1 = mismatch found.
 */

import { buildBipartite, buildIncidenceMatrix, runSigmaLoop, buildGroup } from './src/engine/algorithm.js';
import { EXAMPLES } from './src/data/examples.js';

const verbose = process.argv.includes('--verbose');

/**
 * Convert new example format (with variables/expression) to algorithm-compatible
 * format (with subscripts/output/operandNames/perOpSymmetry arrays).
 * Mirrors normalizeExample in App.jsx.
 */
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

/**
 * Run the full pipeline on an example and return the detected group name.
 */
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

// Expected results — must match Python output.
// These are verified by running:
//   uv run python3 -c "import whest as we; ..."
// See tests/test_subgraph_symmetry.py for the Python-side tests.
const EXPECTED = {
  'gram':         'S2{a,b}',
  'triple-outer': 'S3{a,b,c}',
  'outer':        'S2{a,c}\u00d7S2{b,d}',   // × = \u00d7
  'triangle':     'C3{i,j,k}',
  'four-cycle':   'D4{i,j,k,l}',
  'trace-product':'W: S2{i,j}',
  'declared-c3':  'C3{i,j,k}',
  'declared-d4':  'D4{i,j,k,l}',
  'matrix-chain': 'trivial',
  'mixed-chain':  'trivial',
};

// Run
let passed = 0;
let failed = 0;
const failures = [];

for (const ex of EXAMPLES) {
  const expected = EXPECTED[ex.id];
  if (expected === undefined) {
    console.log(`  SKIP  ${ex.name} (no expected value)`);
    continue;
  }

  try {
    const detected = detectGroup(ex);
    if (detected === expected) {
      passed++;
      if (verbose) console.log(`  PASS  ${ex.name}: ${detected}`);
    } else {
      failed++;
      failures.push({ name: ex.name, expected, detected });
      console.log(`  FAIL  ${ex.name}: expected "${expected}", got "${detected}"`);
    }
  } catch (err) {
    failed++;
    failures.push({ name: ex.name, expected, detected: `ERROR: ${err.message}` });
    console.log(`  ERROR ${ex.name}: ${err.message}`);
  }
}

console.log(`\n${passed} passed, ${failed} failed out of ${passed + failed} examples`);

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
