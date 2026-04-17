import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('MultiplicationCostCard exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx');
  assert.match(src, /export default function MultiplicationCostCard/);
});

test('MultiplicationCostCard shows both μ and M formulas + live rows', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx');
  // Heading uses the canonical μ notation from the Counting Convention band.
  assert.match(src, /Calculating Multiplication Cost \(μ\)/);
  assert.match(src, /Live for this example/i);
  // μ-formula ties the Burnside count M to the binary-multiply cost via (k−1).
  assert.match(src, /\\mu\s*\\;=\\;\s*\(/);
  // Mentions num_terms as the einsum-operand-count scaling factor.
  assert.match(src, /num_terms/);
  // And the raw Burnside sum below it.
  assert.match(src, /\\mathrm\{cycles\}\(g\)/);
  assert.match(src, /multiplicationCount/);
});

test('AccumulationHardCard exports a default React component with pointer text', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/AccumulationHardCard.jsx');
  assert.match(src, /export default function AccumulationHardCard/);
  // Title now uses α to match the Counting Convention band.
  assert.match(src, /Why Accumulation Cost \(α\) is Hard/);
  assert.match(src, /See the Classification Tree below/);
});

test('ComponentCostView imports the two new cards + renders the CLASSIFICATION TREE section', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /import MultiplicationCostCard/);
  assert.match(src, /import AccumulationHardCard/);
  assert.match(src, /Classification Tree/);
});

test('ComponentCostView passes activeLeafIds (all detected) to the DecisionLadder', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /activeLeafIds=/);
  // Should flatmap over components for union of regimeId + shape.
  assert.match(src, /c\.accumulation\?\.regimeId/);
  assert.match(src, /c\.shape/);
});
