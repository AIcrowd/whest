import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) { return readFileSync(resolve(__dirname, rel), 'utf-8'); }

test('TypedPartitionDemo exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /export default function TypedPartitionDemo/);
});

test('TypedPartitionDemo uses theme helpers and contains zero raw hex codes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /explorerThemeColor|explorerThemeTint|notationColor/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
});

test('TypedPartitionDemo carries the relocated section4 ¶4 prose verbatim', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /When projection branches, the explorer can count exactly without expanding the full assignment grid/);
});

test('TypedPartitionDemo carries the relocated PartitionCountingExplainer body ¶3, ¶4, ¶5 prose verbatim', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /The partition-counting method groups assignments by equality pattern/);
  assert.match(src, /For a fixed typed equality pattern, the block-labeling factor counts how many product orbits live above that pattern/);
  assert.match(src, /This is not an approximation\. When both are feasible, the typed partition count and corrected brute-force orbit enumeration must give the same/);
});

test('TypedPartitionDemo imports the engine partition exports', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /generateTypedSetPartitions/);
  assert.match(src, /partitionOrbitReps/);
  assert.match(src, /typedLabelingCount/);
  assert.match(src, /inducedBlockActionSize/);
  assert.match(src, /inducedPrefixMaps/);
});

test('TypedPartitionDemo renders pattern chips area with chip-id attributes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /data-pattern-chip/);
  assert.match(src, /Equality pattern/i);
});

test('TypedPartitionDemo renders a 2-column breakdown panel for the selected pattern', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /data-testid="partition-breakdown-panel"/);
  assert.match(src, /Block structure/i);
  assert.match(src, /Projection reach/i);
});

test('TypedPartitionDemo renders the cumulative table and α total', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /data-testid="partition-cumulative-table"/);
  assert.match(src, /data-testid="partition-alpha-total"/);
});

test('TypedPartitionDemo handles the four regime coverage cases with explicit captions', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /partitionCount/);
  assert.match(src, /this component fires/);
  assert.match(src, /the partition budget is exceeded/);
});
