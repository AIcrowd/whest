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

test('TypedPartitionDemo uses the editorial subsection style (ExplorerSubsectionHeader, no card wrapper)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /import ExplorerSubsectionHeader/);
  assert.match(src, /<ExplorerSubsectionHeader anchorId="typed-partition-demo"/);
  assert.match(src, /id="typed-partition-demo" className="bg-white p-4 scroll-mt-sticky"/);
  assert.doesNotMatch(src, /rounded-xl border bg-white px-6 py-6 shadow-sm/);
});

test('TypedPartitionDemo no longer carries the relocated prose paragraphs (they live back in section4 intro)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.doesNotMatch(src, /When projection branches, the explorer can count exactly without expanding the full assignment grid/);
  assert.doesNotMatch(src, /The partition-counting method groups assignments by equality pattern/);
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
  assert.match(src, /data-testid="partition-workbench"/);
  assert.match(src, /min-\[1180px\]:grid-cols-\[minmax\(0,1fr\)_minmax\(300px,0\.72fr\)\]/);
  assert.match(src, /data-testid="partition-breakdown-panel"/);
  assert.match(src, /Block structure/i);
  assert.match(src, /Projection reach/i);
});

test('TypedPartitionDemo renders the cumulative table and α total', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /data-testid="partition-cumulative-table"/);
  assert.match(src, /data-testid="partition-alpha-total"/);
});

test('TypedPartitionDemo emits captions only for partitionCount + bruteForceOrbit cases', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /partitionCount/);
  assert.match(src, /the partition budget is exceeded/);
  assert.match(src, /bruteForceOrbit/);
  // The functionalProjection / closed-form caption was noisy; it has been removed.
  assert.doesNotMatch(src, /partition counting would give the same/);
});

test('TypedPartitionDemo limits visible chips and table rows with a +N more affordance', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx');
  assert.match(src, /VISIBLE_LIMIT\s*=\s*8/);
  assert.match(src, /data-action="toggle-all-chips"/);
  assert.match(src, /data-action="toggle-all-rows"/);
  assert.match(src, /more patterns/);
  assert.match(src, /more rows/);
});
