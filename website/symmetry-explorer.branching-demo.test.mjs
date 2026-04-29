import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('BranchingDemo exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /export default function BranchingDemo/);
});

test('BranchingDemo uses theme helpers and contains zero raw hex codes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /explorerThemeColor|explorerThemeTint|notationColor/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
});

test('BranchingDemo uses the editorial subsection style (ExplorerSubsectionHeader, no card wrapper)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /import ExplorerSubsectionHeader/);
  assert.match(src, /<ExplorerSubsectionHeader anchorId="branching-demo"/);
  assert.match(src, /id="branching-demo" className="bg-white p-4 scroll-mt-24"/);
  assert.doesNotMatch(src, /rounded-xl border bg-white px-6 py-6 shadow-sm/);
});

test('BranchingDemo no longer renders an orbit dropdown or prev/next (selection happens in the matrix)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.doesNotMatch(src, /data-action="select-orbit"/);
  assert.doesNotMatch(src, /data-action="prev-orbit"/);
  assert.doesNotMatch(src, /data-action="next-orbit"/);
});

test('BranchingDemo wires OrbitRepMatrix instead of the old fan', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /import OrbitRepMatrix from '\.\/branchingViews\/OrbitRepMatrix\.jsx'/);
  assert.match(src, /<OrbitRepMatrix/);
  // Old fan + view-mode tabs are gone.
  assert.doesNotMatch(src, /OrbitProjectionGraph/);
  assert.doesNotMatch(src, /data-view-id=/);
});

test('BranchingDemo no longer carries the relocated prose paragraphs (they live back in section4 intro)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.doesNotMatch(src, /A product-orbit representative can contain many full index assignments/);
  assert.doesNotMatch(src, /A product orbit may contain many full assignments\./);
  assert.doesNotMatch(src, /Counting product orbits alone is therefore not enough/);
});

test('BranchingDemo derives orbit data from costModel.orbitRows', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /costModel\.orbitRows/);
});

test('BranchingDemo offers a curated fallback example with a toggle', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /CURATED_BRANCHING_EXAMPLE/);
  assert.match(src, /data-action="toggle-curated"/);
  assert.match(src, /R\[i,j\] = \\sum_k T\[i,j,k\]/);
});

test('BranchingDemo emits a live α total via data-testid="branching-alpha-total"', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /data-testid="branching-alpha-total"/);
});

test('OrbitRepMatrix exports a default React component with no raw hex', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /export default function OrbitRepMatrix/);
  assert.match(src, /explorerThemeColor|explorerThemeTint|notationColor/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
});

test('OrbitRepMatrix renders the "How to read this matrix" reading guide', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /data-testid="orbit-rep-matrix-reading-guide"/);
  assert.match(src, /How to read this matrix/);
  // The four bullets on rows / columns / filled cells / multi-cell rows
  // (use \s+ to accept JSX-wrapped whitespace between words).
  assert.match(src, /Rows[\s\S]*?product\s+orbits/);
  assert.match(src, /Columns[\s\S]*?stored\s+output\s+representatives/);
  assert.match(src, /Filled cell/);
  assert.match(src, /Multi-cell rows/);
});

test('OrbitRepMatrix wraps the table in PanZoomCanvas for zoom + pan', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /import PanZoomCanvas from '\.\.\/PanZoomCanvas\.jsx'/);
  assert.match(src, /<PanZoomCanvas/);
});

test('OrbitRepMatrix uses the labelled tuple format on hover (k=v style)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  // Must produce the (i=0, j=0, k=1) form for tooltips, matching
  // OrbitInspector.formatTuple's contract.
  assert.match(src, /labelledTuple/);
  assert.match(src, /\$\{k\}=\$\{v\}/);
});

test('OrbitRepMatrix derivation: collects unique reps + builds the cell grid', async () => {
  // The component is JSX so we can't import it here, but the source-grep
  // assertions above plus the visual-smoke check in the dev preview cover
  // the structural contract. This sentinel test just confirms the
  // derivation references the right field names.
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /orbitRows\.forEach/);
  assert.match(src, /row\.outputs/);
  assert.match(src, /tupleKey\(out\.outTuple\)/);
});

test('OrbitRepMatrix renders a label-position legend over actual label names', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /rowLabels/);
  assert.match(src, /colLabels/);
  assert.match(src, /orbit reps over/);
  assert.match(src, /stored reps over/);
});

test('OrbitRepMatrix renders a hover-cell detail panel with rich orbit-rep info', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /data-testid="orbit-rep-matrix-detail"/);
  // Eyebrow + labelled-tuple + status copy.
  assert.match(src, /Focus[\s\S]*?cell/);
  assert.match(src, /orbit&nbsp;rep:/);
  assert.match(src, /stored&nbsp;rep:/);
  assert.match(src, /contributes 1 to α/);
  assert.match(src, /does not contribute/);
  // Branching annotation.
  assert.match(src, /This orbit branches/);
});
