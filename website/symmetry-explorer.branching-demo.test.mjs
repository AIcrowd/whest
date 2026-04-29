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

test('BranchingDemo renders an orbit dropdown with reach hints, no prev/next buttons', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /data-action="select-orbit"/);
  assert.match(src, /reaches \{reach\}/);
  assert.doesNotMatch(src, /data-action="prev-orbit"/);
  assert.doesNotMatch(src, /data-action="next-orbit"/);
});

test('BranchingDemo wires OrbitProjectionGraph (drops the four old view tabs)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /import OrbitProjectionGraph from '\.\/branchingViews\/OrbitProjectionGraph\.jsx'/);
  assert.match(src, /<OrbitProjectionGraph/);
  assert.doesNotMatch(src, /data-view-id="fan"/);
  assert.doesNotMatch(src, /data-view-id="arcs"/);
  assert.doesNotMatch(src, /data-view-id="grids"/);
  assert.doesNotMatch(src, /data-view-id="pile-buckets"/);
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

test('OrbitProjectionGraph exports a default React component using @xyflow/react', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitProjectionGraph.jsx');
  assert.match(src, /export default function OrbitProjectionGraph/);
  assert.match(src, /from '@xyflow\/react'/);
  assert.match(src, /<ReactFlow/);
  assert.match(src, /ReactFlowProvider/);
});

test('OrbitProjectionGraph defines member, orbitCenter, rep node types and uses theme helpers only', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitProjectionGraph.jsx');
  assert.match(src, /function MemberNode/);
  assert.match(src, /function OrbitCenterNode/);
  assert.match(src, /function RepNode/);
  assert.match(src, /opgNodeTypes/);
  assert.match(src, /explorerThemeColor|notationColor/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
});

test('orbitProjectionLayout helper yields the right node + edge counts for the curated branching orbit', async () => {
  const { buildOrbitProjectionGraph } = await import('./components/symmetry-aware-einsum-contractions/components/branchingViews/orbitProjectionLayout.js');
  // Curated orbit 4 (i=0, j=0, k=1): orbit of size 3 reaching Q1 (×1) and Q2 (×2).
  const orbit = {
    size: 3,
    members: [{ repIndex: 0 }, { repIndex: 1 }, { repIndex: 1 }],
  };
  const reachedReps = [{ weight: 1 }, { weight: 2 }];
  const { nodes, edges } = buildOrbitProjectionGraph({ orbit, reachedReps, orbitIdx: 3, totalOrbits: 6, themeId: 'editorial-noir' });

  // 3 member + 1 orbit-center + 2 rep nodes = 6 nodes total.
  assert.equal(nodes.length, 6);
  assert.equal(nodes.filter((n) => n.type === 'member').length, 3);
  assert.equal(nodes.filter((n) => n.type === 'orbitCenter').length, 1);
  assert.equal(nodes.filter((n) => n.type === 'rep').length, 2);

  // 3 member->orbit edges + 2 orbit->rep edges = 5 edges total.
  assert.equal(edges.length, 5);
  assert.equal(edges.filter((e) => e.target === 'orbit-center').length, 3);
  assert.equal(edges.filter((e) => e.source === 'orbit-center').length, 2);
});

test('orbitProjectionLayout helper returns empty result when no orbit is selected', async () => {
  const { buildOrbitProjectionGraph } = await import('./components/symmetry-aware-einsum-contractions/components/branchingViews/orbitProjectionLayout.js');
  const { nodes, edges } = buildOrbitProjectionGraph({ orbit: null, reachedReps: [], orbitIdx: 0, totalOrbits: 0, themeId: 'editorial-noir' });
  assert.equal(nodes.length, 0);
  assert.equal(edges.length, 0);
});

test('OrbitProjectionGraph imports the pure layout helper from the .js sibling', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitProjectionGraph.jsx');
  assert.match(src, /buildOrbitProjectionGraph/);
  assert.match(src, /from '\.\/orbitProjectionLayout\.js'/);
});
