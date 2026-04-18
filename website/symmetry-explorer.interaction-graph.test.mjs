import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { buildLabelInteractionGraph } from './components/symmetry-aware-einsum-contractions/engine/componentDecomposition.js';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';

test('buildLabelInteractionGraph tags each edge with its generator index', () => {
  // Two generators on 4 labels. gen0 swaps (0 1), gen1 swaps (2 3).
  const gen0 = new Permutation([1, 0, 2, 3]);
  const gen1 = new Permutation([0, 1, 3, 2]);
  const { edges, components } = buildLabelInteractionGraph(
    ['a', 'b', 'c', 'd'],
    [gen0, gen1],
  );

  // Every edge is a 3-tuple with a valid generator index.
  for (const edge of edges) {
    assert.equal(edge.length, 3, 'edges should carry [from, to, generatorIdx]');
    const [, , genIdx] = edge;
    assert.ok(Number.isInteger(genIdx) && genIdx >= 0 && genIdx < 2,
      `generatorIdx ${genIdx} must be a valid index into generators`);
  }

  // Provenance: edges from gen0 only touch indices {0, 1}; edges from gen1
  // only touch {2, 3}. This is the contract the UI relies on for edge tooltips.
  for (const [from, to, genIdx] of edges) {
    if (genIdx === 0) {
      assert.ok([0, 1].includes(from) && [0, 1].includes(to),
        `gen0 edges must stay inside {0,1}, got ${from}->${to}`);
    } else if (genIdx === 1) {
      assert.ok([2, 3].includes(from) && [2, 3].includes(to),
        `gen1 edges must stay inside {2,3}, got ${from}->${to}`);
    }
  }

  // Components unchanged by the tagging change.
  const sortedComps = components.map((c) => [...c].sort()).sort((a, b) => a[0] - b[0]);
  assert.deepEqual(sortedComps, [[0, 1], [2, 3]]);
});

test('buildLabelInteractionGraph degrades gracefully with no generators', () => {
  const { edges, components } = buildLabelInteractionGraph(['a', 'b', 'c'], []);
  assert.deepEqual(edges, []);
  // Each label is its own component.
  assert.equal(components.length, 3);
});

test('buildLabelInteractionGraph legacy [a, b] destructuring still works', () => {
  // The LabelInteractionGraph renderer does `edges.map(([a, b]) => ...)`
  // and must keep working even though edges now carry a third element.
  const gen = new Permutation([1, 0]);
  const { edges } = buildLabelInteractionGraph(['x', 'y'], [gen]);
  assert.ok(edges.length > 0);
  for (const edge of edges) {
    const [a, b] = edge;
    assert.ok(typeof a === 'number' && typeof b === 'number');
  }
});

test('Cross-highlight wiring: App → StickyBar (halo) and App → DecisionLadder (spotlight)', () => {
  // This pins the *source-level contract* for the cross-highlight feature.
  // We can't drive real mouseover events in node:test, but we can verify the
  // wiring that makes the behavior possible.
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );
  const componentCostSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url),
    'utf8',
  );
  const graphSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/ComponentView.jsx', import.meta.url),
    'utf8',
  );
  const stickyBarSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/StickyBar.jsx', import.meta.url),
    'utf8',
  );
  const ladderSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx', import.meta.url),
    'utf8',
  );

  // App owns the hover state and forwards it to both consumers.
  assert.match(appSource, /const \[graphHover, setGraphHover\] = useState\(null\)/);
  assert.match(appSource, /hoveredLabels=\{hoveredLabelSet\}/);
  assert.match(appSource, /onGraphHover=\{handleGraphHover\}/);
  assert.match(appSource, /spotlightLeafIds=\{spotlightLeafSet\}/);

  // ComponentCostView plumbs both through to the right children.
  assert.match(componentCostSource, /onHover=\{onGraphHover\}/);
  assert.match(componentCostSource, /spotlightLeafIds=\{spotlightLeafIds\}/);

  // LabelInteractionGraph emits a payload with both label names and leaf keys.
  assert.match(graphSource, /buildHoverPayload/);
  assert.match(graphSource, /leafKeys/);
  // Hull payload carries shape and regimeId so the ladder can
  // match whichever leaf representation the component surfaces.
  assert.match(graphSource, /hull\.comp\.shape/);
  assert.match(graphSource, /hull\.comp\.accumulation\?\.regimeId/);

  // StickyBar tokenizes the formula so matching label chars can be haloed.
  assert.match(stickyBarSource, /FormulaHighlighted/);
  assert.match(stickyBarSource, /hoveredLabels/);

  // DecisionLadder accepts + threads spotlightLeafIds into leaf data, and
  // LeafNode renders a visible distinction when data.spotlight is true.
  assert.match(ladderSource, /spotlightLeafIds/);
  assert.match(ladderSource, /data\.spotlight/);
  assert.match(ladderSource, /data-leaf-spotlight/);
});

test('LabelInteractionGraph card surface is interactive (Stage 2)', () => {
  // Source-level checks for the interactive shell. We can't easily run
  // jsdom-level pointer simulations here, so we lock in the *structure*
  // that makes the tooltip work: the portal-based tooltip, the three
  // hover targets (node, edge, hull), the defensive dismissal effect,
  // and the wider transparent hit-line for edges (1 px edges are unhoverable).
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/ComponentView.jsx', import.meta.url),
    'utf8',
  );

  assert.match(source, /import \{ createPortal \} from 'react-dom'/);
  // The graph body must render a role=tooltip via portal so it escapes
  // ancestor CSS transforms (PanZoomCanvas).
  assert.match(source, /role="tooltip"/);
  assert.match(source, /createPortal\(/);

  // The three hover targets — keep the kind strings stable; the tooltip
  // content dispatch is keyed on them.
  assert.match(source, /kind: 'node'/);
  assert.match(source, /kind: 'edge'/);
  assert.match(source, /kind: 'hull'/);

  // Defensive dismissal mirrors the DecisionLadder/DecisionTree pattern.
  assert.match(source, /addEventListener\('scroll', dismiss, true\)/);
  assert.match(source, /addEventListener\('pointerdown', dismissIfOutside\)/);
  assert.match(source, /addEventListener\('keydown', dismissOnEscape\)/);

  // Edge hit area must be widened — the visible line is 1 px which is
  // effectively un-hoverable on its own.
  assert.match(source, /stroke="transparent"/);
  assert.match(source, /strokeWidth=\{10\}/);

  // Edge tooltip attributes the edge to a specific generator.
  assert.match(source, /Generator σ\$\{genIdx \+ 1\}/);
  // Hull tooltip surfaces the regime/shape presentation body (the whole
  // point of unifying hull color with the ladder via getRegimePresentation).
  assert.match(source, /presentation\?\.tooltip\?\.body/);
});
