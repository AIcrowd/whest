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

  // Defensive dismissal mirrors the DecisionLadder pattern.
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
  assert.match(source, /<RoleBadge key=\{`tooltip-\$\{hovered\.idx\}-\$\{label\}`\} role=\{role\}>/);
  assert.match(source, /<SymmetryBadge value=\{comp\.groupName \|\| 'trivial'\} \/>/);
});

test('StickyBar selected section pill keeps a visible coral inner marker and aligned geometry', () => {
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/StickyBar.jsx', import.meta.url),
    'utf8',
  );

  assert.match(source, /isActive\s*\?\s*'border-\[var\(--coral\)\] bg-white text-\[var\(--coral-hover\)\]/);
  assert.match(source, /isActive\s*\?\s*'border-\[var\(--coral\)\] bg-\[var\(--coral\)\] text-white'/);
  assert.match(source, /inline-flex h-9 min-h-9 items-center gap-2 rounded-full border px-3/);
  assert.match(source, /h-5 min-w-5 items-center justify-center rounded-full/);
  assert.doesNotMatch(source, /isActive\s*\?\s*'bg-primary\/20 text-primary'/);
});

test('StickyBar shows the current n in the leading pill and keeps the einsum formula as trigger', () => {
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/StickyBar.jsx', import.meta.url),
    'utf8',
  );

  assert.match(source, /function StickyMetadataPopover/);
  assert.match(source, /createPortal\(/);
  assert.match(source, /showMetadataPopover/);
  assert.match(source, /bg-white/);
  assert.match(source, /dimensionN = null/);
  assert.match(source, /`\{?n=\$\{dimensionN \?\? '—'\}\}?`/);
  assert.match(source, /import SymmetryBadge from '\.\/SymmetryBadge\.jsx'/);
  assert.match(source, /function SymmetryChip/);
  assert.match(source, /buildMetadataItems/);
  assert.match(source, /const operandNames = Array\.isArray\(example\?\.operandNames\)/);
  assert.match(source, /function symmetryLabelFromPerOp/);
  assert.match(source, /const perOpSymmetry = Array\.isArray\(example\?\.perOpSymmetry\) \? example\.perOpSymmetry : \[\]/);
  assert.match(source, /const variablesByName = new Map/);
  assert.match(source, /const operands = operandNames\.map\(\(name, idx\) => \(\{/);
  assert.match(source, /idx < perOpSymmetry\.length && perOpSymmetry\[idx\] !== undefined/);
  assert.match(source, /symmetryLabelFromPerOp\(perOpSymmetry\[idx\]\)/);
  assert.match(source, /symmetryLabel\(variablesByName\.get\(name\)\)/);
  assert.match(source, /<SymmetryBadge value=\{groupLabel\}/);
  assert.match(source, /inline-flex h-6 items-center gap-1 rounded-full/);
  assert.match(source, /className="h-6 px-2\.5 text-\[11px\] leading-5 shadow-none"/);
  assert.match(source, /text-stone-400">,<\//);
  assert.match(source, /text-stone-500">→<\//);
  assert.match(source, /group\?\.fullGroupName \|\| 'trivial'/);
  assert.match(source, /inline-flex w-max max-w-\[calc\(100vw-2rem\)\]/);
  assert.match(source, /clampedCenterX/);
  assert.doesNotMatch(source, /title="Detected symmetry group — drives compression/);
});

test('SymmetryBadge renders output symmetry in the same black-outline pill language while tokenizing group content', () => {
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/SymmetryBadge.jsx', import.meta.url),
    'utf8',
  );

  assert.match(source, /notationColor\('v_free'\)/);
  assert.match(source, /inline-flex h-6 items-center rounded-full border-black\/70 bg-white px-2\.5 font-mono/);
  assert.match(source, /function ColoredGroupTail/);
  assert.match(source, /font-bold text-black/);
  assert.match(source, /const shorthandMatch = text\.match\(\/\^\(\[A-Z\]\\d\+\)\(\\\{\.\*\\\}\)\?\$\/\);/);
  assert.match(source, /const permGroupMatch = text\.match\(\/\^\(PermGroup\)\(\.\*\)\$\/\);/);
  assert.ok(source.includes(String.raw`const generatedCyclesMatch = text.match(/^(\u27e8)(.*)(\u27e9)$/);`));
});

test('ExplorerThemeDock warms its secondary chrome with the editorial accent', () => {
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/ExplorerThemeDock.jsx', import.meta.url),
    'utf8',
  );

  assert.match(source, /var\(--explorer-editorial-accent\)/);
});
