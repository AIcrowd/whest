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
  const componentCostSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url),
    'utf8',
  );

  assert.match(source, /import \{ createPortal \} from 'react-dom'/);
  assert.match(source, /import RoleBadge from '\.\/RoleBadge\.jsx'/);
  assert.match(source, /import SymmetryBadge from '\.\/SymmetryBadge\.jsx'/);
  assert.match(source, /const COLOR_V = explorerThemeColor\(explorerThemeId, 'hero'\)/);
  assert.match(source, /const COLOR_W = explorerThemeColor\(explorerThemeId, 'summedSide'\)/);
  assert.match(source, /const FREE_HULL_COLOR = explorerThemeColor\(explorerThemeId, 'heroMuted'\)/);
  assert.match(source, /const SUMMED_HULL_COLOR = explorerThemeColor\(explorerThemeId, 'summedSide'\)/);
  assert.match(source, /const hasFreeLabel = indices\.some\(\(idx\) => vSet\.has\(allLabels\[idx\]\)\)/);
  assert.match(source, /color: hasFreeLabel \? FREE_HULL_COLOR : SUMMED_HULL_COLOR/);
  // The graph body must render a role=tooltip via portal so it escapes
  // ancestor CSS transforms (PanZoomCanvas).
  assert.match(source, /role="tooltip"/);
  assert.match(source, /createPortal\(/);

  // The two visible hover targets — keep the kind strings stable; the
  // tooltip content dispatch is keyed on them. The 'edge' kind is preserved
  // in the tooltip switch (handlers at the kind === 'edge' branch) for the
  // moment, but no edge geometry is rendered — the toggle that exposed
  // edges was removed; only nodes and hulls remain visible. The edge
  // dispatch path is dead-but-reserved for a future hover affordance.
  assert.match(source, /kind: 'node'/);
  assert.match(source, /kind: 'hull'/);
  assert.match(source, /kind === 'edge'/);

  // Defensive dismissal mirrors the DecisionLadder pattern.
  assert.match(source, /addEventListener\('scroll', dismiss, true\)/);
  assert.match(source, /addEventListener\('pointerdown', dismissIfOutside\)/);
  assert.match(source, /addEventListener\('keydown', dismissOnEscape\)/);

  // Edge tooltip attributes the edge to a specific generator (string lives
  // in the dispatch handler that runs when an 'edge' kind tooltip fires).
  assert.match(source, /Generator σ\$\{genIdx \+ 1\}/);
  // Hull tooltip surfaces the regime/shape presentation body (the whole
  // point of unifying hull color with the ladder via getRegimePresentation).
  assert.match(source, /presentation\?\.tooltip\?\.body/);
  assert.match(source, /<RoleBadge key=\{`tooltip-\$\{hovered\.idx\}-\$\{label\}`\} role=\{role\}>/);
  assert.match(source, /<SymmetryBadge value=\{comp\.groupName \|\| 'trivial'\} \/>/);

  // The legend should use graph chrome colors, not notation-role colors, so
  // editorial-noir can keep V_free notation restrained without losing the
  // visual distinction in the Interaction Graph.
  assert.match(componentCostSource, /const explorerThemeId = getActiveExplorerThemeId\(\)/);
  assert.match(componentCostSource, /const freeLabelColor = explorerThemeColor\(explorerThemeId, 'hero'\)/);
  assert.match(componentCostSource, /const summedLabelColor = explorerThemeColor\(explorerThemeId, 'summedSide'\)/);
  assert.match(componentCostSource, /const hullColor = explorerThemeColor\(explorerThemeId, 'heroMuted'\)/);
  assert.match(componentCostSource, /backgroundColor: freeLabelColor/);
  assert.match(componentCostSource, /backgroundColor: summedLabelColor/);
  assert.match(componentCostSource, /borderColor: hullColor/);
  assert.match(componentCostSource, /<Latex math=\{notationLatex\('v_free'\)\} inheritColor \/>/);
  assert.match(componentCostSource, /<Latex math=\{notationLatex\('w_summed'\)\} inheritColor \/>/);
  assert.doesNotMatch(componentCostSource, /backgroundColor: notationColor\('v_free'\)/);
});

test('StickyBar omits section nav pills so the live formula owns the top strip', () => {
  const source = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/StickyBar.jsx', import.meta.url),
    'utf8',
  );

  assert.doesNotMatch(source, /EXPLORER_ACTS\.map/);
  assert.doesNotMatch(source, /buttonVariants/);
  assert.doesNotMatch(source, /act\.navTitle/);
  assert.doesNotMatch(source, /href=\{`#\$\{act\.id\}`\}/);
  assert.match(source, /data-sticky-symmetry-summary="true"/);
  assert.match(source, /aria-label="Input and output symmetries"/);
  assert.match(source, /mx-auto flex max-w-\[1460px\] items-center justify-between gap-4/);
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
  assert.match(source, /import \{ restrictStabilizerToPositions \} from '\.\.\/engine\/outputOrbit\.js'/);
  assert.match(source, /function SymmetryChip/);
  assert.match(source, /function CompactSymmetrySummary/);
  assert.match(source, /buildMetadataItems/);
  assert.match(source, /const operandNames = Array\.isArray\(example\?\.operandNames\)/);
  assert.match(source, /function symmetryLabelFromPerOp/);
  assert.match(source, /function outputSymmetryLabel/);
  assert.match(source, /restrictStabilizerToPositions\(elements, visiblePositions\)/);
  assert.match(source, /const perOpSymmetry = Array\.isArray\(example\?\.perOpSymmetry\) \? example\.perOpSymmetry : \[\]/);
  assert.match(source, /const variablesByName = new Map/);
  assert.match(source, /const operands = operandNames\.map\(\(name, idx\) => \(\{/);
  assert.match(source, /idx < perOpSymmetry\.length && perOpSymmetry\[idx\] !== undefined/);
  assert.match(source, /symmetryLabelFromPerOp\(perOpSymmetry\[idx\]\)/);
  assert.match(source, /symmetryLabel\(variablesByName\.get\(name\)\)/);
  assert.match(source, /outputLabel: outputSymmetryLabel\(group\)/);
  assert.match(source, /flex min-w-0 flex-nowrap items-center justify-start gap-2 overflow-x-auto/);
  assert.match(source, /inline-flex h-7 shrink-0 items-center/);
  assert.match(source, /h-5 w-px shrink-0 bg-gray-200/);
  assert.match(source, /<CompactSymmetrySummary/);
  assert.match(source, /text-stone-400">×<\//);
  assert.match(source, /text-stone-400">→<\//);
  assert.match(source, /font-semibold text-primary">R<\//);
  assert.match(source, /return `\|H\|=\$\{outputElements\.length\}`/);
  assert.doesNotMatch(source, /group\?\.fullGroupName \|\| 'trivial'/);
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

  assert.match(source, /EXPLORER_THEME_PRESETS/);
  assert.match(source, /getExplorerThemePreset/);
  assert.match(source, /Explorer Theme/);
  assert.match(source, /activeTheme\.summary/);
  assert.match(source, /border border-\[color:color-mix\(in_oklab,var\(--explorer-editorial-accent\)_28%,var\(--explorer-border\)\)\]/);
  assert.match(source, /border border-\[color:color-mix\(in_oklab,var\(--explorer-editorial-accent\)_24%,var\(--explorer-border\)\)\]/);
  assert.match(source, /text-\[color:color-mix\(in_oklab,var\(--explorer-editorial-accent\)_45%,var\(--explorer-ink\)\)\]/);
  assert.match(source, /const warmInkLabel =/);
  assert.match(source, /<div className=\{`text-\[10px\] font-semibold uppercase tracking-\[0\.18em\] \$\{warmInkLabel\}`\}>/);
  assert.match(source, /<label className=\{`block text-\[11px\] font-semibold uppercase tracking-\[0\.16em\] \$\{warmInkLabel\}`\}>/);
});
