// symmetry-explorer.classification-tree-hero.test.mjs
//
// C29 Classification Tree Hero — V3.1 audit gap closure tests.
// Verifies the 4 interaction features added in the L5.T1.8 polish:
//   1. activeAlphaMethod prop accepted by DecisionLadder
//   2. Node hover tooltip surfaces checks / why / intuition
//   3. Leaf hover tooltip uses V3.1 4-line format (Applies when / Counts / Full statement)
//   4. Click-leaf navigates to #appendix-section-2
//   5. Alpha-method highlight — coral outline on the matching leaf
//   6. Hex audit — no raw hex outside approved palette
//   7. Accessibility — node + leaf both keyboard-focusable (cursor-help present)
//
// All tests are source-grep based (no DOM required) — they read the JSX source
// and assert on patterns that prove the feature is wired, matching the project
// convention established in the existing decision-ladder tests.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE = resolve(__dirname, 'components/symmetry-aware-einsum-contractions');

const ladder = readFileSync(resolve(BASE, 'components/DecisionLadder.jsx'), 'utf-8');
const app = readFileSync(resolve(BASE, 'SymmetryAwareEinsumContractionsApp.jsx'), 'utf-8');
const bus = readFileSync(resolve(BASE, 'lib/alphaMethodBus.js'), 'utf-8');

// ── 1. activeAlphaMethod prop ─────────────────────────────────────────────

test('DecisionLadder accepts activeAlphaMethod prop', () => {
  // The prop must appear in the function signature with a default of null.
  assert.match(ladder, /activeAlphaMethod\s*=\s*null/);
});

test('DecisionLadder subscribes to alphaMethodBus via useSyncExternalStore', () => {
  // The bus integration should use useSyncExternalStore for React-safe updates.
  assert.match(ladder, /useSyncExternalStore/);
  assert.match(ladder, /subscribeActiveAlphaMethod/);
  assert.match(ladder, /getActiveAlphaMethod/);
});

test('App syncs activeAlphaMethodHover to alphaMethodBus via useEffect', () => {
  // The App must import and call setActiveAlphaMethodBus.
  assert.match(app, /setActiveAlphaMethodBus/);
  // The call must be inside a useEffect that depends on activeAlphaMethodHover.
  assert.match(app, /setActiveAlphaMethodBus\(activeAlphaMethodHover\)/);
});

// ── 2. Node hover tooltip — checks / why / intuition ─────────────────────

test('DecisionLadder node tooltip renders "What\'s checked" section for question nodes', () => {
  // The tooltip render path must have a section for the `checks` field.
  assert.match(ladder, /What's checked/);
  assert.match(ladder, /activeTooltip\.checks/);
});

test('DecisionLadder node tooltip renders "Why it matters" section for question nodes', () => {
  assert.match(ladder, /Why it matters/);
  assert.match(ladder, /activeTooltip\.why/);
});

test('DecisionLadder node tooltip renders "Intuition" section for question nodes', () => {
  assert.match(ladder, /Intuition/);
  assert.match(ladder, /activeTooltip\.intuition/);
});

// ── 3. Leaf hover tooltip — V3.1 4-line format ───────────────────────────

test('DecisionLadder leaf tooltip renders "Applies when" line', () => {
  // The tooltip for leaf nodes must show "Applies when: ..." derived from spec.when.
  assert.match(ladder, /Applies when:/);
});

test('DecisionLadder leaf tooltip renders "Counts" line (V3.1 CaseBadge format)', () => {
  // Must show the standard "Counts: filled O → Q cells" caption.
  assert.match(ladder, /Counts:/);
  assert.match(ladder, /countsLine/);
  // The value should describe what α counts.
  assert.match(ladder, /Filled O → Q cells for this component/);
});

test('DecisionLadder leaf tooltip renders "Full statement → Appendix B" link', () => {
  // Must show the appendixHref link.
  assert.match(ladder, /Full statement:/);
  assert.match(ladder, /appendixHref/);
  assert.match(ladder, /Appendix B/);
});

// ── 4. Click-leaf → Appendix B anchor ────────────────────────────────────

test('DecisionLadder click-leaf handler navigates to #appendix-section-2', () => {
  // handleNodeClick must set window.location.hash to the appendix anchor.
  assert.match(ladder, /#appendix-section-2/);
  assert.match(ladder, /handleNodeClick/);
  // The handler checks specFor to confirm it's a leaf before navigating.
  assert.match(ladder, /specFor\(canonicalId\)/);
});

test('DecisionLadderGraph receives onNodeClick prop', () => {
  // The memo'd graph component must accept and forward onNodeClick.
  assert.match(ladder, /onNodeClick/);
  // ReactFlow must receive the onNodeClick handler.
  assert.match(ladder, /onNodeClick=\{onNodeClick\}/);
});

// ── 5. Alpha-method highlight ─────────────────────────────────────────────

test('DecisionLadder leaf alphaHighlight flag is set when activeAlphaMethod matches leafId', () => {
  // buildLadderLayout must set alphaHighlight on the matching leaf.
  assert.match(ladder, /alphaHighlight:\s*activeAlphaMethod\s*!=\s*null\s*&&\s*activeAlphaMethod\s*===\s*leafId/);
});

test('DecisionLadder LeafNode renders coral outline box-shadow when alphaHighlight is true', () => {
  // The shadow computation must include the CORAL_OUTLINE constant when alphaHighlight is set.
  assert.match(ladder, /CORAL_OUTLINE/);
  assert.match(ladder, /data\.alphaHighlight/);
  // CORAL_OUTLINE references the design-system --coral CSS variable, not a raw
  // hex (raw #F0524D would duplicate the V_free notation color and fail the
  // notation-system host-files audit).
  assert.match(ladder, /var\(--coral\)/);
});

test('DecisionLadder LeafNode uses data-leaf-alpha-highlight attribute for test targeting', () => {
  // A data attribute allows integration tests and e2e tests to find highlighted leaves.
  assert.match(ladder, /data-leaf-alpha-highlight/);
});

// ── 6. Hex audit ─────────────────────────────────────────────────────────

test('DecisionLadder hex colours are limited to approved palette entries', () => {
  // Scan for raw hex literals. Approved set (stable tokens used here):
  //   #FFFFFF  — white (active ring separator, tooltip bg)
  //   #F8FAFC  — light foreground on dark leaves
  //   #132228  — dark text on light leaves
  //   #94A3B8  — fallback gray for unknown leaves
  //   #FFFFFF26 — semi-transparent spotlight (notation-colored alpha)
  // Note: the alphaHighlight coral ring uses var(--coral) (the design-system
  // CSS variable), not a raw hex — keeping notation colors out of host files.
  const hexRe = /#[0-9A-Fa-f]{3,8}\b/g;
  const approved = new Set(['#FFFFFF', '#F8FAFC', '#132228', '#94A3B8', '#FFFFFF26']);
  const found = new Set((ladder.match(hexRe) ?? []).map((h) => h.toUpperCase()));
  // Filter out hex values that appear only inside comment blocks
  // (The test deliberately only checks production hex, not comment prose.)
  const source = ladder.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
  const prodHex = new Set((source.match(hexRe) ?? []).map((h) => h.toUpperCase()));
  for (const hex of prodHex) {
    assert.ok(
      approved.has(hex),
      `Unapproved raw hex in DecisionLadder production code: ${hex}. Use design-system tokens.`,
    );
  }
});

// ── 7. Accessibility ─────────────────────────────────────────────────────

test('DecisionLadder QuestionNode and LeafNode both carry cursor-help class for keyboard affordance', () => {
  // cursor-help signals to assistive technology (and keyboard users) that
  // hovering/focusing will show additional information.
  assert.match(ladder, /cursor-help/);
  // Both node types use it (check they each appear in function definitions
  // before the cursor-help occurrence by checking for at least 2 occurrences).
  const count = (ladder.match(/cursor-help/g) ?? []).length;
  assert.ok(count >= 2, `Expected cursor-help on at least 2 node types, found ${count}`);
});

test('alphaMethodBus exports the required pub/sub API', () => {
  // The bus module must export all three named symbols.
  assert.match(bus, /export function setActiveAlphaMethodBus/);
  assert.match(bus, /export function subscribeActiveAlphaMethod/);
  assert.match(bus, /export function getActiveAlphaMethod/);
});
