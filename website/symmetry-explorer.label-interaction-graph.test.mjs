/**
 * Tests for V3.1 §C20 — Label Interaction Graph polish.
 *
 * Closes 3 gaps without DOM rendering:
 *   Gap 1 — generator-support hulls distinguishable from cycle-adjacency hulls
 *   Gap 2 — toggle "certified factors / generator supports"
 *   Gap 3 — hover-component bus (activeComponentId + onActiveComponentHoverChange)
 *
 * All assertions are source-grep or import-based — no jsdom required.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CVPath = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/ComponentView.jsx',
);
const AppPath = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
);
const StoriesPath = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/BipartiteGraph.stories.jsx',
);

const cv = readFileSync(CVPath, 'utf-8');
const app = readFileSync(AppPath, 'utf-8');
const stories = readFileSync(StoriesPath, 'utf-8');

// ─────────────────────────────────────────────────────────────────────────────
// T1: File exists and exports LabelInteractionGraph
// ─────────────────────────────────────────────────────────────────────────────
test('ComponentView.jsx exists and exports LabelInteractionGraph as a named export', () => {
  assert.match(cv, /export function LabelInteractionGraph\s*\(/,
    'LabelInteractionGraph must be a named export');
});

// ─────────────────────────────────────────────────────────────────────────────
// T2: Toggle button group — 2 buttons with "certified factors" + "generator supports"
// ─────────────────────────────────────────────────────────────────────────────
test('LabelInteractionGraph renders a 2-button toggle group (Gap 2)', () => {
  assert.match(cv, /certified factors/,
    'Toggle must include "certified factors" label');
  assert.match(cv, /generator supports/,
    'Toggle must include "generator supports" label');
  // Both buttons must be in a role="group"
  assert.match(cv, /role="group"/,
    'Toggle group must have role="group"');
});

// ─────────────────────────────────────────────────────────────────────────────
// T3: Hover-component bus props accepted (Gap 3)
// ─────────────────────────────────────────────────────────────────────────────
test('LabelInteractionGraph accepts activeComponentId and onActiveComponentHoverChange props (Gap 3)', () => {
  assert.match(cv, /activeComponentId\s*=\s*null/,
    'activeComponentId prop must be in the signature with default null');
  assert.match(cv, /onActiveComponentHoverChange\s*=\s*null/,
    'onActiveComponentHoverChange prop must be in the signature with default null');
  // Must call the setter on hull hover
  assert.match(cv, /onActiveComponentHoverChange\(/,
    'onActiveComponentHoverChange must be called inside the component');
});

// ─────────────────────────────────────────────────────────────────────────────
// T4: Generator-support hull — visual distinction (Gap 1)
// ─────────────────────────────────────────────────────────────────────────────
test('LabelInteractionGraph applies dashed stroke to multi-cycle-glued hulls (Gap 1)', () => {
  // Heuristic comment must be present to document the approach
  assert.match(cv, /isMultiCycleGlued|multi-cycle-glued/,
    'Source must contain "isMultiCycleGlued" or "multi-cycle-glued" heuristic');
  // Dashed visual cue must be rendered
  assert.match(cv, /strokeDasharray/,
    'Hull rendering must use strokeDasharray for dashed visual encoding');
  // Separate inner dashed stroke for generator mode
  assert.match(cv, /generator.*mode|viewMode.*generator/i,
    'Generator mode must toggle visibility of dashed inner stroke');
});

// ─────────────────────────────────────────────────────────────────────────────
// T5: App-level wiring — activeComponentId state + setter passed to graph
// ─────────────────────────────────────────────────────────────────────────────
test('App wires activeComponentId state and setter to LabelInteractionGraph (Gap 3)', () => {
  assert.match(app, /activeComponentId.*setActiveComponentId.*=.*useState/,
    'App must declare useState for activeComponentId');
  assert.match(app, /activeComponentId=\{activeComponentId\}/,
    'App must pass activeComponentId prop to LabelInteractionGraph');
  assert.match(app, /onActiveComponentHoverChange=\{setActiveComponentId\}/,
    'App must pass setActiveComponentId as onActiveComponentHoverChange');
});

// ─────────────────────────────────────────────────────────────────────────────
// T6: Hex audit — no raw hex colors in ComponentView (only design-system tokens)
// ─────────────────────────────────────────────────────────────────────────────
test('ComponentView.jsx uses no raw hex colors outside the TOKEN map', () => {
  // Strip the TOKEN map declaration itself (lines defining hex values in the map)
  // then check the rest of the file for stray hex literals.
  const tokenMapMatch = cv.match(/const TOKEN\s*=\s*\{[\s\S]*?\};/);
  const bodyWithoutTokenMap = tokenMapMatch
    ? cv.slice(cv.indexOf(tokenMapMatch[0]) + tokenMapMatch[0].length)
    : cv;
  // Raw hex patterns: #RRGGBB or #RGB (but not inside template literals from
  // explorerThemeColor calls — those return tokens).  We flag any direct
  // '#[0-9a-fA-F]{3,8}' that appears as a JSX prop value or string literal.
  const hexInBody = bodyWithoutTokenMap.match(/"#[0-9a-fA-F]{3,8}"/g) ?? [];
  // Allow stone-* className strings ("stone-900" etc.) — these are Tailwind, not hex.
  const bareHexValues = hexInBody.filter(h => !h.includes('stone') && !h.includes('gray'));
  assert.equal(
    bareHexValues.length, 0,
    `Found raw hex string literals in ComponentView body: ${bareHexValues.join(', ')}`,
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// T7: Accessibility — toggle buttons have aria-pressed, hulls have role/tabIndex
// ─────────────────────────────────────────────────────────────────────────────
test('Toggle buttons have aria-pressed and hull polygons are keyboard-accessible (a11y)', () => {
  assert.match(cv, /aria-pressed=\{active\}/,
    'Toggle buttons must have aria-pressed={active}');
  // Hull polygons must be keyboard-reachable (tabIndex=0 or role)
  assert.match(cv, /tabIndex=\{0\}/,
    'Hull polygons must have tabIndex={0} for keyboard focus');
  assert.match(cv, /role="button"/,
    'Hull polygons must have role="button" for assistive technology');
  // Focus handler on hulls
  assert.match(cv, /onFocus=\{/,
    'Hull polygons must have onFocus handler for keyboard users');
});

// ─────────────────────────────────────────────────────────────────────────────
// T8: Stories include new C20 stories for toggle and hover-bus
// ─────────────────────────────────────────────────────────────────────────────
test('BipartiteGraph.stories.jsx includes LIG stories for toggle and hover-bus', () => {
  assert.match(stories, /LIGCertifiedFactors|LIG.*certified/i,
    'Stories must have a certified-factors story');
  assert.match(stories, /LIGHoveredComponent|activeComponentId/,
    'Stories must exercise the hover-component bus');
  assert.match(stories, /LIGGeneratorSupports|generator supports/i,
    'Stories must have a generator-supports story');
});
