/**
 * symmetry-explorer.live-contraction-strip.test.mjs
 *
 * Source-grep tests for C01 Live Contraction Strip behaviors added in
 * L5.T1.1. Each test asserts that a structural pattern is present in
 * StickyBar.jsx (and related files) so that the 4 V3.1 behaviors cannot be
 * accidentally removed without a test failure.
 *
 * Test plan:
 *  T01 — Behavior 1: label-hover write direction (onHoveredLabelsChange)
 *  T02 — Behavior 1: letter tokens fire onHoveredLabelsChange on enter/leave
 *  T03 — Behavior 2: DimensionStepper helper exists with keyboard controls
 *  T04 — Behavior 2: stepper shows n=${dimensionN ?? '—'} in the live span
 *  T05 — Behavior 3: α-method badge fires onActiveAlphaMethodHoverChange
 *  T06 — Behavior 4: compact-on-scroll rAF-throttled scroll listener
 *  T07 — Behavior 4: prefers-reduced-motion guard on transition
 *  T08 — App wires onDimensionNChange, onHoveredLabelsChange, and
 *         activeAlphaMethod props to StickyBar
 */

import { strict as assert } from 'node:assert';
import fs from 'node:fs';
import { describe, test } from 'node:test';

const STICKY_BAR_URL = new URL(
  './components/symmetry-aware-einsum-contractions/components/StickyBar.jsx',
  import.meta.url,
);
const APP_URL = new URL(
  './components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
  import.meta.url,
);

describe('C01 Live Contraction Strip behaviors', () => {
  const source = fs.readFileSync(STICKY_BAR_URL, 'utf8');
  const appSource = fs.readFileSync(APP_URL, 'utf8');

  // ------------------------------------------------------------------
  // Behavior 1 — Label-hover cross-highlighting WRITE direction
  // ------------------------------------------------------------------
  test('T01: StickyBar accepts onHoveredLabelsChange prop and threads it to FormulaHighlighted', () => {
    // Prop must appear in the default export function signature
    assert.match(source, /onHoveredLabelsChange = null/);
    // Must be forwarded to FormulaHighlighted
    assert.match(source, /onHoveredLabelsChange=\{onHoveredLabelsChange\}/);
  });

  test('T02: SubscriptTokens fires onHoveredLabelsChange on mouse enter and leave', () => {
    // Hover-in: call with a Set containing the hovered letter
    assert.match(source, /onMouseEnter.*onHoveredLabelsChange\(new Set\(\[ch\]\)\)/s);
    // Hover-out: call with null
    assert.match(source, /onMouseLeave.*onHoveredLabelsChange\(null\)/s);
    // Also handles keyboard focus for accessibility
    assert.match(source, /onFocus.*onHoveredLabelsChange\(new Set\(\[ch\]\)\)/s);
    assert.match(source, /onBlur.*onHoveredLabelsChange\(null\)/s);
  });

  // ------------------------------------------------------------------
  // Behavior 2 — Dimension knob in strip
  // ------------------------------------------------------------------
  test('T03: DimensionStepper helper exists with − and + buttons and min/max bounds', () => {
    assert.match(source, /function DimensionStepper/);
    // Decrement button has aria-label
    assert.match(source, /aria-label="Decrease dimension"/);
    // Increment button has aria-label
    assert.match(source, /aria-label="Increase dimension"/);
    // Bounds are enforced
    assert.match(source, /Math\.max\(min, dimensionN - 1\)/);
    assert.match(source, /Math\.min\(max, dimensionN \+ 1\)/);
  });

  test('T04: DimensionStepper renders the n= display with the dimensionN ?? template and aria-live', () => {
    // Template literal in the live span — also satisfies the pinned assertion in
    // interaction-graph.test.mjs T196 (dimensionN = null default)
    assert.match(source, /`n=\$\{dimensionN \?\? '—'\}`/);
    // aria-live polite region for screen-reader announcements
    assert.match(source, /aria-live="polite"/);
    // aria-atomic for atomic announcements
    assert.match(source, /aria-atomic="true"/);
  });

  // ------------------------------------------------------------------
  // Behavior 3 — α-method badge → tree-leaf hover bus
  // ------------------------------------------------------------------
  test('T05: α-method badge fires onActiveAlphaMethodHoverChange on hover and blur', () => {
    // Prop accepted by StickyBar
    assert.match(source, /onActiveAlphaMethodHoverChange = null/);
    // On enter: call with the method id
    assert.match(source, /onActiveAlphaMethodHoverChange\?\.\(activeAlphaMethod\)/);
    // On leave: call with null
    assert.match(source, /onActiveAlphaMethodHoverChange\?\.\(null\)/);
    // Badge has an accessible label
    assert.match(source, /aria-label=\{`Alpha method: \$\{activeAlphaMethod\}`\}/);
  });

  // ------------------------------------------------------------------
  // Behavior 4 — Compact-on-scroll
  // ------------------------------------------------------------------
  test('T06: compact-on-scroll uses rAF-throttled window scroll listener with 200px threshold', () => {
    // requestAnimationFrame throttle guard
    assert.match(source, /requestAnimationFrame/);
    // 200px scroll threshold
    assert.match(source, /window\.scrollY > 200/);
    // data-compact attribute on the wrapper for CSS/testing hooks
    assert.match(source, /data-compact=\{isCompact \? 'true' : 'false'\}/);
    // Passive listener for perf
    assert.match(source, /passive: true/);
    // rAF cleanup on unmount
    assert.match(source, /cancelAnimationFrame/);
  });

  test('T07: compact transition is gated by prefers-reduced-motion detection', () => {
    // matchMedia detection
    assert.match(source, /prefers-reduced-motion: reduce/);
    // Conditional: no transition class when reduced motion is preferred
    assert.match(source, /prefersReducedMotionRef\.current/);
    // The transition classes themselves
    assert.match(source, /transition-\[height,opacity\]/);
    assert.match(source, /duration-200/);
  });

  // ------------------------------------------------------------------
  // App-level wiring
  // ------------------------------------------------------------------
  test('T08: App wires all 4 new StickyBar props', () => {
    // Dimension setter
    assert.match(appSource, /onDimensionNChange=\{setDefaultSize\}/);
    // Hover-labels setter
    assert.match(appSource, /onHoveredLabelsChange=\{setStripHoveredLabels\}/);
    // activeAlphaMethod state passed down
    assert.match(appSource, /activeAlphaMethod=\{activeAlphaMethodHover\}/);
    // α-method hover change handler
    assert.match(appSource, /onActiveAlphaMethodHoverChange=\{setActiveAlphaMethodHover\}/);
    // stripHoveredLabels state initialized
    assert.match(appSource, /setStripHoveredLabels/);
  });
});
