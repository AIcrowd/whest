/**
 * Tests for C39 Component Accounting Table — V3.1 §C39 gaps closed by
 * L5.T1.5: V_a/W_a split column, G_a column, component-row hover bus,
 * method-badge hover bus, app-level wiring.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const COMP_SRC = read(
  'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx',
);
const APP_SRC = read(
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
);

// ─── 1. Props ─────────────────────────────────────────────────────────────────

test('ComponentCostView accepts onActiveComponentHoverChange prop', () => {
  assert.match(
    COMP_SRC,
    /onActiveComponentHoverChange/,
    'ComponentCostView must declare the onActiveComponentHoverChange prop',
  );
});

test('ComponentCostView accepts onActiveAlphaMethodHoverChange prop', () => {
  assert.match(
    COMP_SRC,
    /onActiveAlphaMethodHoverChange/,
    'ComponentCostView must declare the onActiveAlphaMethodHoverChange prop',
  );
});

// ─── 2. Column headers ────────────────────────────────────────────────────────

test('ComponentCostView has V_a / W_a column header (V<sub>a</sub> or V_a)', () => {
  const hasSubscriptForm = COMP_SRC.includes('V<sub>a</sub>');
  const hasPlainForm = COMP_SRC.includes('V_a');
  assert.ok(
    hasSubscriptForm || hasPlainForm,
    'ComponentCostView must render a V_a column header using V<sub>a</sub> or V_a',
  );
});

test('ComponentCostView has G_a column header (G<sub>a</sub> or G_a)', () => {
  const hasSubscriptForm = COMP_SRC.includes('G<sub>a</sub>');
  const hasPlainForm = COMP_SRC.includes('G_a');
  assert.ok(
    hasSubscriptForm || hasPlainForm,
    'ComponentCostView must render a G_a column header using G<sub>a</sub> or G_a',
  );
});

// ─── 3. Hover bus wiring ──────────────────────────────────────────────────────

test('ComponentCostView component row emits onActiveComponentHoverChange on hover', () => {
  // Row must have both mouseEnter and mouseLeave handlers calling the prop.
  assert.match(
    COMP_SRC,
    /onMouseEnter.*onActiveComponentHoverChange/s,
    'ComponentCostView rows must call onActiveComponentHoverChange on mouseEnter',
  );
  assert.match(
    COMP_SRC,
    /onMouseLeave.*onActiveComponentHoverChange/s,
    'ComponentCostView rows must call onActiveComponentHoverChange(null) on mouseLeave',
  );
});

test('ComponentCostView method-badge wrapper emits onActiveAlphaMethodHoverChange on hover', () => {
  assert.match(
    COMP_SRC,
    /onMouseEnter.*onActiveAlphaMethodHoverChange/s,
    'ComponentCostView method badge must call onActiveAlphaMethodHoverChange on mouseEnter',
  );
  assert.match(
    COMP_SRC,
    /onMouseLeave.*onActiveAlphaMethodHoverChange/s,
    'ComponentCostView method badge must call onActiveAlphaMethodHoverChange(null) on mouseLeave',
  );
});

// ─── 4. App-level wiring ─────────────────────────────────────────────────────

test('SymmetryAwareEinsumContractionsApp wires setActiveComponentId to ComponentCostView', () => {
  assert.match(
    APP_SRC,
    /onActiveComponentHoverChange\s*=\s*\{setActiveComponentId\}/,
    'App must pass setActiveComponentId as onActiveComponentHoverChange to ComponentCostView',
  );
});

test('SymmetryAwareEinsumContractionsApp wires setActiveAlphaMethodHover to ComponentCostView', () => {
  assert.match(
    APP_SRC,
    /onActiveAlphaMethodHoverChange\s*=\s*\{setActiveAlphaMethodHover\}/,
    'App must pass setActiveAlphaMethodHover as onActiveAlphaMethodHoverChange to ComponentCostView',
  );
});

// ─── 5. No raw hex outside design-system tokens ───────────────────────────────

test('ComponentCostView new code does not introduce additional raw hex colors', () => {
  // #6B7280 is a pre-existing hex in InteractionGraphLegend (edge color swatch)
  // that predates this task. All other hex colors are forbidden.
  const ALLOWED_HEX = new Set(['#6B7280']);
  const hexes = COMP_SRC.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  for (const h of hexes) {
    assert.ok(
      ALLOWED_HEX.has(h.toUpperCase()) || ALLOWED_HEX.has(h),
      `Disallowed hex ${h} in ComponentCostView — use a design-system token instead`,
    );
  }
});

// ─── 6. Accessibility — row hover targets have tabindex and role ──────────────

test('ComponentCostView component rows have tabIndex and role for keyboard accessibility', () => {
  assert.match(
    COMP_SRC,
    /tabIndex=\{0\}/,
    'ComponentCostView rows must have tabIndex={0} for keyboard focus',
  );
  assert.match(
    COMP_SRC,
    /role="row"/,
    'ComponentCostView rows must have role="row" for screen readers',
  );
});

// ─── 7. V_a/W_a data is read from comp.va and comp.wa ───────────────────────

test('ComponentCostView VWSplitCell reads comp.va and comp.wa', () => {
  assert.match(
    COMP_SRC,
    /comp\.va/,
    'ComponentCostView must read comp.va for the V_a labels',
  );
  assert.match(
    COMP_SRC,
    /comp\.wa/,
    'ComponentCostView must read comp.wa for the W_a labels',
  );
});

// ─── 8. G_a reads groupName from component data ───────────────────────────────

test('ComponentCostView G_a column reads comp.groupName', () => {
  assert.match(
    COMP_SRC,
    /comp\.groupName/,
    'ComponentCostView must read comp.groupName for the G_a column',
  );
});
