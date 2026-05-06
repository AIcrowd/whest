// website/symmetry-explorer.product-savings-row.test.mjs
//
// Source-grep coverage for V3.1 §11 — C11 Product Savings Metric Row on
// MultiplicationCostCard. The row shows four metric cards (dense
// baseline, product-rep count M, multiplication-chain events μ, and
// product-side reduction) plus a warning pill that scrolls back to the
// O→Q matrix when the reader confuses rows (products) with updates (α).
//
// Spec phrasing the tests pin (V3.1 §11, verbatim):
//   - "Dense product chains"
//   - "Product representatives M"
//   - "Multiplication-chain events mu"
//   - "Product-side reduction"
//   - "Products are rows, not updates."

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx',
);
const SRC = readFileSync(SRC_PATH, 'utf-8');

test('ProductSavingsMetricRow renders the four V3.1 verbatim metric labels', () => {
  // V3.1 §11 mandates these exact card labels. We pin them as JS string
  // literals rather than rendered text so the spec phrasing is locked
  // even if the surrounding JSX is restructured.
  assert.match(SRC, /'Dense product chains'/);
  assert.match(SRC, /'Product representatives M'/);
  assert.match(SRC, /'Multiplication-chain events mu'/);
  assert.match(SRC, /'Product-side reduction'/);
});

test('Warning pill carries the V3.1 verbatim copy', () => {
  // The pill is the single most-clicked corrective on this card: it
  // tells readers that big numbers here are products (rows), not
  // accumulation updates (α). The exact phrasing is part of the spec.
  assert.match(SRC, /'Products are rows, not updates\.'/);
  // It must be addressable as a constant so downstream copy-edit
  // sweeps can find it.
  assert.match(SRC, /WARNING_PILL_TEXT\s*=\s*'Products are rows, not updates\.'/);
});

test('Warning pill is a clickable button with onClick + aria-label', () => {
  // Surfacing the pill as a real <button type="button"> rather than a
  // styled <div> so it's keyboard-activatable and announced as a
  // button to screen readers.
  assert.match(SRC, /<button[\s\S]*?type="button"[\s\S]*?onClick=\{onClick\}[\s\S]*?aria-label=/);
  // The aria-label must spell out the click affordance so a screen
  // reader user knows it scrolls.
  assert.match(SRC, /aria-label="Products are rows, not updates\. Click to scroll to O→Q matrix\."/);
});

test('Each metric card has tabIndex=0 and onMouseEnter/onFocus', () => {
  // Pull the MetricCard component body and assert each interaction
  // hook is wired. We grep on the rendered <div> so structural changes
  // (e.g. switching to a <button>) still pass as long as the
  // accessibility wiring stays intact.
  const cardBlock = SRC.match(/function MetricCard[\s\S]*?^\}/m);
  assert.ok(cardBlock, 'expected a MetricCard function body');
  assert.match(cardBlock[0], /tabIndex=\{0\}/);
  assert.match(cardBlock[0], /role="button"/);
  assert.match(cardBlock[0], /onMouseEnter=\{onHover\}/);
  assert.match(cardBlock[0], /onMouseLeave=\{onLeave\}/);
  assert.match(cardBlock[0], /onFocus=\{onHover\}/);
  assert.match(cardBlock[0], /onBlur=\{onLeave\}/);
});

test('Warning pill click handler scrolls a target into view', () => {
  // The click handler must (a) look up the O→Q matrix DOM target and
  // (b) call scrollIntoView on it. We pin both halves so a future
  // refactor can't silently drop the scroll behavior.
  assert.match(SRC, /document\.querySelector\(['"]\[data-testid="orbit-rep-matrix"\]['"]\)/);
  assert.match(SRC, /scrollIntoView\(\{[\s\S]*?behavior:\s*['"]smooth['"]/);
  // Spec also calls out a fallback to the BranchingDemo container
  // (id="orbit-rep-matrix") if the testid attribute can't be found.
  assert.match(SRC, /getElementById\(['"]orbit-rep-matrix['"]\)/);
});

test('Metric row has a stable data-testid + four named card testids', () => {
  // The row container is keyed off `product-savings-metric-row` so
  // integration tests can target it without relying on class names.
  assert.match(SRC, /data-testid="product-savings-metric-row"/);
  // The MetricCard component receives a `testId` prop and renders it
  // as `data-testid`. We pin both halves: the wiring on the component
  // and the four call-site values that name each metric.
  assert.match(SRC, /data-testid=\{testId\}/);
  assert.match(SRC, /testId="product-savings-dense"/);
  assert.match(SRC, /testId="product-savings-M"/);
  assert.match(SRC, /testId="product-savings-mu"/);
  assert.match(SRC, /testId="product-savings-reduction"/);
  // Warning pill testid is rendered directly (not via MetricCard).
  assert.match(SRC, /data-testid="products-are-rows-pill"/);
});

test('M and μ card hovers fire onHoveredLabelsChange', () => {
  // Hovering the M card highlights all product-orbit labels; hovering
  // the μ card writes a sentinel 'mu-k-minus-1' token so downstream
  // views (e.g. the formula block) can emphasize the (k−1) factor
  // without disturbing the existing label-name bus.
  assert.match(SRC, /const onHoverM\s*=\s*\(\)\s*=>\s*writeHover\(allLabels\.size\s*\?\s*new Set\(allLabels\)\s*:\s*null\)/);
  assert.match(SRC, /const onHoverMu\s*=\s*\(\)\s*=>\s*writeHover\(new Set\(\['mu-k-minus-1'\]\)\)/);
  // The card accepts onHoveredLabelsChange and forwards it through.
  assert.match(SRC, /onHoveredLabelsChange\s*=\s*null/);
  assert.match(SRC, /onHoveredLabelsChange=\{onHoveredLabelsChange\}/);
});

test('Card accepts numTerms prop and computes mu = (k - 1) M', () => {
  // The card needs num_terms to compute μ. Default is 2 (binary
  // multiply) so existing call sites keep working without changes.
  assert.match(SRC, /numTerms\s*=\s*2/);
  // The μ formula is exactly (k − 1) · M, applied once globally.
  assert.match(SRC, /const mu\s*=\s*M\s*!=\s*null\s*\?\s*\(k\s*-\s*1\)\s*\*\s*M\s*:\s*null/);
});
