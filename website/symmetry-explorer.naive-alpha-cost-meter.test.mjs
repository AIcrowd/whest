/**
 * Tests for NaiveAlphaCostMeter (C28) — source-grep style.
 *
 * These tests parse the JSX source and stories as text strings and
 * assert structural properties. They run in Node.js via the native
 * test runner (no DOM required).
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

const COMPONENT_PATH = 'components/symmetry-aware-einsum-contractions/components/NaiveAlphaCostMeter.jsx';
const STORIES_PATH   = 'components/symmetry-aware-einsum-contractions/components/NaiveAlphaCostMeter.stories.jsx';
const APP_PATH       = 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports a default function
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — component file exists and exports default', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /export default function NaiveAlphaCostMeter/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Renders 4 metric labels
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — renders "product orbits" metric label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /product orbits/);
});

test('NaiveAlphaCostMeter — renders "tuple/group touches" metric label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /tuple\/group touches/);
});

test('NaiveAlphaCostMeter — renders "projected outputs canonicalized" metric label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /projected outputs canonicalized/);
});

test('NaiveAlphaCostMeter — renders "interactive budget" metric label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /interactive budget/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Renders 4-tier gauge labels (small / feasible / expensive / unavailable)
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — gauge tier "small" present', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /['"]small['"]/);
});

test('NaiveAlphaCostMeter — gauge tier "feasible" present', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /['"]feasible['"]/);
});

test('NaiveAlphaCostMeter — gauge tier "expensive" present', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /['"]expensive['"]/);
});

test('NaiveAlphaCostMeter — gauge tier "unavailable" present', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /['"]unavailable['"]/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. "Show literal algorithm" toggle button is present
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — has "Show literal algorithm" toggle button', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /Show literal algorithm/);
});

test('NaiveAlphaCostMeter — toggle button uses aria-expanded', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /aria-expanded/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Pseudocode block is present in source
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — pseudocode contains "alpha = 0"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /alpha = 0/);
});

test('NaiveAlphaCostMeter — pseudocode contains "for O in product_orbits"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /for O in product_orbits/);
});

test('NaiveAlphaCostMeter — pseudocode contains "canonical_output_rep"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /canonical_output_rep/);
});

test('NaiveAlphaCostMeter — renders <pre><code> block for pseudocode', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /<pre\b/);
  assert.match(src, /<code>/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. Hex audit — only design-system tokens
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — no raw hex outside TOKEN block', () => {
  const src = read(COMPONENT_PATH);
  // Strip the TOKEN constant block, then scan remaining source for bare hex.
  const withoutTokenBlock = src.replace(/const TOKEN\s*=\s*\{[\s\S]*?\};/, '');
  const remainingHexes = withoutTokenBlock.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  assert.deepEqual(
    remainingHexes,
    [],
    `Raw hex outside TOKEN block: ${remainingHexes.join(', ')}`,
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Accessibility — aria attributes on interactive elements
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — toggle button has aria-expanded', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /aria-expanded/);
});

test('NaiveAlphaCostMeter — toggle button has aria-controls', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /aria-controls/);
});

test('NaiveAlphaCostMeter — gauge segments have role="listitem"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /role="listitem"/);
});

test('NaiveAlphaCostMeter — gauge segments have tabIndex', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /tabIndex/);
});

test('NaiveAlphaCostMeter — gauge segments have aria-label', () => {
  const src = read(COMPONENT_PATH);
  // Gauge segments use aria-label including the tier name
  assert.match(src, /aria-label=\{`Cost tier:/);
});

test('NaiveAlphaCostMeter — metric values use aria-live="polite"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /aria-live="polite"/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. App mounts NaiveAlphaCostMeter in §7 counting-shortcuts section
// ─────────────────────────────────────────────────────────────────────────────
test('App — imports NaiveAlphaCostMeter', () => {
  const src = read(APP_PATH);
  assert.match(src, /import NaiveAlphaCostMeter from '\.\/components\/NaiveAlphaCostMeter\.jsx'/);
});

test('App — mounts <NaiveAlphaCostMeter in §7 counting-shortcuts section', () => {
  const src = read(APP_PATH);
  const idx7 = src.indexOf('§7 Counting Shortcuts');
  const idxMount = src.indexOf('<NaiveAlphaCostMeter');
  assert.ok(idx7 > -1, '§7 Counting Shortcuts comment not found in App');
  assert.ok(idxMount > -1, '<NaiveAlphaCostMeter mount not found in App');
  assert.ok(idxMount > idx7, '<NaiveAlphaCostMeter must appear after §7 comment');
});

test('App — NaiveAlphaCostMeter receives dimensionN prop', () => {
  const src = read(APP_PATH);
  // The mount block is self-closing; extract from <NaiveAlphaCostMeter to />
  const start = src.indexOf('<NaiveAlphaCostMeter');
  assert.ok(start > -1, '<NaiveAlphaCostMeter not found in App');
  const end = src.indexOf('/>', start);
  assert.ok(end > -1, 'Self-closing /> not found after <NaiveAlphaCostMeter');
  const mountBlock = src.slice(start, end + 2);
  assert.match(mountBlock, /dimensionN=/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. Stories file: ≥3 named exports
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter stories — file exists and has ≥3 named exports', () => {
  const src = read(STORIES_PATH);
  const exports = src.match(/^export const \w+/gm) ?? [];
  assert.ok(
    exports.length >= 3,
    `Expected ≥3 named story exports, found ${exports.length}: ${exports.join(', ')}`,
  );
});

test('NaiveAlphaCostMeter stories — includes Small story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const Small/);
});

test('NaiveAlphaCostMeter stories — includes Feasible story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const Feasible/);
});

test('NaiveAlphaCostMeter stories — includes Unavailable story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const Unavailable/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 10. Tier thresholds — logical correctness check
// ─────────────────────────────────────────────────────────────────────────────
test('NaiveAlphaCostMeter — TIERS array has 4 entries in ascending order', () => {
  const src = read(COMPONENT_PATH);
  // Extract threshold values from TIERS definition
  const thresholdMatches = src.match(/threshold:\s*([\de.]+|Infinity)/g) ?? [];
  assert.equal(thresholdMatches.length, 4, `Expected 4 threshold entries, got ${thresholdMatches.length}`);
  // Ensure Infinity is the last threshold
  assert.ok(thresholdMatches[thresholdMatches.length - 1].includes('Infinity'), 'Last tier threshold should be Infinity');
});

test('NaiveAlphaCostMeter — interactive budget constant is 1_000_000', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /INTERACTIVE_BUDGET\s*=\s*1_?000_?000/);
});
