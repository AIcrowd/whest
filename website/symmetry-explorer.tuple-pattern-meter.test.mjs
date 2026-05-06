/**
 * Tests for TuplePatternMeter (C37) — source-grep style.
 *
 * Mirrors symmetry-explorer.naive-alpha-cost-meter.test.mjs.
 * These tests parse the JSX source as text and assert structural
 * properties — no DOM, no React renderer required.
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

const COMPONENT_PATH = 'components/symmetry-aware-einsum-contractions/components/TuplePatternMeter.jsx';
const STORIES_PATH   = 'components/symmetry-aware-einsum-contractions/components/TuplePatternMeter.stories.jsx';
const APP_PATH       = 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports a default function with the right name
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — component file exists and exports default', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /export default function TuplePatternMeter/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Prop signature — accepts dimensionN, allLabels, groupSize, componentData
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — accepts dimensionN, allLabels, groupSize, componentData props', () => {
  const src = read(COMPONENT_PATH);
  // Locate the default-export signature and inspect destructured props.
  const start = src.indexOf('export default function TuplePatternMeter(');
  assert.ok(start > -1, 'default export signature not found');
  const end = src.indexOf(') {', start);
  const sig = src.slice(start, end);
  assert.match(sig, /dimensionN/);
  assert.match(sig, /allLabels/);
  assert.match(sig, /groupSize/);
  assert.match(sig, /componentData/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Renders the four bars (dense / tuple-group / typed-patterns / orbits)
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — renders "dense assignments" bar label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /dense assignments \|X_a\|/);
});

test('TuplePatternMeter — renders "tuple/group touches" bar label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /tuple\/group touches/);
});

test('TuplePatternMeter — renders "typed patterns" bar label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /typed patterns/);
});

test('TuplePatternMeter — renders "pattern orbits" bar label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /pattern orbits/);
});

test('TuplePatternMeter — BARS array has 4 entries', () => {
  const src = read(COMPONENT_PATH);
  // Match each id in the BARS literal.
  const ids = src.match(/id:\s*'(dense|tupleGroup|typedPatterns|patternOrbits)'/g) ?? [];
  assert.equal(ids.length, 4, `Expected 4 bar ids, found ${ids.length}: ${ids.join(', ')}`);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. V3.1 caption verbatim — "Same α, fewer objects counted."
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — shows V3.1 caption "Same α, fewer objects counted." verbatim', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /Same α, fewer objects counted\./);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Over-budget caveat — "when pattern budget passes" surfaces when the
//    typed-pattern enumeration is unavailable
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — shows "when pattern budget passes" caveat for over-budget case', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /when pattern budget passes/);
});

test('TuplePatternMeter — has overBudget branch that drives the caveat', () => {
  const src = read(COMPONENT_PATH);
  // The component must compute an overBudget flag and use it to gate the caveat.
  assert.match(src, /overBudget/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. Hex audit — no raw notation hex; only design-system tokens via TOKEN map
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — no raw notation hex literals', () => {
  const src = read(COMPONENT_PATH);
  // The notation-system audit blocks these specific hex codes (Tier 2/3A
  // notation colors). We assert each is absent as a string literal.
  const FORBIDDEN_NOTATION_HEXES = [
    '#F0524D', // coral
    '#64748B', // ein-w (summed)
    '#4A7CFF', // ein-v / info
    '#FA9E33', // warning
    '#23B761', // success
    '#292C2D', // gray-900
    '#5D5F60', // gray-600
  ];
  for (const hex of FORBIDDEN_NOTATION_HEXES) {
    assert.equal(
      src.includes(hex),
      false,
      `Raw notation hex ${hex} must not appear in TuplePatternMeter source — use a CSS variable instead`,
    );
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Accessibility — aria-label + cursor-help on bars
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — bars carry aria-label', () => {
  const src = read(COMPONENT_PATH);
  // Each bar track exposes its label + live value via aria-label.
  assert.match(src, /aria-label=\{`\$\{bar\.label\}/);
});

test('TuplePatternMeter — bar fill uses cursor: help for hover-tooltip affordance', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /cursor:\s*['"]help['"]/);
});

test('TuplePatternMeter — root list uses role="list" for the four bars', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /role="list"/);
});

test('TuplePatternMeter — respects prefers-reduced-motion', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /prefers-reduced-motion/);
});

test('TuplePatternMeter — live values use aria-live="polite"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /aria-live="polite"/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. App mounts <TuplePatternMeter> in §8, above <TypedPartitionDemo>
// ─────────────────────────────────────────────────────────────────────────────
test('App — imports TuplePatternMeter', () => {
  const src = read(APP_PATH);
  assert.match(src, /import TuplePatternMeter from '\.\/components\/TuplePatternMeter\.jsx'/);
});

test('App — mounts <TuplePatternMeter in §8 Typed Partition Counting section', () => {
  const src = read(APP_PATH);
  const idx8 = src.indexOf('§8 Typed Partition Counting');
  const idxMount = src.indexOf('<TuplePatternMeter');
  assert.ok(idx8 > -1, '§8 Typed Partition Counting comment not found in App');
  assert.ok(idxMount > -1, '<TuplePatternMeter mount not found in App');
  assert.ok(idxMount > idx8, '<TuplePatternMeter must appear after §8 comment');
});

test('App — TuplePatternMeter mount is positioned above <TypedPartitionDemo> within §8', () => {
  const src = read(APP_PATH);
  const idxMeter = src.indexOf('<TuplePatternMeter');
  // Find the TypedPartitionDemo mount that lives in §8 (the one inside the
  // <ExplorerSectionCard>, not the import line). We look for the JSX element.
  const idxDemo = src.indexOf('<TypedPartitionDemo', idxMeter);
  assert.ok(idxMeter > -1, '<TuplePatternMeter mount not found');
  assert.ok(idxDemo > -1, '<TypedPartitionDemo mount not found after meter');
  assert.ok(idxDemo > idxMeter, 'TuplePatternMeter must appear above TypedPartitionDemo');
});

test('App — TuplePatternMeter receives dimensionN, allLabels, groupSize, componentData props', () => {
  const src = read(APP_PATH);
  const start = src.indexOf('<TuplePatternMeter');
  assert.ok(start > -1, '<TuplePatternMeter mount not found');
  const end = src.indexOf('/>', start);
  assert.ok(end > -1, 'self-closing /> not found after <TuplePatternMeter');
  const mountBlock = src.slice(start, end + 2);
  assert.match(mountBlock, /dimensionN=/);
  assert.match(mountBlock, /allLabels=/);
  assert.match(mountBlock, /groupSize=/);
  assert.match(mountBlock, /componentData=/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. Stories file — ≥3 named exports including OverBudget
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter stories — file exists with ≥3 named exports', () => {
  const src = read(STORIES_PATH);
  const exports = src.match(/^export const \w+/gm) ?? [];
  assert.ok(
    exports.length >= 3,
    `Expected ≥3 named story exports, found ${exports.length}: ${exports.join(', ')}`,
  );
});

test('TuplePatternMeter stories — includes Default story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const Default/);
});

test('TuplePatternMeter stories — includes LargeDimension story (n=10 case)', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const LargeDimension/);
});

test('TuplePatternMeter stories — includes OverBudget story (caveat case)', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const OverBudget/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 10. Engine reuse — pulls typed-pattern counts from the existing partition
//     primitives instead of duplicating combinatorial logic
// ─────────────────────────────────────────────────────────────────────────────
test('TuplePatternMeter — imports generateTypedSetPartitions from the partition engine', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /generateTypedSetPartitions/);
  assert.match(src, /from '\.\.\/engine\/partition\/typedPartitions\.js'/);
});

test('TuplePatternMeter — imports partitionOrbitReps for the orbit count', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /partitionOrbitReps/);
});
