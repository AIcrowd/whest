// V3.1 §40 — Cost Savings Spread (L5.T2.6 / C40).
//
// Source-grep tests pinning the V3.1 §40 contract for the new
// CostSavingsSpread component + the Section 9 consolidation contract.
// The component remains available as the original C40 card/table spread,
// but TotalCostView now presents the dense-vs-symmetry comparison once,
// using the later Cost Savings editorial spread before the formula breakdown.
//
//   1. CostSavingsSpread.jsx exists and exports a default React component.
//   2. Renders the verbatim "Dense Direct" column heading.
//   3. Renders the verbatim "Symmetry-Aware Direct" column heading.
//   4. Has data-savings-row attributes for at least 5 rows.
//   5. Renders a log-scale toggle button (Linear / Log).
//   6. Computes "Speedup" and "Savings" labels.
//   7. TotalCostView does not mount CostSavingsSpread; it mounts one
//      EditorialComparisonSpread before SectionFiveIntroBlock.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const COST_SAVINGS_SPREAD = 'components/symmetry-aware-einsum-contractions/components/CostSavingsSpread.jsx';
const TOTAL_COST_VIEW = 'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx';

// ─── 1. File exists and exports a default React component ───────────────────

test('§40 — CostSavingsSpread.jsx exists and exports a default React component', () => {
  const src = read(COST_SAVINGS_SPREAD);
  assert.match(
    src,
    /export default function CostSavingsSpread\s*\(/,
    'CostSavingsSpread must be the default export',
  );
  // JSX presence is the cheapest proof this is a React component module.
  assert.match(src, /<\w/, 'CostSavingsSpread must render JSX');
});

// ─── 2. Verbatim "Dense Direct" column heading ──────────────────────────────

test('§40 — CostSavingsSpread renders the verbatim "Dense Direct" column heading', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // Column heading is the V3.1 §40 left-column label. Pin it verbatim so
  // a refactor can't accidentally rename it to "Dense baseline" or similar.
  assert.match(
    src,
    /Dense Direct/,
    '"Dense Direct" column heading must appear verbatim',
  );
});

// ─── 3. Verbatim "Symmetry-Aware Direct" column heading ─────────────────────

test('§40 — CostSavingsSpread renders the verbatim "Symmetry-Aware Direct" column heading', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // The right-column heading. Hyphenated "Symmetry-Aware" matches the rest
  // of the §5 register (e.g. "Symmetry-Aware Direct Events" in TotalCostView).
  assert.match(
    src,
    /Symmetry-Aware Direct/,
    '"Symmetry-Aware Direct" column heading must appear verbatim',
  );
});

// ─── 4. data-savings-row attributes for at least 5 rows ─────────────────────

test('§40 — CostSavingsSpread emits data-savings-row attributes for at least 5 rows', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // The five required rows (Product chains / Updates / Active alpha method
  // / Speedup / Savings) each carry a data-savings-row="<id>" attribute so
  // tests and the savings-category bus subscribers can find them by id.
  const matches = src.match(/data-savings-row=/g) ?? [];
  assert.ok(
    matches.length >= 5,
    `Expected ≥5 data-savings-row occurrences, got ${matches.length}`,
  );
  // Sanity-check the canonical category ids appear at least once.
  for (const id of ['product-chains', 'updates', 'active-method', 'speedup', 'savings']) {
    assert.match(
      src,
      new RegExp(`data-savings-row="${id}"`),
      `data-savings-row="${id}" must be present`,
    );
  }
});

// ─── 5. Log-scale toggle button ─────────────────────────────────────────────

test('§40 — CostSavingsSpread renders a log-scale toggle button', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // The scale toggle has two options — Linear and Log. Pin both labels and
  // a <button> element with the data-savings-scale="log" attribute so the
  // toggle can be exercised by future interaction tests without a brittle
  // text-only selector.
  assert.match(src, /Linear/, '"Linear" scale label must appear');
  assert.match(src, /\bLog\b/, '"Log" scale label must appear');
  assert.match(
    src,
    /data-savings-scale="log"/,
    'log-scale button must carry data-savings-scale="log"',
  );
  // And there must be at least one <button> tag — the toggle is keyboard-
  // operable, not a styled span.
  assert.match(src, /<button[\s>]/, 'scale toggle must use a <button> element');
});

// ─── 6. Speedup and Savings labels ──────────────────────────────────────────

test('§40 — CostSavingsSpread computes "Speedup" and "Savings" labels', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // The supporting-rows table lists Speedup and Savings as separate rows
  // (in addition to the column-card numbers). Pin both labels.
  assert.match(src, /Speedup/, '"Speedup" label must appear');
  assert.match(src, /Savings/, '"Savings" label must appear');
  // Speedup formula: denseBaseline / total.
  assert.match(
    src,
    /denseBaseline\s*\/\s*total/,
    'Speedup must be computed as denseBaseline / total',
  );
  // Savings percent: 1 - total / denseBaseline.
  assert.match(
    src,
    /total\s*\/\s*denseBaseline/,
    'Savings must be computed using total / denseBaseline',
  );
});

// ─── 7. TotalCostView uses one consolidated comparison spread ───────────────

test('§40 — TotalCostView leaves CostSavingsSpread unmounted after consolidation', () => {
  const src = read(TOTAL_COST_VIEW);
  assert.doesNotMatch(
    src,
    /import CostSavingsSpread from '\.\/CostSavingsSpread\.jsx'/,
    'TotalCostView should not import the older C40 card/table spread after consolidation',
  );
  assert.doesNotMatch(src, /<CostSavingsSpread[\s\S]*?\/>/, 'TotalCostView should not mount <CostSavingsSpread />');
});

test('§40 — TotalCostView mounts one EditorialComparisonSpread after the formula glossary', () => {
  const src = read(TOTAL_COST_VIEW);
  const spreads = src.match(/<EditorialComparisonSpread[\s\S]*?\/>/g) ?? [];
  assert.equal(spreads.length, 1, 'Section 9 should render exactly one dense-vs-symmetry comparison surface');
  const recapIdx = src.indexOf('<ComponentRecap');
  const introIdx = src.indexOf('<SectionFiveIntroBlock');
  const glossaryIdx = src.indexOf('<AggregationExplainer');
  const spreadIdx = src.indexOf('<EditorialComparisonSpread');
  assert.ok(recapIdx > 0, '<ComponentRecap must remain mounted');
  assert.ok(introIdx > 0, '<SectionFiveIntroBlock must remain mounted');
  assert.ok(glossaryIdx > 0, '<AggregationExplainer must remain mounted');
  assert.ok(spreadIdx > 0, '<EditorialComparisonSpread must be mounted in TotalCostView');
  assert.ok(
    recapIdx < introIdx && introIdx < glossaryIdx && glossaryIdx < spreadIdx,
    `Render order must be ComponentRecap < SectionFiveIntroBlock < AggregationExplainer < EditorialComparisonSpread; got recapIdx=${recapIdx}, introIdx=${introIdx}, glossaryIdx=${glossaryIdx}, spreadIdx=${spreadIdx}`,
  );
});

// ─── Bonus token-discipline guard — no raw notation hex literals ────────────

test('§40 — CostSavingsSpread does not introduce raw notation hex literals', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // Token-discipline: colors must come from CSS variables on the explorer
  // theme, not from raw #RRGGBB literals.
  const bareHex = src.match(/#[0-9A-Fa-f]{6}\b/g) ?? [];
  assert.deepEqual(
    bareHex,
    [],
    `Raw hex literals found in CostSavingsSpread: ${bareHex.join(', ')}`,
  );
});

// ─── Accessibility — focusable rows + prefers-reduced-motion respected ─────

test('§40 — CostSavingsSpread rows are keyboard-focusable (tabIndex=0)', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // Every interactive savings row must be reachable by Tab — no mouse-only
  // controls. Pin tabIndex={0} so the focus contract can't drift.
  assert.match(
    src,
    /tabIndex=\{0\}/,
    'Savings rows must be focusable (tabIndex={0})',
  );
});

test('§40 — CostSavingsSpread respects prefers-reduced-motion', () => {
  const src = read(COST_SAVINGS_SPREAD);
  // The bar widths animate by default. Users who opt into reduced-motion
  // must not see the transition — pin the matchMedia hook so the gate
  // can't accidentally be deleted in a future refactor.
  assert.match(
    src,
    /prefers-reduced-motion/,
    'CostSavingsSpread must read the prefers-reduced-motion preference',
  );
});
