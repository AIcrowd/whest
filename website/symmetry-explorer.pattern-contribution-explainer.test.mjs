// V3.1 §34 — C34 Pattern Contribution Explainer.
//
// Pins the per-pattern detail card and the reverse-direction hover bus on
// TypedPartitionDemo. C36 wires formula → table (hovering a formula term
// tints the matching column); C34 is the reverse arrow — hovering a
// labeled detail field or a per-pattern column header tints the matching
// term in the formula card above. C34 also adds a "compare brute force"
// toggle that expands a tiny per-n tuple-count table for n = 2..6 so
// readers can see how partition counting compresses raw enumeration.
//
// Why source-grep instead of a browser test: the detail card and the
// reverse-hover handlers are pure JSX wiring with no engine logic.
// Source-grep keeps the test light and locks the structural contract
// (state, fields, labels, attributes) so reviewers can spot drift in a
// one-line diff.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL(
    './components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx',
    import.meta.url,
  ),
  'utf8',
);

// ─── 1. selectedPatternIdx is declared ───────────────────────────────────────
test('TypedPartitionDemo declares selectedPatternIdx (derived from selectedPatternKey)', () => {
  // The V3.1 §34 spec calls for a `selectedPatternIdx` so the detail card
  // can name "Selected pattern detail · idx N" and downstream consumers
  // can index into the chip array. We derive it from the canonical
  // selectedPatternKey state (which already drives chip clicks and row
  // hovers) so the two names stay in lockstep — pin the derivation so a
  // future rename of either name fails fast.
  assert.match(
    source,
    /const selectedPatternIdx = chips\.findIndex\(\(chip\) => chip\.key === selectedPatternKey\)/,
    'TypedPartitionDemo must declare selectedPatternIdx as a derivation off chips + selectedPatternKey',
  );
  assert.match(
    source,
    /idx \{selectedPatternIdx >= 0 \? selectedPatternIdx : 0\}/,
    'Selected pattern detail eyebrow must surface selectedPatternIdx (with a >= 0 fallback to 0)',
  );
});

// ─── 2. V3.1 §34 6-line labeled fields ───────────────────────────────────────
test('TypedPartitionDemo renders the V3.1 §34 6-line labeled detail fields', () => {
  // The V3.1 §34 spec dictates this exact 6-line format so the detail
  // card mirrors the four formula terms plus the contribution arithmetic.
  // Pinning the labels keeps the card readable as a "decoder" for the
  // formula header — drift on either side breaks the analogy.
  assert.match(source, />Pattern:</, 'Detail card must render "Pattern:" label');
  assert.match(source, />Blocks:</, 'Detail card must render "Blocks:" label');
  assert.match(
    source,
    />Concrete labelings:</,
    'Detail card must render "Concrete labelings:" label',
  );
  assert.match(
    source,
    />Induced block-symmetry divisor:</,
    'Detail card must render "Induced block-symmetry divisor:" label',
  );
  assert.match(source, />Output reach:</, 'Detail card must render "Output reach:" label');
  assert.match(
    source,
    />Contribution:</,
    'Detail card must render "Contribution:" label',
  );
  // The detail card itself carries a stable testid so a future migration
  // can grab the panel without reaching into the prose around it.
  assert.match(
    source,
    /data-testid="pattern-contribution-explainer"/,
    'Detail card must carry data-testid="pattern-contribution-explainer"',
  );
});

// ─── 3. Reverse hover: per-pattern column header sets formulaTermHover ───────
test('Per-pattern column headers set formulaTermHover on hover/focus (reverse direction)', () => {
  // C34 is the reverse arrow of C36. Each of the four <th> column headers
  // wires onMouseEnter / onFocus → setFormulaTermHover('<term>') so
  // hovering a column tints the matching term in the formula above.
  // tabIndex={0} keeps the contract reachable for keyboard users; pin
  // the four (term, handler) pairs so a careless edit can't drop one.
  assert.match(
    source,
    /onMouseEnter=\{\(\) => setFormulaTermHover\('pattern'\)\}/,
    'pattern column header must set formulaTermHover to "pattern" on hover',
  );
  assert.match(
    source,
    /onMouseEnter=\{\(\) => setFormulaTermHover\('fallingFactorial'\)\}/,
    'fallingFactorial column header must set formulaTermHover to "fallingFactorial" on hover',
  );
  assert.match(
    source,
    /onMouseEnter=\{\(\) => setFormulaTermHover\('divisor'\)\}/,
    'divisor column header must set formulaTermHover to "divisor" on hover',
  );
  assert.match(
    source,
    /onMouseEnter=\{\(\) => setFormulaTermHover\('outputReach'\)\}/,
    'outputReach column header must set formulaTermHover to "outputReach" on hover',
  );
  // Keyboard parity. onFocus must mirror onMouseEnter so tab-cycling
  // through the headers fires the same hover bus events.
  assert.match(
    source,
    /onFocus=\{\(\) => setFormulaTermHover\('pattern'\)\}/,
    'pattern column header must set formulaTermHover to "pattern" on focus (keyboard parity)',
  );
  assert.match(
    source,
    /onFocus=\{\(\) => setFormulaTermHover\('outputReach'\)\}/,
    'outputReach column header must set formulaTermHover to "outputReach" on focus (keyboard parity)',
  );
});

// ─── 4. "Compare brute force" toggle button ─────────────────────────────────
test('"Compare brute force" toggle button exists with a stable data attribute and aria-label', () => {
  // The toggle gates the brute-force compare table. Pin the data-action
  // hook (so QA / future tests can locate it without selector churn), the
  // aria-label (AT users get the same "expand / collapse" cue sighted
  // users get from the chevron-less text label), and the aria-expanded
  // attribute (so AT can read whether the panel is currently open).
  assert.match(
    source,
    /data-action="toggle-brute-force"/,
    'Toggle button must carry data-action="toggle-brute-force"',
  );
  assert.match(
    source,
    /aria-label=\{bruteForceOpen \? 'Hide the brute-force tuple comparison table' : 'Compare brute force — show tuple count for n = 2 through 6'\}/,
    'Toggle button must carry an aria-label that flips with bruteForceOpen state',
  );
  assert.match(
    source,
    /aria-expanded=\{bruteForceOpen\}/,
    'Toggle button must announce expanded state via aria-expanded',
  );
  assert.match(
    source,
    /const \[bruteForceOpen, setBruteForceOpen\] = useState\(false\)/,
    'TypedPartitionDemo must declare bruteForceOpen via useState(false)',
  );
});

// ─── 5. Brute-force expansion renders tuple counts for n = 2..6 ─────────────
test('"Compare brute force" expansion renders tuple counts for n = 2..6', () => {
  // The expanded panel uses BRUTE_FORCE_NS = [2,3,4,5,6] so readers see
  // five concrete data points before the falling-factorial tails off into
  // overflow territory. Pin the constant + the per-row data attribute so
  // a future rename of the row hook fails fast, and pin the container
  // testid so QA can locate the table.
  assert.match(
    source,
    /const BRUTE_FORCE_NS = \[2, 3, 4, 5, 6\]/,
    'BRUTE_FORCE_NS must be the list [2, 3, 4, 5, 6]',
  );
  assert.match(
    source,
    /data-testid="brute-force-compare"/,
    'Brute-force compare panel must carry data-testid="brute-force-compare"',
  );
  assert.match(
    source,
    /data-brute-force-row=\{row\.n\}/,
    'Each brute-force row must emit data-brute-force-row with the per-n value',
  );
  // Both columns of the table must surface — (n)_b for the equality
  // pattern's tuple count, and n^k for the raw enumeration cost. The
  // ratio is the saving the equality pattern alone gives, before the |Ḡ|
  // / |A_p̃/H_a| collapse compresses further.
  assert.match(
    source,
    />tuples in pattern \(n\)_b</,
    'Brute-force table must label the (n)_b column "tuples in pattern (n)_b"',
  );
  assert.match(
    source,
    />raw enum n\^k</,
    'Brute-force table must label the n^k column "raw enum n^k"',
  );
  // Pin the per-n compute helper so the math behind each row is stable.
  assert.match(
    source,
    /function bruteForceTupleCount\(partition, n\)/,
    'bruteForceTupleCount(partition, n) helper must exist',
  );
  assert.match(
    source,
    /return fallingFactorial\(n, blocks\)/,
    'bruteForceTupleCount must return fallingFactorial(n, blocks)',
  );
});

// ─── 6. Detail card has cursor-help on each labeled field ───────────────────
test('Selected pattern detail card has cursor-help on each labeled field', () => {
  // Each of the six detail rows is a hover surface — cursor-help on the
  // mouse path, tabIndex={0} on the keyboard path, an aria-label naming
  // what hover will highlight, and (for the four formula-mapped fields)
  // an onMouseEnter / onFocus handler that sets formulaTermHover. Pin the
  // five data-detail-field hooks (pattern, blocks, fallingFactorial,
  // divisor, outputReach, contribution) plus the cursor-help class so
  // the hover affordance never drops silently.
  assert.match(
    source,
    /data-detail-field="pattern"/,
    'Detail card must mark the Pattern field with data-detail-field="pattern"',
  );
  assert.match(
    source,
    /data-detail-field="blocks"/,
    'Detail card must mark the Blocks field with data-detail-field="blocks"',
  );
  assert.match(
    source,
    /data-detail-field="fallingFactorial"/,
    'Detail card must mark the Concrete labelings field with data-detail-field="fallingFactorial"',
  );
  assert.match(
    source,
    /data-detail-field="divisor"/,
    'Detail card must mark the divisor field with data-detail-field="divisor"',
  );
  assert.match(
    source,
    /data-detail-field="outputReach"/,
    'Detail card must mark the output reach field with data-detail-field="outputReach"',
  );
  assert.match(
    source,
    /data-detail-field="contribution"/,
    'Detail card must mark the contribution field with data-detail-field="contribution"',
  );
  // Each of the six fields carries cursor-help (mouse affordance) and
  // tabIndex={0} (keyboard reach). Six occurrences of cursor-help inside
  // the detail card is the floor — header + four formula-mapped fields +
  // contribution = 6.
  const cursorHelpMatches = source.match(/cursor-help/g) ?? [];
  assert.ok(
    cursorHelpMatches.length >= 6,
    `expected at least 6 cursor-help affordances (one per detail field + reused on column headers); found ${cursorHelpMatches.length}`,
  );
  // describeBlocks helper renders the human-readable block list (e.g.
  // "{a,b}, {c}") for the Blocks field. Pin its existence so the field
  // never falls back to a raw `[0,0,1]` partition array on the page.
  assert.match(
    source,
    /function describeBlocks\(partition, labels\)/,
    'describeBlocks(partition, labels) helper must exist for the Blocks field',
  );
});
