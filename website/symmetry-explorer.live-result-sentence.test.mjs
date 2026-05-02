// V3.1 §41 — Live Result Sentence (L5.T2.1).
//
// Source-grep tests pinning the V3.1 §41 prose contract for
// LiveResultSentence plus the Section 9 consolidation contract. The sentence
// remains available as the original §41 prose component, but TotalCostView now
// avoids repeating the dense-vs-symmetry result before the Cost Savings spread.
// Each test pins one piece of the component/spec contract so cosmetic edits
// survive but structural drift trips immediately.
//
//   1. LiveResultSentence file exists and exports a default React component.
//   2. Verbatim phrase "For the selected contraction" (line 1).
//   3. Verbatim phrase "Therefore total" (line 3).
//   4. Verbatim phrase "this is a" (line 4 savings clause).
//   5. V3.1 unavailable-explanation sentence renders when applicable.
//   6. aria-live region present on at least one value-bearing span.
//   7. TotalCostView does not mount LiveResultSentence after Section 9
//      consolidation; the EditorialComparisonSpread is the single summary
//      before the formula card.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const LIVE_RESULT_SENTENCE = 'components/symmetry-aware-einsum-contractions/components/LiveResultSentence.jsx';
const TOTAL_COST_VIEW = 'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx';

// ─── 1. File exists and exports a default React component ───────────────────

test('§41 — LiveResultSentence.jsx exists and exports a default React component', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // Default export of a function component named LiveResultSentence.
  assert.match(
    src,
    /export default function LiveResultSentence\s*\(/,
    'LiveResultSentence must be the default export',
  );
  // Renders JSX (component returns markup) — the file should contain JSX
  // tag syntax, the cheapest proof that this is a React component module.
  assert.match(src, /<\w/, 'LiveResultSentence must render JSX');
});

// ─── 2. Verbatim phrase "For the selected contraction" (line 1) ─────────────

test('§41 — LiveResultSentence renders the verbatim phrase "For the selected contraction"', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // The opening clause anchors the V3.1 §41 sentence template — readers
  // recognise the section by this phrase. Drift here breaks the contract.
  assert.match(
    src,
    /For the selected contraction/,
    'Line 1 of the §41 template must appear verbatim',
  );
});

// ─── 3. Verbatim phrase "Therefore total" (line 3) ──────────────────────────

test('§41 — LiveResultSentence renders the verbatim phrase "Therefore total"', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // Line 3 of the template — the synthesis step "total = mu + alpha". This
  // is the one line every reader is here to see; pin it verbatim so a
  // refactor can't accidentally drop or rename it.
  assert.match(
    src,
    /Therefore total/,
    'Line 3 of the §41 template ("Therefore total") must appear verbatim',
  );
});

// ─── 4. Verbatim phrase "this is a" (line 4 savings clause) ─────────────────

test('§41 — LiveResultSentence renders the verbatim phrase "this is a" (savings clause)', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // Line 4 closes the sentence with "this is a [N%] reduction." The phrase
  // "this is a" anchors the savings comparison to the preceding dense
  // baseline so the percent isn't reported in the abstract.
  assert.match(
    src,
    /this is a/,
    'Line 4 savings clause must include the verbatim phrase "this is a"',
  );
  // And the word "reduction" — the comparison is framed as a reduction,
  // not a "speedup" or "savings"; the spec is explicit on this wording.
  assert.match(
    src,
    /reduction/,
    'Line 4 savings clause must use the word "reduction"',
  );
});

// ─── 5. V3.1 unavailable-explanation sentence renders when applicable ───────

test('§41 — LiveResultSentence renders the unavailable-explanation phrase when α is unavailable', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // V3.1 §41 unavailable line:
  //   "The exact alpha count is unavailable for this component under the
  //    current interactive budget because [reason]."
  // Pin the lead phrase + the "current interactive budget" anchor so the
  // sentence can't drift to a different wording silently.
  assert.match(
    src,
    /The exact alpha count is unavailable/,
    'unavailable explanation must lead with "The exact alpha count is unavailable"',
  );
  assert.match(
    src,
    /current interactive budget/,
    'unavailable explanation must mention the "current interactive budget"',
  );
  // The sentence is gated by the componentUnavailable prop — assert the
  // prop is referenced and used to switch between alpha line and the
  // unavailable line.
  assert.match(src, /componentUnavailable/);
});

// ─── 6. aria-live region present on a value-bearing span ────────────────────

test('§41 — LiveResultSentence has an aria-live region on the value-bearing area', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // The values (μ, α, total, % reduction) update whenever the dimension
  // knob fires. aria-live="polite" announces those updates to screen
  // readers without interrupting other speech.
  assert.match(
    src,
    /aria-live="polite"/,
    'LiveResultSentence must mark its value-bearing region with aria-live="polite"',
  );
});

// ─── 7. TotalCostView uses the consolidated Cost Savings spread ─────────────

test('§41 — TotalCostView leaves LiveResultSentence unmounted after consolidation', () => {
  const src = read(TOTAL_COST_VIEW);
  assert.doesNotMatch(
    src,
    /import LiveResultSentence from '\.\/LiveResultSentence\.jsx'/,
    'TotalCostView should not import the standalone prose sentence after consolidation',
  );
  assert.doesNotMatch(src, /<LiveResultSentence[\s\S]*?\/>/, 'TotalCostView should not mount <LiveResultSentence />');
});

test('§41 — TotalCostView renders the consolidated Cost Savings spread above the formula card', () => {
  const src = read(TOTAL_COST_VIEW);
  const spreadIdx = src.indexOf('<EditorialComparisonSpread');
  const introIdx = src.indexOf('<SectionFiveIntroBlock');
  assert.ok(spreadIdx > 0, '<EditorialComparisonSpread must be mounted in TotalCostView');
  assert.ok(introIdx > 0, '<SectionFiveIntroBlock (the formula card) must remain mounted');
  assert.ok(
    spreadIdx < introIdx,
    `<EditorialComparisonSpread must render before <SectionFiveIntroBlock; got spreadIdx=${spreadIdx}, introIdx=${introIdx}`,
  );
});

// ─── Bonus token-discipline guard — no raw notation hex literals ────────────

test('§41 — LiveResultSentence does not introduce raw notation hex literals', () => {
  const src = read(LIVE_RESULT_SENTENCE);
  // Token-discipline: colors must come from CSS variables on the explorer
  // theme, not from raw #RRGGBB literals. (Per the migration's standing
  // rule against raw notation hex outside textcolor helpers.)
  const bareHex = src.match(/#[0-9A-Fa-f]{6}\b/g) ?? [];
  assert.deepEqual(
    bareHex,
    [],
    `Raw hex literals found in LiveResultSentence: ${bareHex.join(', ')}`,
  );
});
