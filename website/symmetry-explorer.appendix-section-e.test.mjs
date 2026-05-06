// V3.1 Appendix E — Scope, assumptions, and non-goals.
//
// Pins:
//   1. content/appendix/section8.ts exists, exports a SectionCopy whose
//      title references the V3.1 Appendix E framing (scope / assumptions /
//      non-goals).
//   2. section8 declares slots for all four V3.1 sub-sections (E.1 cost
//      model, E.2 included assumptions, E.3 excluded phenomena, E.4
//      exactness contract).
//   3. The intro slot orients the reader by referencing scope or
//      assumptions — this is the canonical landing entry point when the
//      reader asks "what does this number mean?"
//   4. The excludedPhenomena slot lists wall-clock runtime explicitly —
//      runtime claims are the most common scope misunderstanding the
//      explorer has to defuse.
//   5. The excludedPhenomena slot lists memory traffic explicitly —
//      same reason as wall-clock runtime; locking it in via a test
//      prevents accidental drift if the bullets are reordered or
//      paraphrased.
//   6. The exactnessContract paragraph contains the literal phrase
//      "exact direct cost" — the central V3.1 claim equation hinges on
//      that phrase.
//   7. ExpressionLevelModal imports section8 and mounts an
//      AppendixSection with n={8}, anchorId="appendix-section-8",
//      labelled "Appendix E".

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const root = new URL('./components/symmetry-aware-einsum-contractions/', import.meta.url);
const read = (relative) => fs.readFileSync(new URL(relative, root), 'utf8');

const section8Source = read('content/appendix/section8.ts');
const modalSource = read('components/ExpressionLevelModal.jsx');

// ─── 1. section8.ts is a SectionCopy describing Appendix E ───────────────────
test('section8.ts exports a SectionCopy with an E-relevant title', () => {
  // Mirrors section6/section7 authoring conventions: imports SectionCopy,
  // declares a default const, and asserts the type via `satisfies SectionCopy`.
  assert.match(section8Source, /import type \{ SectionCopy \} from '\.\.\/schema'/);
  assert.match(section8Source, /satisfies SectionCopy/);
  assert.match(section8Source, /export default section8/);
  // The title must surface the V3.1 Appendix E framing — the reader scans
  // the appendix nav and needs to recognise this as the scope/contract
  // landing page, not yet another classification or theorem block.
  assert.match(section8Source, /title:\s*'Scope, assumptions, and non-goals'/);
});

// ─── 2. section8 declares slots for E.1–E.4 ──────────────────────────────────
test('section8 declares slots for all four V3.1 sub-sections (E.1–E.4)', () => {
  // E.1 — cost model.
  assert.match(section8Source, /costModel:\s*\[/,
    'section8 must declare a costModel slot for E.1');
  // E.2 — included assumptions (the list itself; lead + label may also exist).
  assert.match(section8Source, /includedAssumptions:\s*\[/,
    'section8 must declare an includedAssumptions slot for E.2');
  // E.3 — excluded phenomena.
  assert.match(section8Source, /excludedPhenomena:\s*\[/,
    'section8 must declare an excludedPhenomena slot for E.3');
  // E.4 — exactness contract.
  assert.match(section8Source, /exactnessContract:\s*\[/,
    'section8 must declare an exactnessContract slot for E.4');
  // intro is also required; it sets up the section before the four sub-blocks.
  assert.match(section8Source, /intro:\s*\[/);
});

// ─── 3. intro orients the reader toward scope/assumptions ────────────────────
test('section8.ts intro mentions scope or assumptions', () => {
  // The intro slot is the modal-scroll equivalent of the V3.1 narrative's
  // section preamble. It must explicitly cue the reader that this section
  // is about the explorer's *contract*, not another piece of math — a
  // surprised reader who lands here should recognise immediately why
  // they were sent here.
  const introMatch = section8Source.match(/intro:\s*\[([\s\S]*?)\n\s{4}\],/);
  assert(introMatch, 'intro slot must be parsable');
  assert.match(
    introMatch[1],
    /scope|assumption/i,
    'intro must reference scope or assumptions to orient the reader',
  );
});

// ─── 4. excludedPhenomena lists wall-clock runtime ───────────────────────────
test('section8.ts excludedPhenomena lists wall-clock runtime', () => {
  const excludedMatch = section8Source.match(/excludedPhenomena:\s*\[([\s\S]*?)\n\s{4}\],/);
  assert(excludedMatch, 'excludedPhenomena slot must be parsable');
  // The most common reader confusion is treating reported numbers as
  // wall-clock predictions; pinning the bullet here prevents accidental
  // drop or paraphrase that would weaken the contract.
  assert.match(
    excludedMatch[1],
    /wall-clock runtime/,
    'excludedPhenomena must list "wall-clock runtime" verbatim',
  );
});

// ─── 5. excludedPhenomena lists memory traffic ───────────────────────────────
test('section8.ts excludedPhenomena lists memory traffic', () => {
  const excludedMatch = section8Source.match(/excludedPhenomena:\s*\[([\s\S]*?)\n\s{4}\],/);
  assert(excludedMatch, 'excludedPhenomena slot must be parsable');
  // Memory traffic is the second-most-common scope misunderstanding —
  // readers benchmark memory-bound kernels and assume the count tracks
  // bytes moved. Lock the bullet in.
  assert.match(
    excludedMatch[1],
    /memory traffic/,
    'excludedPhenomena must list "memory traffic" verbatim',
  );
});

// ─── 6. exactnessContract paragraph contains "exact direct cost" ─────────────
test('section8.ts exactnessContract paragraph includes "exact direct cost"', () => {
  const contractMatch = section8Source.match(/exactnessContract:\s*\[([\s\S]*?)\n\s{4}\],/);
  assert(contractMatch, 'exactnessContract slot must be parsable');
  // The central V3.1 claim is
  //   exact direct cost = product-representative multiplication chains
  //                     + projection-induced accumulation updates.
  // The phrase "exact direct cost" anchors that claim; if it drifts to
  // "exact cost" or similar, the equation in the modal stops matching the
  // narrative and the contract weakens silently.
  assert.match(
    contractMatch[1],
    /exact direct cost/,
    'exactnessContract must contain the literal phrase "exact direct cost"',
  );
});

// ─── 7. Modal imports section8 and mounts the Appendix E section ─────────────
test('ExpressionLevelModal imports section8 and mounts the Appendix E section', () => {
  assert.match(modalSource, /import appendixSection8 from '\.\.\/content\/appendix\/section8\.ts'/);
  // V3.1 letter-strip restructure: the Appendix E mount now uses the new
  // sub-eyebrow label ("E") and is wrapped in <AppendixGroup letter="E" …>.
  // `n` is the sub-position WITHIN the group (n=1, since E has only one
  // section). The unique-per-section anchor is `appendix-section-8`.
  assert.match(
    modalSource,
    /<AppendixSection[\s\S]{0,400}n=\{1\}[\s\S]{0,400}label="E"[\s\S]{0,400}subEyebrow[\s\S]{0,400}anchorId="appendix-section-8"[\s\S]{0,400}title=\{appendixSection8\.title\}/,
  );
  // The new AppendixGroup wrapper provides the letter framing.
  assert.match(
    modalSource,
    /<AppendixGroup[\s\S]{0,200}letter="E"[\s\S]{0,200}title="Scope, assumptions, and non-goals"[\s\S]{0,200}anchorId="appendix-letter-e"/,
  );
  // The body must reference the four E.x slots — proves the four
  // sub-sections actually appear under the section in render order.
  assert.match(modalSource, /appendixSection8\.slots\.intro/);
  assert.match(modalSource, /appendixSection8\.slots\.costModel/);
  assert.match(modalSource, /appendixSection8\.slots\.includedAssumptions/);
  assert.match(modalSource, /appendixSection8\.slots\.excludedPhenomena/);
  assert.match(modalSource, /appendixSection8\.slots\.exactnessContract/);
});
