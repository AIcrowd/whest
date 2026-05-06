// V3.1 Appendix B — Classification-tree cases.
//
// Pins:
//   1. content/appendix/section7.ts exists, exports a SectionCopy whose
//      title references the classification cases.
//   2. section7 declares slots for at least 9 cases (B.1-B.9).
//   3. ExpressionLevelModal imports section7 and mounts an AppendixSection
//      with n={7}, anchorId="appendix-section-7", labelled "Appendix B".
//   4. DecisionLadder routes leaf clicks AND the "Full statement" tooltip
//      link to #appendix-section-7 (not #appendix-section-2 = Appendix D).
//   5. TotalCostView's α-row link (`appendixHref`) is #appendix-section-7.
//   6. UnavailableDetailsPanel's "Read Appendix B.9 →" link is
//      #appendix-section-7.
//   7. Modal mount order: Appendix B (n=7) renders before §6
//      (partition-counting theorem) so the V3.1 narrative order
//      A → B → C is preserved when the reader scrolls.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const root = new URL('./components/symmetry-aware-einsum-contractions/', import.meta.url);
const read = (relative) => fs.readFileSync(new URL(relative, root), 'utf8');

const section7Source = read('content/appendix/section7.ts');
const modalSource = read('components/ExpressionLevelModal.jsx');
const ladderSource = read('components/DecisionLadder.jsx');
const totalCostSource = read('components/TotalCostView.jsx');
const unavailableSource = read('components/UnavailableDetailsPanel.jsx');

// ─── 1. section7.ts is a SectionCopy describing Appendix B ───────────────────
test('section7.ts exports a SectionCopy whose title references the classification cases', () => {
  // Mirrors section6.ts's authoring conventions: imports SectionCopy, declares
  // a default const, and asserts the type via `satisfies SectionCopy`.
  assert.match(section7Source, /import type \{ SectionCopy \} from '\.\.\/schema'/);
  assert.match(section7Source, /satisfies SectionCopy/);
  assert.match(section7Source, /export default section7/);
  // The title must orient the reader toward the V3.1 Appendix B framing.
  assert.match(section7Source, /title:\s*'Classification-tree cases'/);
  // The deck must reference $\\alpha_a$ — this section is the canonical
  // landing page for the local accumulation count's classification.
  assert.match(section7Source, /\\\\alpha_a/);
});

// ─── 2. section7 declares slots for B.1-B.9 ──────────────────────────────────
test('section7 declares slots for nine classification-tree cases (B.1-B.9)', () => {
  // Each case has a (case<k>Label, case<k>) pair. The label slot carries the
  // "B.x — …" kicker; the body slot carries the Condition / Claim / Intuition
  // paragraphs. Both slots must be present for all nine cases.
  for (let i = 1; i <= 9; i += 1) {
    assert.match(section7Source, new RegExp(`case${i}Label:\\s*\\[`),
      `section7 must declare case${i}Label slot for B.${i}`);
    assert.match(section7Source, new RegExp(`case${i}:\\s*\\[`),
      `section7 must declare case${i} body slot for B.${i}`);
    // Body slots must mention "Condition." and "Claim." paragraphs as the
    // V3.1 narrative does — this is the contract reviewers check.
    const bodyMatch = section7Source.match(
      new RegExp(`case${i}:\\s*\\[([\\s\\S]*?)\\n\\s{4}\\],`),
    );
    assert(bodyMatch, `case${i} body slot must be parsable`);
    assert.match(bodyMatch[1], /Condition\./,
      `B.${i} body must include a "Condition." paragraph`);
    assert.match(bodyMatch[1], /Claim\./,
      `B.${i} body must include a "Claim." paragraph`);
  }
  // intro and closingNote are also required.
  assert.match(section7Source, /intro:\s*\[/);
  assert.match(section7Source, /closingNote:\s*\[/);
});

// ─── 3. Modal imports section7 and mounts the Appendix B section ─────────────
test('ExpressionLevelModal imports section7 and mounts the Appendix B section', () => {
  assert.match(modalSource, /import appendixSection7 from '\.\.\/content\/appendix\/section7\.ts'/);
  // V3.1 letter-strip restructure: the Appendix B mount now uses the new
  // sub-eyebrow label ("B") and is wrapped in <AppendixGroup letter="B" …>.
  // `n` is the sub-position WITHIN the group (n=1, since B has only one
  // section). The unique-per-section anchor is `appendix-section-7`.
  assert.match(
    modalSource,
    /<AppendixSection[\s\S]{0,400}n=\{1\}[\s\S]{0,400}label="B"[\s\S]{0,400}subEyebrow[\s\S]{0,400}anchorId="appendix-section-7"[\s\S]{0,400}title=\{appendixSection7\.title\}/,
  );
  // The new AppendixGroup wrapper provides the letter framing.
  assert.match(
    modalSource,
    /<AppendixGroup[\s\S]{0,200}letter="B"[\s\S]{0,200}title="Classification-tree cases"[\s\S]{0,200}anchorId="appendix-letter-b"/,
  );
  // The body must render at least one of the case slots — proves the
  // classification cases actually appear under the section.
  assert.match(modalSource, /appendixSection7\.slots\[`case\$\{caseIdx\}`\]/);
  assert.match(modalSource, /appendixSection7\.slots\.intro/);
  assert.match(modalSource, /appendixSection7\.slots\.closingNote/);
});

// ─── 4. DecisionLadder routes both leaf-click and tooltip to #appendix-section-7 ─
test('DecisionLadder appendixHref + click handler target #appendix-section-7', () => {
  // V3.1 "Full statement" tooltip link.
  assert.match(ladderSource, /appendixHref:\s*'#appendix-section-7'/);
  // C29 V3.1 click-leaf interaction.
  assert.match(ladderSource, /window\.location\.hash\s*=\s*'#appendix-section-7'/);
  // Guard against regression to the old D-pointing anchor in either
  // semantic position. (The doesNotMatch comments may still mention -2.)
  assert.doesNotMatch(ladderSource, /appendixHref:\s*'#appendix-section-2'/);
  assert.doesNotMatch(
    ladderSource,
    /window\.location\.hash\s*=\s*'#appendix-section-2'/,
  );
});

// ─── 5. TotalCostView α-row links to #appendix-section-7 ─────────────────────
test('TotalCostView appendixHref is #appendix-section-7', () => {
  assert.match(totalCostSource, /const appendixHref\s*=\s*'#appendix-section-7'/);
  assert.doesNotMatch(totalCostSource, /const appendixHref\s*=\s*'#appendix-section-2'/);
});

// ─── 6. UnavailableDetailsPanel B.9 link points to #appendix-section-7 ───────
test('UnavailableDetailsPanel "Read Appendix B.9" link targets #appendix-section-7', () => {
  // The link wraps the "Read Appendix B.9 →" CTA shown in the unavailable
  // budget panel. V3.1 Appendix B.9 is the unavailable-case leaf, so the
  // link must land on Appendix B (#appendix-section-7), not on Appendix C
  // (#appendix-section-6, the typed-partition theorem).
  assert.match(unavailableSource, /href="#appendix-section-7"/);
  assert.match(unavailableSource, /Read Appendix B\.9/);
  assert.doesNotMatch(unavailableSource, /href="#appendix-section-6"/);
});

// ─── 7. Modal renders Appendix B (§7) before §6 to preserve V3.1 order ───────
test('ExpressionLevelModal mounts Appendix B (n=7) before §6 (partition theorem)', () => {
  // V3.1 narrative order is A (§1-§5) → B (new §7) → C (§6, partition
  // theorem) → D (§2, dummy renamings) → E. To keep the reader's scroll
  // path matching the narrative, the new section7 mount must precede §6.
  const sec7Start = modalSource.indexOf('anchorId="appendix-section-7"');
  const sec6Start = modalSource.indexOf('anchorId="appendix-section-6"');
  assert(sec7Start > 0, '#appendix-section-7 mount not found in modal');
  assert(sec6Start > 0, '#appendix-section-6 mount not found in modal');
  assert(
    sec7Start < sec6Start,
    `Appendix B (#appendix-section-7) must mount before §6 (#appendix-section-6); ` +
      `got positions ${sec7Start} vs ${sec6Start}`,
  );
});
