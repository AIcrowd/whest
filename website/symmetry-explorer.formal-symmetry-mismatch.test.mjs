import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// V3.1 §44 Formal-Symmetry Mismatch Explorer pins:
//   - the new top-level summary caption that frames the section's theme
//     ("formal symmetry relates these terms after summation; it does not
//     make them equal products"),
//   - the V3.1 wording for each of the three alphaComparison.state branches
//     (mismatch / coincident / none),
//   - the explicit `\neq` numerical inequality that names the bilinear
//     witness (4 ≠ 6) so the reader can see the gap, not just hear about it,
//   - and the preset picker offering all three V3.1 presets
//     (bilinear-trace, direct-s2-c3, mixed-chain).

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = (...parts) => resolve(__dirname, ...parts);
const read = (...parts) => readFileSync(root(...parts), 'utf-8');

const section5 = read(
  'components/symmetry-aware-einsum-contractions/content/appendix/section5.ts',
);
const modal = read(
  'components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx',
);

test('section5 exposes the V3.1 summary caption verbatim', () => {
  assert.match(section5, /summaryCaption:\s*\[/);
  assert.match(
    section5,
    /Formal symmetry relates these terms after summation\. It does not make them equal products\./,
  );
});

test('section5 mismatchLead names the gap between formal equivalence and pointwise equality', () => {
  assert.match(section5, /mismatchLead:\s*\[/);
  assert.match(
    section5,
    /the gap is exactly the gap between formal equivalence and pointwise equality/,
  );
});

test('section5 coincidentLead uses V3.1 phrasing — formal shortcut + cost group', () => {
  assert.match(section5, /coincidentLead:\s*\[/);
  // V3.1 specifies: "This numerical coincidence does not validate the
  // formal shortcut. The valid cost group is still G_pt with output action H."
  assert.match(section5, /does not validate the formal shortcut/);
  assert.match(section5, /valid cost group is still \$G_\{\\\\text\{pt\}\}\$ with output action \$H\$/);
  // The pre-V3.1 phrasing ("does not change the rule" + "valid product-side
  // group is G_pt and the valid output-side action is H") was retired in
  // favor of the more direct V3.1 wording — it is not allowed to creep back.
  assert.doesNotMatch(section5, /does not change the rule/);
  assert.doesNotMatch(section5, /valid product-side group is/);
});

test('section5 noneLead uses V3.1 phrasing — pre vs post summation crisp axis', () => {
  assert.match(section5, /noneLead:\s*\[/);
  // V3.1 specifies: "The conceptual distinction remains: G_f is
  // post-summation; G_pt is pre-summation."
  assert.match(
    section5,
    /\$G_\{\\\\text\{f\}\}\$ is post-summation; \$G_\{\\\\text\{pt\}\}\$ is pre-summation/,
  );
  // The pre-V3.1 phrasing ("describes post-summation label-renaming
  // symmetry, while G_pt and H are the actions used by the main alpha")
  // was retired in favor of the V3.1 pre/post temporal-axis wording.
  assert.doesNotMatch(
    section5,
    /describes post-summation label-renaming symmetry, while/,
  );
});

test('section5 bilinearInequality renders the explicit 4 ≠ 6 witness in LaTeX', () => {
  assert.match(section5, /bilinearInequality:\s*\[/);
  // Bilinear trace at A=[[1,2],[3,4]]: A[0,0]A[1,1] = 1·4 = 4 and
  // A[0,1]A[1,0] = 2·3 = 6. V3.1 wants the inequality stated explicitly,
  // not just implied by the ledger. The source file is a TS literal that
  // escapes each LaTeX backslash, so `\,` shows up as `\\,` and `\neq`
  // shows up as `\\neq` in the file's text.
  assert.match(
    section5,
    /A\[0,0\]\\\\,A\[1,1\] = 4 \\\\neq 6 = A\[0,1\]\\\\,A\[1,0\]/,
  );
});

test('ExpressionLevelModal renders the new section5 slots in section §5', () => {
  // The summary caption renders once near the top of section 5 — i.e. before
  // the alphaComparison.state branches but after the intro slot.
  const section5Start = modal.indexOf('n={5}');
  assert.notEqual(section5Start, -1, 'section §5 marker missing');
  const section6Start = modal.indexOf('n={6}', section5Start + 1);
  const section5Block = modal.slice(
    section5Start,
    section6Start === -1 ? undefined : section6Start,
  );

  assert.match(section5Block, /appendixSection5\.slots\.summaryCaption/);
  assert.match(section5Block, /appendixSection5\.slots\.bilinearInequality/);
  assert.match(section5Block, /appendixSection5\.slots\.directInequality/);
  assert.match(section5Block, /appendixSection5\.slots\.mixedInequality/);

  // The summaryCaption should appear before the state-branching block, so
  // it frames the whole section rather than being buried inside one branch.
  const summaryIdx = section5Block.indexOf('appendixSection5.slots.summaryCaption');
  const mismatchBranchIdx = section5Block.indexOf("alphaComparison.state === 'mismatch'");
  const coincidentBranchIdx = section5Block.indexOf("alphaComparison.state === 'coincident'");
  const noneBranchIdx = section5Block.indexOf("alphaComparison.state === 'none'");
  assert.ok(summaryIdx >= 0, 'summaryCaption must render in section 5');
  assert.ok(summaryIdx < mismatchBranchIdx, 'summaryCaption should render before mismatch branch');
  assert.ok(summaryIdx < coincidentBranchIdx, 'summaryCaption should render before coincident branch');
  assert.ok(summaryIdx < noneBranchIdx, 'summaryCaption should render before none branch');
});

test('ExpressionLevelModal preset picker offers all three V3.1 presets', () => {
  // BURNSIDE_GAP_PRESET_IDS is the source of truth for which presets the
  // §5 picker offers; V3.1 §44 specifies bilinear-trace, direct-s2-c3, and
  // mixed-chain.
  assert.match(
    modal,
    /BURNSIDE_GAP_PRESET_IDS\s*=\s*\['bilinear-trace',\s*'direct-s2-c3',\s*'mixed-chain'\]/,
  );
  // And the picker renders BURNSIDE_GAP_PRESETS as buttons under the
  // presetPickerLabel slot in §5.
  assert.match(modal, /BURNSIDE_GAP_PRESETS\.map\(\(suggestedPreset\)/);
  assert.match(modal, /appendixSection5\.slots\.presetPickerLabel/);
});

test('ExpressionLevelModal renders the explicit numerical product values for the bilinear witness', () => {
  // V3.1 wants the actual numerical computation visible in the worked
  // example — "= 1 · 4 = 4" and "= 2 · 3 = 6" — so the reader can see the
  // gap without having to open the engine. WorkedExampleTensorProduct
  // renders scalarValues as a `· `-joined product followed by the total.
  assert.match(modal, /scalarValues=\{\[1, 4\]\}[\s\S]*?total=\{4\}/);
  assert.match(modal, /scalarValues=\{\[2, 3\]\}[\s\S]*?total=\{6\}/);
});
