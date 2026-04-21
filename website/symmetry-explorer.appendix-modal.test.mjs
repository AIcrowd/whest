import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx', import.meta.url),
  'utf8',
);

test('appendix modal uses paper-appendix section titles and removes audit provenance copy', () => {
  assert.match(source, /Appendix/);
  assert.match(source, /Expression-level symmetry and output storage/);
  assert.match(source, /text-center/);
  assert.match(source, /font-serif text-\[17px\] leading-\[1\.75\] text-gray-700/);
  assert.match(source, /The distinction/);
  assert.match(source, /How the formal group is built/);
  assert.match(source, /Why Burnside on the formal group overcounts/);
  assert.match(source, /Storage-aware savings/);
  assert.match(source, /This is why the main explorer prices accumulation with the detected pointwise group rather than the larger formal symmetry group/);

  assert.doesNotMatch(source, /<NarrativeCallout label="Definition">/);
  assert.doesNotMatch(source, /<NarrativeCallout label="Interpretation on the output tensor" tone="algorithm">/);
  assert.doesNotMatch(source, /<NarrativeCallout label="Why every permutation of W is a formal symmetry" tone="algorithm">/);
  assert.doesNotMatch(source, /<NarrativeCallout label="The open optimization">/);
  assert.doesNotMatch(source, /<NarrativeCallout label={<span className={EYEBROW_CAPTION_CLASS}>Scope of the reported <Latex math="\\alpha" \/><\/span>} tone="accent">/);
  assert.doesNotMatch(source, /rounded-md border border-primary\/40 bg-primary\/5 px-5 py-4 text-\[13px\] leading-7 text-foreground/);

  assert.doesNotMatch(source, /VERBATIM, AUDIT-VERIFIED/);
  assert.doesNotMatch(source, /REVIEW_RESPONSE\.md §5/);
  assert.doesNotMatch(source, /AUDIT\.md/);
  assert.doesNotMatch(source, /empirically verified on 22 presets \+ 543 σ-checks/);
});

test('appendix modal frames storage savings as prose-led paper table rather than inset card', () => {
  assert.match(source, /The table below records the additional savings available when output storage also respects the visible-label symmetry induced by /);
  assert.match(source, /G_\{\\\\text\{pt\}\}\\\\big\|_\{V_\{\\\\mathrm\{free\}\}\}/);
  assert.match(source, /<div className="mt-5 overflow-x-auto">/);
  assert.match(source, /<thead className="border-b border-gray-200">/);
  assert.match(source, /<tbody className="\[\&_tr\]:border-b \[\&_tr\]:border-gray-100">/);
  assert.doesNotMatch(source, /Magnitude of the gap, across every preset in the explorer\./);
});
