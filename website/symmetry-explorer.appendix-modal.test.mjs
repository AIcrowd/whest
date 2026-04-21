import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx', import.meta.url),
  'utf8',
);
const formalSource = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/AppendixFormalBlock.jsx', import.meta.url),
  'utf8',
);

test('appendix modal shell is wider paper editorial rail and removes audit provenance copy', () => {
  assert.match(source, /relative w-full max-w-\[(1460px|var\(--content-max\))\] rounded-lg border border-gray-200 bg-white shadow-2xl/);
  assert.doesNotMatch(source, /max-w-5xl/);
  assert.match(source, /appendixRailClass = 'mx-auto w-full max-w-\[(1460px|var\(--content-max\))\] px-6 md:px-8 lg:px-10'/);
  assert.match(source, /Appendix/);
  assert.match(source, /Expression-level symmetry and output storage/);
  assert.match(source, /text-center/);
  assert.match(source, /font-serif text-\[17px\] leading-\[1\.75\] text-gray-700/);
  assert.match(source, /The distinction/);
  assert.match(source, /How the formal group is built/);
  assert.match(source, /Why Burnside on the formal group overcounts/);
  assert.match(source, /Storage-aware savings/);
  assert.match(source, /The distinction from Section 1 becomes operational at the row level/);
  assert.match(source, /This is the boundary used by the rest of the appendix/);

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

test('appendix sections adopt two-column divider rhythm and Section 2 is a formal bridge block', () => {
  assert.match(source, /editorial-two-col-divider-md grid gap-y-4 gap-x-8 md:grid-cols-2/);
  assert.match(source, /editorial-two-col-divider-lg grid gap-y-4 gap-x-8 lg:grid-cols-2/);
  assert.match(source, /<AppendixTwoColBlock/);
  assert.match(source, /<div className=\"mt-8\">/);
  assert.match(source, /<AppendixFormalBlock>/);
  assert.match(source, />\s*Setup\s*</);
  assert.match(source, />\s*Proposition\s*</);
  assert.match(source, />\s*Detection Principle\s*</);
  assert.match(source, /\\pi_\\sigma \\neq \\mathrm\{id\}/);
  assert.match(source, /\\pi_\\sigma = \\mathrm\{id\}/);
  assert.match(source, /Among the admissible induced relabelings/);
  assert.match(source, /row-witnessed visible-label action inherited from the detected pointwise symmetry group/);
  assert.match(source, /Section 3 extracts the visible-label action/);
  assert.doesNotMatch(source, /derivePi/);
  assert.doesNotMatch(source, /<AppendixTheoremBlock/);
  assert.doesNotMatch(source, /<AppendixProofBlock>/);

  assert.match(
    source,
    /The table below records the additional savings available when output storage also respects the visible-label symmetry induced by /,
  );
  assert.match(source, /G_\{\\\\text\{pt\}\}\\\\big\|_\{V_\{\\\\mathrm\{free\}\}\}/);
  assert.match(source, /<div className=\"mt-5 overflow-x-auto\">/);
  assert.match(source, /<thead className=\"border-b border-gray-200\">/);
  assert.match(source, /<tbody/);
  assert.ok(source.includes('[&_tr]:border-b [&_tr]:border-gray-100'));
  assert.doesNotMatch(source, /Magnitude of the gap, across every preset in the explorer\./);
});

test('appendix sections 1-5 now read as a linear argument rather than early foreshadowing', () => {
  assert.match(source, /The appendix question is therefore where that boundary appears in the structure of the contraction itself/);
  assert.match(source, /Section 2 answers it at the row level, by showing which candidate row moves induce genuine relabelings of the contraction and which do not/);
  assert.doesNotMatch(source, /These groups are related through \$G_\{\\text\{pt\}\}\\big\|_\{V_\{\\mathrm\{free\}\}\}/);
  assert.doesNotMatch(source, /That induced action becomes the \$V_\{\\mathrm\{free\}\}\$-factor of \$G_\{\\text\{f\}\}/);

  assert.match(source, /Among the admissible induced relabelings/);
  assert.match(source, /This is the row-witnessed visible-label action inherited from the detected pointwise symmetry group/);

  assert.match(source, /Section 3 isolated the visible-label action that remains pointwise on the output tensor/);
  assert.match(source, /The complementary factor comes from the summed labels alone/);

  assert.match(source, /Sections 3 and 4 have now isolated the two ingredients of the formal symmetry story/);
  assert.match(source, /We now combine them/);
  assert.match(source, /The set of all such lifts is/);
});

test('section 2 handoff paragraph spans the full appendix row width', () => {
  assert.doesNotMatch(
    source,
    /<p className="mt-5 max-w-\[72ch\] font-serif text-\[17px\] leading-\[1\.75\] text-gray-700">[\s\S]*This is the boundary used by the rest of the appendix/,
  );
  assert.match(source, /This is the boundary used by the rest of the appendix/);
});

test('appendix formal block is a full-width black-outline box', () => {
  assert.match(formalSource, /w-full rounded-\[?[^\" ]*\]? border border-black bg-white px-6 py-6/);
  assert.doesNotMatch(formalSource, /border-l-2 border-stone-300/);
  assert.doesNotMatch(source, /<div className=\"mt-8 grid gap-6 md:gap-8 lg:gap-10\">/);
  assert.match(source, /h-px bg-gray-200/);
  assert.match(source, /marker:font-semibold marker:text-gray-600/);
});

test('storage-aware appendix prose uses a single full-width block and drops the redundant right-column note', () => {
  assert.match(source, /<div className=\{`mt-4 space-y-4 \$\{APPENDIX_PROSE_JUSTIFIED_CLASS\}`\}>/);
  assert.match(source, /contributes nothing at the storage level/);
  assert.doesNotMatch(
    source,
    /<AppendixTwoColBlock[\s\S]*The \$S\(W_\{\\mathrm\{summed\}\}\)\$ factor of \$G_\{\\text\{f\}\}\$ contributes nothing at the storage level[\s\S]*right=\{[\s\S]*×k[\s\S]*<\/AppendixTwoColBlock>/,
  );
});

test('appendix einsum hover tooltip escapes the scroll-clipped table via a portal', () => {
  assert.match(source, /import \{ createPortal \} from 'react-dom';/);
  assert.match(source, /function AppendixEinsumHoverCell\(/);
  assert.match(source, /function AppendixPresetHoverLabel\(/);
  assert.match(source, /getBoundingClientRect\(\)/);
  assert.match(source, /createPortal\(/);
  assert.match(source, /pointer-events-none fixed z-\[9999\]/);
  assert.match(source, /<AppendixEinsumHoverCell[\s\S]*subs=\{subs\}[\s\S]*output=\{output\}[\s\S]*preset=\{preset\}[\s\S]*groupLabel=\{groupLabel\}[\s\S]*\/>/);
  assert.match(source, /Hover any preset name or einsum to see the full construction/);
  assert.doesNotMatch(source, /group-hover:block/);
});

test('appendix typography is organized around explicit shared editorial registers', () => {
  assert.match(source, /const APPENDIX_PROSE_CLASS = 'font-serif text-\[17px\] leading-\[1\.75\] text-gray-700';/);
  assert.match(source, /const APPENDIX_PROSE_JUSTIFIED_CLASS = `\$\{APPENDIX_PROSE_CLASS\} text-justify`;/);
  assert.match(source, /const APPENDIX_KICKER_CLASS = 'text-\[11px\] font-semibold uppercase tracking-\[0\.16em\] text-gray-500';/);
  assert.match(source, /const APPENDIX_FOOTNOTE_CLASS = 'text-\[11px\] italic text-muted-foreground';/);
});
