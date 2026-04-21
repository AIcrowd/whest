import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx', import.meta.url),
  'utf8',
);

test('appendix modal shell is wider paper editorial rail and removes audit provenance copy', () => {
  assert.match(source, /relative w-full max-w-\[(1460px|var\(--content-max\))\] rounded-lg border border-gray-200 bg-white shadow-2xl/);
  assert.doesNotMatch(source, /max-w-5xl/);
  assert.match(source, /appendixRailClass = 'mx-auto w-full max-w-\[(1460px|var\(--content-max\))\] px-6 md:px-8 lg:px-10'/);
  assert.match(source, /Appendix/);
  assert.match(source, /Expression-level symmetry and symmetry aware storage/);
  assert.match(source, /text-\[10px\] font-semibold uppercase text-gray-400/);
  assert.match(source, /h-px w-8 align-middle bg-gray-300/);
  assert.match(source, /fontSize: 'clamp\(36px, 5vw, 52px\)'/);
  assert.match(source, /fontVariationSettings: "'opsz' 72"/);
  assert.match(source, /fontVariationSettings: "'opsz' 18"/);
  assert.match(source, /max-w-\[min\(100%,980px\)\] text-\[17px\] italic text-gray-600/);
  assert.match(source, /The distinction/);
  assert.match(source, /How the formal group is built/);
  assert.match(source, /Why Burnside on the formal group overcounts/);
  assert.match(source, /Storage-aware savings/);
  assert.match(source, /At the row level, the question is whether a candidate row move/);
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

test('appendix section 2 is rebuilt as one opener, one compact table, two featured outcomes, and one formal takeaway shelf', () => {
  assert.match(source, /editorial-two-col-divider-md grid gap-y-4 gap-x-8 md:grid-cols-2/);
  assert.match(source, /editorial-two-col-divider-lg grid gap-y-4 gap-x-8 lg:grid-cols-2/);
  assert.match(source, /<AppendixTwoColBlock/);
  assert.match(source, /Outcome summary/);
  assert.match(source, /Featured outcomes/);
  assert.match(source, /Frobenius outcome/);
  assert.match(source, /Triangle outcome/);
  assert.match(source, /Formal takeaway/);
  assert.match(source, /\\pi_\\sigma \\neq \\mathrm\{id\}/);
  assert.match(source, /\\pi_\\sigma = \\mathrm\{id\}/);
  assert.match(source, /A row move \$\$\{notationLatex\('sigma_row_move'\)\}/);
  assert.match(source, /is admissible when the column fingerprints of \$M_/);
  assert.match(source, /If every admissible .* Frobenius-class presets/);
  assert.match(source, /Record every admissible/);
  assert.match(source, /Section 3 extracts the visible-label action/);
  assert.doesNotMatch(source, /<AppendixFormalBlock>/);
  assert.doesNotMatch(source, />\s*Setup\s*</);
  assert.doesNotMatch(source, />\s*Proposition\s*</);
  assert.doesNotMatch(source, />\s*Detection Principle\s*</);
  assert.doesNotMatch(source, /<ul className=\{`space-y-1\.5 \$\{APPENDIX_SMALL_TEXT_CLASS\}`\}>/);
  assert.doesNotMatch(source, /identity only/);
  assert.doesNotMatch(source, /all three outcomes visible/);
  assert.doesNotMatch(source, /rounded-md border-l-4 border-border\/60 bg-muted\/20 px-4 py-3/);
  assert.doesNotMatch(source, /derivePi/);

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
  assert.match(source, /This is the part of \$G_\{\\\\text\{pt\}\}\$ that still acts on the computed output tensor/);

  assert.match(source, /Section 3 isolated the visible-label action that remains pointwise on the output tensor/);
  assert.match(source, /The complementary factor comes from the summed labels alone/);

  assert.match(source, /Sections 3 and 4 have now isolated the two ingredients of the formal symmetry story/);
  assert.match(source, /We now combine them/);
  assert.match(source, /The set of all such lifts is/);
});

test('appendix section 3 is the visible-action extraction chapter with reduced bilinear-trace evidence', () => {
  assert.match(source, /Among the admissible induced relabelings/);
  assert.match(source, /Restricting them to \$\$\{notationLatex\('v_free'\)\}\$ produces \$\$\{notationLatex\('g_pointwise_restricted_v'\)\}\$/);
  assert.match(source, /Formal takeaway/);
  assert.match(source, /the part of \$G_\{\\\\text\{pt\}\}\$ that still acts on the computed output tensor/);
  assert.match(source, /R\[<span style=\{vStyle\}>0<\/span>,<span style=\{vStyle\}>1<\/span>\]/);
  assert.match(source, /R\[<span style=\{vStyle\}>1<\/span>,<span style=\{vStyle\}>0<\/span>\]/);
  assert.match(source, /= 3 \+ 4 \+ 6 \+ 8 = <strong>21<\/strong>/);
  assert.match(source, /R\[i,j\] = v_i\\\\,v_j/);
  assert.doesNotMatch(source, /R\[<span style=\{vStyle\}>0<\/span>,<span style=\{vStyle\}>0<\/span>\]/);
  assert.doesNotMatch(source, /R\[<span style=\{vStyle\}>1<\/span>,<span style=\{vStyle\}>1<\/span>\]/);
});

test('section 2 handoff paragraph spans the full appendix row width', () => {
  assert.doesNotMatch(
    source,
    /<p className="mt-5 max-w-\[72ch\] font-serif text-\[17px\] leading-\[1\.75\] text-gray-700">[\s\S]*This is the boundary used by the rest of the appendix/,
  );
  assert.match(source, /This is the boundary used by the rest of the appendix/);
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
  assert.match(source, /const APPENDIX_PROSE_CLASS = 'font-serif text-\[17px\] leading-\[1\.75\] text-gray-900';/);
  assert.match(source, /const APPENDIX_PROSE_JUSTIFIED_CLASS = `\$\{APPENDIX_PROSE_CLASS\} text-justify`;/);
  assert.match(source, /const APPENDIX_APP_TEXT_CLASS = 'text-\[13px\] leading-\[1\.55\] text-gray-700';/);
  assert.match(source, /const APPENDIX_SMALL_TEXT_CLASS = 'text-\[12px\] leading-5 text-gray-600';/);
  assert.match(source, /const APPENDIX_MONO_LEDGER_CLASS = 'font-mono text-\[13px\] leading-relaxed text-gray-900';/);
  assert.match(source, /const APPENDIX_KICKER_CLASS = 'text-\[10px\] font-semibold uppercase tracking-\[0\.16em\] text-gray-400';/);
  assert.match(source, /const APPENDIX_FOOTNOTE_CLASS = 'text-\[11px\] italic text-muted-foreground';/);
});

test('appendix section 4 is rewritten as a formal-only contrast chapter with a two-row dummy-swap example', () => {
  assert.match(source, /function AppendixWorkedExample\(/);
  assert.match(source, /function WorkedExampleEquation\(/);
  assert.match(source, /function WorkedExampleEquationLedger\(/);
  assert.match(source, /Section 3 isolated the visible-label action that remains pointwise on the output tensor\./);
  assert.match(source, /The complementary factor comes from the summed labels alone\./);
  assert.match(source, /Formal takeaway/);
  assert.match(source, /S\(W_\{\\mathrm\{summed\}\}\) is the full symmetric group on the summed labels/);
  assert.match(source, /preserve the full sum after aggregation but do not give pointwise equal summands/);
  assert.match(source, /dummy swap \(k l\) preserves the double sum but sends individual summands to different products/);
  assert.match(source, /\(k,l\) = \(\s*<span style=\{wStyle\}>0<\/span>,<span style=\{wStyle\}>1<\/span>\)\s*:/);
  assert.match(source, /\(k,l\) = \(\s*<span style=\{wStyle\}>1<\/span>,<span style=\{wStyle\}>0<\/span>\)\s*:/);
  assert.match(source, /= 1 · 4 = <strong>4<\/strong>/);
  assert.match(source, /= 2 · 3 = <strong>6<\/strong>/);
  assert.match(source, /formal symmetry does not imply pointwise equality/);
  assert.doesNotMatch(source, /\(continued\)/);
});
