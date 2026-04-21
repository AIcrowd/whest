import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx', import.meta.url),
  'utf8',
);

test('appendix modal shell keeps the editorial paper rail and uses the restored appendix masthead title', () => {
  assert.match(source, /relative w-full max-w-\[(1460px|var\(--content-max\))\] rounded-lg border border-gray-200 bg-white shadow-2xl/);
  assert.match(source, /appendixRailClass = 'mx-auto w-full max-w-\[(1460px|var\(--content-max\))\] px-6 md:px-8 lg:px-10'/);
  assert.match(source, /id="expr-modal-heading"[\s\S]*>\s*Expression-level symmetry and Symmetry-aware storage\s*<span style=\{\{ color: 'var\(--coral\)' \}\}>/);
  assert.match(source, /This appendix has two parts\./);
  assert.match(source, /expression-level[\s\S]*row-level boundary[\s\S]*assembled formal group/);
  assert.match(source, /Symmetry-aware storage/i);
  assert.doesNotMatch(source, /Expression-level symmetry and symmetry aware storage/);
});

test('appendix introduces two internal parts and only three numbered chapters', () => {
  assert.match(source, /function AppendixPartHeader\(/);
  assert.match(source, /part="Part I"[\s\S]*title="Expression-level symmetry"/);
  assert.match(source, /part="Part II"[\s\S]*title="Symmetry-aware storage"/);
  assert.match(source, /<AppendixSection[\s\S]*n=\{1\}[\s\S]*title="Where does the boundary appear\?"/);
  assert.match(source, /<AppendixSection[\s\S]*n=\{2\}[\s\S]*title="Pointwise symmetry versus formal symmetry"/);
  assert.match(source, /<AppendixSection[\s\S]*n=\{3\}[\s\S]*title="Assembling the formal group"/);
  assert.doesNotMatch(source, /<AppendixSection[\s\S]*n=\{4\}/);
  assert.doesNotMatch(source, /<AppendixSection[\s\S]*n=\{5\}/);
  assert.doesNotMatch(source, /<AppendixSection[\s\S]*n=\{6\}/);
  assert.doesNotMatch(source, /How the formal group is built/);
  assert.doesNotMatch(source, /Why Burnside on the formal group overcounts/);
});

test('chapter 1 keeps the row-level ledger and decision logic but drops the old heading stack', () => {
  assert.match(source, /n=\{1\}[\s\S]*The main page works with \$\$\{notationLatex\('g_pointwise'\)\}/);
  assert.match(source, /n=\{1\}[\s\S]*The distinction first becomes visible at the row level/);
  assert.match(source, /n=\{1\}[\s\S]*The ledger below records the outcome for four representative presets/);
  assert.match(source, /n=\{1\}[\s\S]*G_\{\\text\{wreath\}\}/);
  assert.match(source, /n=\{1\}[\s\S]*Frobenius\./);
  assert.match(source, /n=\{1\}[\s\S]*Triangle\./);
  assert.match(source, /n=\{1\}[\s\S]*Formal takeaway/);
  assert.match(source, /n=\{1\}[\s\S]*Record every admissible/);
  assert.match(source, /n=\{1\}[\s\S]*Treat admissible moves with/);
  assert.match(source, /n=\{1\}[\s\S]*Reject every non-admissible/);
  assert.match(source, /Chapter 2 places the visible-label action/);

  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Pointwise symmetry<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Formal symmetry<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Evidence<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Decision boundary<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Outcome summary<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Featured outcomes<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Frobenius consequence<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Decision rule<\/p>/);
  assert.doesNotMatch(source, /<p className=\{APPENDIX_KICKER_CLASS\}>Why this matters<\/p>/);
});

test('chapter 2 remains the contrast chapter with aligned takeaways and both worked examples', () => {
  assert.match(source, /n=\{2\}[\s\S]*Chapter 1 isolates the row moves that genuinely relabel the contraction/);
  assert.match(source, /Visible action on the output tensor\./);
  assert.match(source, /Formal symmetry on the summed labels\./);
  assert.match(source, /n=\{2\}[\s\S]*holds on output cells themselves/);
  assert.match(source, /n=\{2\}[\s\S]*R\[0,1\] = R\[1,0\]/);
  assert.match(source, /n=\{2\}[\s\S]*= 1 · 4 = <strong>4<\/strong>/);
  assert.match(source, /n=\{2\}[\s\S]*= 2 · 3 = <strong>6<\/strong>/);
  assert.match(source, /n=\{2\}[\s\S]*title="bilinear trace"/);
  assert.match(source, /formal symmetry does not imply pointwise equality/);
});

test('chapter 3 merges formal-group assembly with the Burnside overcount warning', () => {
  assert.match(source, /n=\{3\}[\s\S]*Putting them together yields the larger formal symmetry group/);
  assert.match(source, /n=\{3\}[\s\S]*direct product/);
  assert.match(source, /n=\{3\}[\s\S]*<VSubSwConstruction/);
  assert.match(source, /Applying Burnside to/);
  assert.match(source, /would yield/);
  assert.match(source, /too optimistic/);
  assert.match(source, /only preserves the post-summation expression/);
  assert.match(source, /That closes the expression-level story\. Part II turns to storage/);
});

test('chapter 3 no-overcount branch shows the selected einsum and page-wide preset suggestions', () => {
  assert.match(source, /export default function ExpressionLevelModal\(\{ isOpen, onClose, analysis, group, example = null, onSelectPreset = null \}\)/);
  assert.match(source, /const BURNSIDE_GAP_PRESET_IDS = \['bilinear-trace', 'young-s3', 'young-s4-v2w2'\];/);
  assert.match(source, /Selected einsum/);
  assert.match(source, /<FormulaHighlighted example=\{example\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /To see the impact, jump to one of these presets:/);
  assert.match(source, /onClick=\{\(\) => onSelectPreset\?\.\(suggestedPreset\.idx\)\}/);
});

test('storage-aware savings is now a separate unnumbered part-II item', () => {
  assert.match(source, /part="Part II"[\s\S]*title="Symmetry-aware storage"/);
  assert.match(source, /The governing group for storage is/);
  assert.match(source, /pointwise-symmetric under it/);
  assert.match(source, /separate optimization axis/);
  assert.match(source, /mirrored writes to output cells/);
  assert.match(source, /<section className="pt-8">/);
  assert.match(source, /<thead className="border-b border-gray-200">/);
  assert.match(source, /<th className="px-2 py-2 font-semibold text-right">Saving<\/th>/);
  assert.match(source, /The formal-only factor \$\$\{notationLatex\('s_w_summed'\)\}\$ contributes nothing at the storage level/);
  assert.match(source, /Scope/);
  assert.doesNotMatch(source, /Section 5 closed the accumulation-count story/);
  assert.doesNotMatch(source, /appeared in Section 4 as the \$V_\{\\mathrm\{free\}\}\$-factor/);
});

test('appendix hover surfaces and shared typography registers remain intact', () => {
  assert.match(source, /import \{ createPortal \} from 'react-dom';/);
  assert.match(source, /function AppendixEinsumHoverCell\(/);
  assert.match(source, /function AppendixPresetHoverLabel\(/);
  assert.match(source, /createPortal\(/);
  assert.match(source, /pointer-events-none fixed z-\[9999\]/);

  assert.match(source, /const APPENDIX_PROSE_CLASS = 'font-serif text-\[17px\] leading-\[1\.75\] text-gray-900';/);
  assert.match(source, /const APPENDIX_PROSE_JUSTIFIED_CLASS = `\$\{APPENDIX_PROSE_CLASS\} text-justify`;/);
  assert.match(source, /const APPENDIX_APP_TEXT_CLASS = 'text-\[13px\] leading-\[1\.55\] text-gray-700';/);
  assert.match(source, /const APPENDIX_SMALL_TEXT_CLASS = 'text-\[12px\] leading-5 text-gray-600';/);
  assert.match(source, /const APPENDIX_MONO_LEDGER_CLASS = 'font-mono text-\[13px\] leading-relaxed text-gray-900';/);
  assert.match(source, /const APPENDIX_KICKER_CLASS = 'text-\[10px\] font-semibold uppercase tracking-\[0\.16em\] text-gray-400';/);
  assert.match(source, /const APPENDIX_FOOTNOTE_CLASS = 'text-\[11px\] italic text-muted-foreground';/);
  assert.doesNotMatch(source, /style=\{vStyle\}/);
  assert.doesNotMatch(source, /style=\{wStyle\}/);
  assert.match(source, /style=\{vStyle\(\)\}/);
  assert.match(source, /style=\{wStyle\(\)\}/);
});
