import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx', import.meta.url),
  'utf8',
);
const preambleSource = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx', import.meta.url),
  'utf8',
);
const editorialCalloutSource = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/EditorialCallout.jsx', import.meta.url),
  'utf8',
);
const appendixSectionSource = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/AppendixSection.jsx', import.meta.url),
  'utf8',
);
const inlineMathTextSource = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/InlineMathText.jsx', import.meta.url),
  'utf8',
);

test('appendix modal shell keeps the editorial rail and the new cost-vs-expression masthead', () => {
  assert.match(source, /import renderProseBlocks from '\.\.\/content\/renderProseBlocks\.jsx'/);
  assert.match(source, /import appendixSection1 from '\.\.\/content\/appendix\/section1\.ts'/);
  assert.match(source, /import appendixSection2 from '\.\.\/content\/appendix\/section2\.ts'/);
  assert.match(source, /import appendixSection3 from '\.\.\/content\/appendix\/section3\.ts'/);
  assert.match(source, /import appendixSection4 from '\.\.\/content\/appendix\/section4\.ts'/);
  assert.match(source, /import appendixSection5 from '\.\.\/content\/appendix\/section5\.ts'/);
  assert.match(source, /import appendixSection6 from '\.\.\/content\/appendix\/section6\.ts'/);
  assert.match(source, /function normalizeAppendixDisplayText\(/);
  assert.match(source, /const APPENDIX_REQUIRED_SLOT_ERRORS = \{/);
  assert.match(source, /function invariantAppendixSlot\(/);
  assert.match(source, /function renderAppendixSlot\(/);
  assert.match(source, /throw new Error\(APPENDIX_REQUIRED_SLOT_ERRORS\[slotKey\] \?\? `Missing appendix slot: \$\{slotKey\}`\)/);
  assert.match(source, /throw new Error\(`Missing appendix slot block: \$\{slotKey\}\[\$\{index\}\]`\)/);
  assert.match(source, /renderProseBlocks\(normalizedBlocks/);
  // Theme color helpers (vStyle/wStyle) live in the shared workedExample
  // module after the Task-1 hoist; ExpressionLevelModal pulls them in via
  // that module rather than importing explorerTheme.js directly. The new
  // module is the source of truth for the colored coordinate styling.
  assert.match(source, /vStyle,\s*\n\s*wStyle,\s*\n\}\s*from\s*'\.\/workedExample\/index\.jsx'/);
  const workedExampleSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/workedExample/index.jsx', import.meta.url),
    'utf8',
  );
  assert.match(workedExampleSource, /import \{ explorerThemeColor, getActiveExplorerThemeId \} from '\.\.\/\.\.\/lib\/explorerTheme\.js';/);
  assert.match(source, /relative w-full max-w-\[(1460px|var\(--content-max\))\] rounded-lg border border-gray-200 bg-white shadow-2xl/);
  assert.match(source, /appendixRailClass = 'mx-auto w-full max-w-\[(1460px|var\(--content-max\))\] px-6 md:px-8 lg:px-10'/);
  // V3.1 letter-strip restructure: title now reflects the whole appendix,
  // not §5's narrower lesson. The italic deck below it (lines 786–795)
  // continues to explain the appendix's cost-vs-expression scope.
  assert.match(source, /id="expr-modal-heading"[\s\S]*>\s*Appendix to the symmetry-aware count\s*<span style=\{\{ color: 'var\(--coral\)' \}\}>/);
  assert.match(source, /<SectionReferenceLink href="#cost-savings" beforeNavigate=\{onClose\}>Section 5<\/SectionReferenceLink>/);
  assert.match(source, /on the main page computed a symmetry-aware accumulation count/);
  assert.match(source, /href="#cost-savings"/);
  assert.match(source, /product-side symmetry for representative products, output-side action for stored output representatives, and formal dummy symmetry after summation/);
  assert.match(source, /max-w-\[min\(100%,980px\)\]/);
  assert.doesNotMatch(source, /Act 5 computed a symmetry-aware accumulation count\./);
  assert.doesNotMatch(source, /This appendix has two parts\./);
  assert.doesNotMatch(source, /Expression-level symmetry and Symmetry-aware storage/);
  assert.doesNotMatch(source, /if \(!block\) return null;/);
});

test('appendix uses an editorial spine with asymmetric support shelves', () => {
  // V3.1 letter-strip restructure: sub-section eyebrows use `n` as the
  // sub-position WITHIN their AppendixGroup (so n=1 appears in A.1, B, C,
  // D.1, and E). The unique-per-section anchor is `appendix-section-{N}`.
  // We slice by anchorId, walking back to the section's `<AppendixSection`
  // open and forward to the next mount.
  const sectionBlock = (n) => {
    const anchor = source.indexOf(`anchorId="appendix-section-${n}"`);
    assert.notEqual(anchor, -1, `missing section marker anchorId="appendix-section-${n}"`);
    const start = source.lastIndexOf('<AppendixSection', anchor);
    assert.notEqual(start, -1, `<AppendixSection open not found before anchor ${n}`);
    const next = source.indexOf('<AppendixSection', start + '<AppendixSection'.length);
    return source.slice(start, next === -1 ? undefined : next);
  };

  const section1 = sectionBlock(1);
  const section2 = sectionBlock(2);
  const section3 = sectionBlock(3);
  const section4 = sectionBlock(4);
  const section5 = sectionBlock(5);
  const section6 = sectionBlock(6);

  assert.match(source, /function AppendixSupportSplit\(/);
  assert.match(source, /grid-cols-\[0\.95fr_1\.25fr\]/);
  assert.match(source, /strict \? 'grid gap-y-6 gap-x-10 lg:grid-cols-2 lg:items-start'/);

  assert.match(section1, /AppendixSupportSplit/);
  assert.match(section1, /className="lg:grid-cols-\[0\.95fr_1\.25fr\]"/);
  assert.match(section1, /appendixSection1\.slots\.auditIntro/);

  assert.match(section2, /AppendixSupportSplit/);
  assert.match(section2, /className="lg:grid-cols-\[0\.95fr_1\.25fr\]"/);
  assert.match(section2, /appendixSection2\.slots\.runningExampleLabelPrefix/);
  assert.match(section2, /appendixSection2\.slots\.runningExamplePresetLabel/);
  assert.match(section2, /<FormulaHighlighted example=\{runningExamplePreset\} hoveredLabels=\{null\} \/>/);
  assert.doesNotMatch(section2, /inline-flex max-w-full rounded-xl border border-stone-200 bg-white px-3 py-2\.5 shadow-sm/);

  assert.match(section3, /AppendixSupportSplit/);
  assert.match(section3, /Worked example — bilinear trace/);

  assert.match(source, /SECTION4_FORMAL_GROUP_PRESET_IDS = \['frobenius', 'direct-s2-c3', 'triple-outer'\]/);
  assert.match(source, /SECTION4_FORMAL_GROUP_PRESETS = SECTION4_FORMAL_GROUP_PRESET_IDS/);
  assert.match(section4, /AppendixSupportSplit/);
  assert.match(section4, /renderAppendixSingleBlock\(appendixSection4\.slots\.constructionTitle, 0\)/);
  assert.match(section4, /renderAppendixSingleBlock\(appendixSection4\.slots\.presetPickerLabel, 0\)/);
  assert.match(section4, /SECTION4_FORMAL_GROUP_PRESETS\.map\(\(suggestedPreset\) =>/);
  assert.match(section4, /onClick=\{\(\) => onSelectPreset\?\.\(suggestedPreset\.idx\)\}/);
  assert.match(section4, /<VSubSwConstruction[\s\S]*showHeading=\{false\}/);
  assert.match(section4, /VSubSwConstruction/);

  assert.match(section5, /AppendixSupportSplit/);
  assert.match(section5, /supportClassName="space-y-5 xl:pt-1"/);
  assert.doesNotMatch(section5, /supportClassName=\{APPENDIX_SUPPORT_SHELF_CLASS\}/);

  // Section 6 (styling pass): renders intro + footer slots inside the shared
  // `AppendixSupportSplit` lane + shelf grid (matching Sections 1–5). The
  // partition-counting equation appears as KaTeX display math inside an
  // The right-hand shelf now renders the equation and glossary entries
  // directly in a centered container — no AppendixDefinitionPanel box.
  // Legacy savings-comparison table, scopeLabel, tableNote, footnote slots,
  // and the negative-margin footer band are gone.
  assert.match(section6, /AppendixSupportSplit/);
  assert.doesNotMatch(section6, /AppendixDefinitionPanel/);
  assert.match(section6, /<Latex\s+math=\{[^}]+\}\s+display[^/]*\/>/);
  assert.match(section6, /themeOverride=\{APPENDIX_SECTION_SIX_THEME_OVERRIDE\}/);
  assert.match(section6, /deckClassName="max-w-none"/);
  assert.match(section6, /appendixSection6\.slots\.intro/);
  assert.match(section6, /appendixSection6\.slots\.footer/);
  assert.doesNotMatch(section6, /Storage-only saving/);
  assert.doesNotMatch(section6, /appendixSection6\.slots\.scopeLabel/);
  assert.doesNotMatch(section6, /appendixSection6\.slots\.tableNote/);
  assert.match(section6, /-mx-6 -mb-10 mt-8 border-t border-stone-200\/70 bg-gray-50 px-6 py-4 md:-mx-8 md:px-8 lg:-mx-10 lg:px-10/);
  assert.doesNotMatch(section6, /bg-stone-50\/60/);
  assert.doesNotMatch(section6, /isModelBlock/);
  assert(section6.indexOf('appendixSection6.slots.intro') < section6.indexOf('appendixSection6.slots.footer'), 'intro should render before footer');

  assert.match(appendixSectionSource, /anchorId = ''/);
  assert.match(appendixSectionSource, /deckClassName = ''/);
  assert.match(appendixSectionSource, /import InlineMathText from '\.\/InlineMathText\.jsx';/);
  assert.match(appendixSectionSource, /const renderedTitle = typeof title === 'string' \? <InlineMathText>\{title\}<\/InlineMathText> : title;/);
  assert.match(appendixSectionSource, /<section id=\{anchorId \|\| undefined\}/);
  assert.match(appendixSectionSource, /<SectionEyebrow n=\{n\} label=\{label\} anchorId=\{anchorId\} \/>/);
  assert.match(appendixSectionSource, /<h3 className="mt-2 font-heading text-\[24px\] font-semibold leading-tight text-gray-900">\s*\{renderedTitle\}\s*<\/h3>/);
  assert.match(appendixSectionSource, /className=\{\['mt-3 font-serif text-\[17px\] leading-\[1\.75\] text-gray-700', deckClassName \|\| 'max-w-\[70ch\]'\]\.filter\(Boolean\)\.join\(' '\)\}/);
  assert.match(source, /anchorId="appendix-section-1"/);
  assert.match(source, /anchorId="appendix-section-2"/);
  assert.match(source, /anchorId="appendix-section-3"/);
  assert.match(source, /anchorId="appendix-section-4"/);
  assert.match(source, /anchorId="appendix-section-5"/);
  assert.match(source, /anchorId="appendix-section-6"/);
});

test('section 1 explains G_pt first and moves the σ-loop ledger into an audit block', () => {
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*title=\{appendixSection1\.title\}/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*deck=\{appendixSection1\.deck\}/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*appendixSection1\.slots\.intro/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*appendixSection1\.slots\.takeaway/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.auditIntro, 0\)/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.auditIntro, 1\)/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*appendixSection1\.slots\.columnGuide/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 0\)/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 1\)/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 2\)/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 3\)/);
  assert.match(source, /data-takeaway=\{String\.raw`G_\{\\mathrm\{pt\}\} is a cost group: it is valid for direct product\/update compression because it identifies equal indexed products under the declared equality model\.`\}/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*className=\{APPENDIX_PROSE_JUSTIFIED_CLASS\}/);
  assert.match(source, /appendixSection1\.slots\.auditIntro/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*appendixSection1\.slots\.columnGuide/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*<Latex math="\|G_\{\\mathrm\{wreath\}\}\|" \/>/);
  assert.match(source, /anchorId="appendix-section-1"[\s\S]*<Latex math="\|G_\{\\text\{pt\}\}\|" \/>/);
  assert.doesNotMatch(source, /`\$\|G_\{\\mathrm\{wreath\}\}\|\$ counts candidate row moves before filtering\.[\s\S]*\$\|G_\{\\text\{pt\}\}\|\$ is the final detected pointwise group size\.`/);
  assert.match(source, /appendixSection1\.slots\.followups/);
  assert.match(source, /trace-product/);
  assert.match(source, /triangle/);
  assert.match(source, /young-s3/);
  assert.doesNotMatch(source, /anchorId="appendix-section-1"[\s\S]*Let \$\$\{notationLatex\('l_labels'\)\}\$ be the set of all labels/);
  assert.doesNotMatch(source, /anchorId="appendix-section-1"[\s\S]*When such a bijection exists, it induces a label relabeling/);
  assert.doesNotMatch(source, /<AppendixAuditBlock/);
  assert.doesNotMatch(source, /Audit detail/);
  assert.doesNotMatch(source, /Takeaway\.\s*<\/span>\s*<Latex math=\{String\.raw`G_\\\{\\text\\\{pt\\\}\\\}`/);
});

test('sections 2 through 4 introduce same-domain dummy renaming, then G_out, then G_f in dependency order', () => {
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*title=\{appendixSection2\.title\}/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection2\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*appendixSection2\.slots\.intro/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*appendixSection2\.slots\.takeaway/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*appendixSection2\.slots\.runningExampleLabelPrefix/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*appendixSection2\.slots\.runningExamplePresetLabel/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*appendixSection2\.slots\.runningExampleLead/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*<FormulaHighlighted example=\{runningExamplePreset\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /const runningExampleExpressionParts = useMemo\(/);
  assert.match(source, /const runningExampleFactors = useMemo\(/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*outputCoords=\{runningExampleExpressionParts\.outputLabels\}/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*sumCoords=\{runningExampleExpressionParts\.summedLabels\}/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*factors=\{runningExampleFactors\}/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*coords: \['i', 'l'\]/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*coords: \['j', 'k'\]/);
  assert.doesNotMatch(source, /anchorId="appendix-section-2"[\s\S]*This gives a second group, \$\$\{notationLatex\('s_w_summed'\)\}\$/);
  assert.doesNotMatch(source, /anchorId="appendix-section-2"[\s\S]*Rename dummy variables back/);
  assert.doesNotMatch(source, /anchorId="appendix-section-2"[\s\S]*The swap [\s\S]* preserves the double sum as a formal expression/);
  assert.match(source, /anchorId="appendix-section-2"[\s\S]*className=\{APPENDIX_PROSE_JUSTIFIED_CLASS\}/);
  assert.doesNotMatch(source, /anchorId="appendix-section-2"[\s\S]*rounded-lg border border-gray-200 bg-white px-5 py-4 shadow-sm/);
  assert.doesNotMatch(source, /anchorId="appendix-section-2"[\s\S]*4 \\neq 6/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*title=\{appendixSection3\.title\}/);
  // Section 3's deck contains LaTeX (`$H = \mathrm{Stab}_{G_{\text{pt}}}(V)|_V$`)
  // so the modal wraps it in <InlineMathText> + normalizeAppendixDisplayText
  // — same pattern sections 2 / 5 / 6 use for their LaTeX-bearing decks.
  // Without this wrap, the dollar-delimited math renders as raw `$...$`
  // text in the deck under the A.2 sub-section heading.
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection3\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*appendixSection3\.slots\.definitionLead/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*appendixSection3\.slots\.takeaway/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*appendixSection3\.slots\.workedExampleLead/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*appendixSection3\.slots\.workedExampleNote/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*data-reader-facing-formula=/);
  assert.match(source, /const APPENDIX_G_OUT_DEFINITION_LATEX/);
  assert.match(source, /APPENDIX_PI_RESTRICT_V_LATEX/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*appendixSection3\.slots\.workedExampleLead/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*<FormulaHighlighted example=\{EXAMPLES_BY_ID\.get\('bilinear-trace'\)\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /anchorId="appendix-section-3"[\s\S]*\\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.doesNotMatch(source, /anchorId="appendix-section-3"[\s\S]*Among the elements of \$\$\{notationLatex\('g_pointwise'\)\}\$/);
  assert.doesNotMatch(source, /anchorId="appendix-section-3"[\s\S]*supportClassName=\{`\$\{APPENDIX_SUPPORT_SHELF_CLASS\} xl:pt-5`\}/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*title=\{appendixSection4\.title\}/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*deck=\{appendixSection4\.deck\}/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*appendixSection4\.slots\.intro/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*appendixSection4\.slots\.takeaway/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*renderAppendixSingleBlock\(appendixSection4\.slots\.constructionTitle, 0\)/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*appendixSection4\.slots\.presetPickerLabel/);
  assert.match(source, /SECTION4_FORMAL_GROUP_PRESET_IDS/);
  assert.match(source, /frobenius/);
  assert.match(source, /direct-s2-c3/);
  assert.match(source, /triple-outer/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*appendixSection4\.slots\.constructionNote/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*<Latex math=\{String\.raw`G_\{\\text\{f\}\} = H \\times \\prod_d S\(W_d\)`\} \/>/);
  assert.match(source, /anchorId="appendix-section-4"[\s\S]*<VSubSwConstruction/);
  assert.doesNotMatch(source, /anchorId="appendix-section-4"[\s\S]*We can now name the label-renaming formal group considered in this appendix\./);
  assert.doesNotMatch(source, /anchorId="appendix-section-4"[\s\S]*rounded-xl border border-gray-200 bg-gray-50 px-5 py-4 text-center/);
  assert.doesNotMatch(source, /anchorId="appendix-section-4"[\s\S]*supportClassName=\{`\$\{APPENDIX_SUPPORT_SHELF_CLASS\} xl:pt-1`\}/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('s_w_summed'\)\}\$/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('g_output'\)\}\$/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('g_formal'\)\}\$/);
});

test('section 5 uses alphaComparison branches and the bilinear witness to reject Burnside on G_f for cost', () => {
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*title=\{appendixSection5\.title\}/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection5\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.intro/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*renderAppendixSingleBlock\(appendixSection5\.slots\.rule, 0\)/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.presetPickerLabel/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.mismatchLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.coincidentLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.noneLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.workedExampleLabelPrefix/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.workedExampleLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.assignmentLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.bilinearOrbitLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.directOrbitLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.mixedOrbitLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.genericOrbitLead/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.genericAssignmentTemplate/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*appendixSection5\.slots\.genericNoteTemplate/);
  // Worked-example primitives (and the vStyle/wStyle theme helpers they
  // build on) were hoisted into a sibling shared module in Task 1 of the
  // orbit-rep-matrix redesign. ExpressionLevelModal now imports them
  // rather than redefining; the bodies live (verbatim) in
  // components/workedExample/index.jsx. Pin the imports here, and check
  // the function bodies in the shared module instead.
  {
    const workedExampleSource = fs.readFileSync(
      new URL('./components/symmetry-aware-einsum-contractions/components/workedExample/index.jsx', import.meta.url),
      'utf8',
    );
    assert.match(workedExampleSource, /export function WorkedExampleIndex\(/);
    assert.match(workedExampleSource, /export function WorkedExampleCoords\(/);
    assert.match(workedExampleSource, /export function WorkedExampleTensorRef\(/);
    assert.match(workedExampleSource, /export function WorkedExampleTensorProduct\(/);
    assert.match(workedExampleSource, /export function WorkedExampleDisplayEquation\(/);
    assert.match(workedExampleSource, /export function vStyle\(\) \{\s*return \{\s*color: explorerThemeColor\(getActiveExplorerThemeId\(\), 'hero'\),\s*fontWeight: 600,\s*\};\s*\}/);
    assert.match(workedExampleSource, /export function wStyle\(\) \{\s*return \{\s*color: explorerThemeColor\(getActiveExplorerThemeId\(\), 'summedSide'\),\s*fontWeight: 600,\s*\};\s*\}/);
    assert.match(source, /WorkedExampleTensorRef,/);
    assert.match(source, /WorkedExampleCoords,/);
    assert.match(source, /WorkedExampleTensorProduct,/);
    assert.match(source, /WorkedExampleDisplayEquation,/);
  }
  assert.match(source, /function buildWorkedExampleFactors\(/);
  assert.match(source, /alphaComparison\.state === 'mismatch'/);
  assert.match(source, /alphaComparison\.state === 'coincident'/);
  assert.match(source, /alphaComparison\.state === 'none'/);
  assert.doesNotMatch(source, /anchorId="appendix-section-5"[\s\S]*The main page’s \$\\alpha\$ counts accumulation representatives/);
  assert.doesNotMatch(source, /anchorId="appendix-section-5"[\s\S]*A naive formal count using \$\$\{notationLatex\('g_formal'\)\}\$ gives \$\\alpha_\{\\text\{formal\}\}/);
  assert.doesNotMatch(source, /anchorId="appendix-section-5"[\s\S]*The pointwise accumulation count used by the engine is \$\\alpha_\{\\text\{engine\}\}/);
  assert.match(source, /\\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.doesNotMatch(source, /A = \[\[1, 2\], \[3, 4\]\]/);
  assert.match(source, /anchorId="appendix-section-5"[\s\S]*<WorkedExampleDisplayEquation/);
  assert.match(source, /appendixSection5\.slots\.assignmentLead/);
  assert.match(source, /<Latex math=\{String\.raw`i = 0,\\; j = 1`\} \/>/);
  assert.match(source, /outputCoords=\{\[0, 1\]\}/);
  assert.match(source, /sumCoords=\{\['k', 'l'\]\}/);
  assert.match(source, /appendixSection5\.slots\.bilinearOrbitLead/);
  assert.match(source, /scalarValues=\{\[1, 4\]\}/);
  assert.match(source, /scalarValues=\{\[2, 3\]\}/);
  assert.match(source, /const showDirectS2C3FormalOrbitExample =/);
  assert.match(source, /const showMixedChainFormalOrbitExample =/);
  assert.match(source, /example\?\.id === 'direct-s2-c3'/);
  assert.match(source, /example\?\.id === 'mixed-chain'/);
  assert.match(source, /appendixSection5\.slots\.directIntro/);
  assert.match(source, /sumCoords=\{\['c', 'd', 'e'\]\}/);
  assert.match(source, /appendixSection5\.slots\.directOrbitLead/);
  assert.match(source, /A = \\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.match(source, /B = \\begin\{bmatrix\} 1 & 2 \\\\ 4 & 5 \\end\{bmatrix\}/);
  assert.match(source, /sumCoords=\{\['j', 'k'\]\}/);
  assert.match(source, /scalarValues=\{\[1, 2, 3\]\}/);
  assert.match(source, /scalarValues=\{\[2, 4, 1\]\}/);
  assert.match(source, /const formalOrbitExample = useMemo\(/);
  assert.match(source, /const exampleExpressionParts = useMemo\(/);
  assert.match(source, /buildFormalOrbitExampleData\(\{ example, labelOrder: analysis\?\.symmetry\?\.allLabels \?\? \[\], witness: alphaComparison\.witness \}\)/);
  assert.doesNotMatch(source, /getExampleExpressionParts\(example\) \{[\s\S]*subscripts\.includes\('->'\)/);
  assert.match(source, /formalOrbitExample\?\.outputAssignmentLatex/);
  assert.match(source, /formalOrbitExample\??\.outputValues/);
  assert.match(source, /formalOrbitExample\??\.summedValuesA/);
  assert.match(source, /formalOrbitExample\??\.summedValuesB/);
  assert.match(source, /formalOrbitExample\?\.summedAssignmentA/);
  assert.match(source, /formalOrbitExample\?\.summedAssignmentB/);
  assert.match(source, /outputTargetNoun: outputLabels\.length \? 'output entry' : 'scalar output'/);
  assert.match(source, /appendixSection5\.slots\.genericNoteTemplate/);
  assert.doesNotMatch(source, /Rule\./);
  assert.match(source, /appendixSection5\.slots\.rule/);
  assert.match(source, /font-mono text-\[13px\] leading-6 text-stone-900/);
  assert.match(source, /<FormulaHighlighted example=\{example\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /selectedOperandItems = useMemo/);
  assert.match(source, /selectedOperandItems\.map/);
  assert.match(source, /<SymmetryChip key=\{`\$\{operand\.name\}-\$\{operand\.sym\}`\} name=\{operand\.chipName\} symmetry=\{operand\.sym\} \/>/);
  assert.doesNotMatch(source, /Selected einsum[\s\S]*rounded-xl border border-stone-200 bg-white px-3 py-2\.5 shadow-sm/);
  assert.doesNotMatch(source, /alphaComparison\.state === 'coincident' \|\| alphaComparison\.state === 'none'[\s\S]*<EditorialCallout/);
  assert.match(source, /appendixSection5\.slots\.presetPickerLabel/);
  assert.doesNotMatch(source, /alphaComparison\.state === 'coincident' \|\| alphaComparison\.state === 'none'/);
  assert.doesNotMatch(source, /formatWitnessTuple\(alphaComparison\.witness\.tupleA\)/);
  assert.doesNotMatch(source, /formatWitnessTuple\(alphaComparison\.witness\.tupleB\)/);
  assert.doesNotMatch(source, /The following two assignments lie in the same formal orbit and contribute to the same output entry, but they belong to different pointwise orbits and produce different products\./);
  const section5Start = source.indexOf('anchorId="appendix-section-5"');
  const presetsIdx = source.indexOf('appendixSection5.slots.presetPickerLabel', section5Start);
  const mismatchIdx = source.indexOf("alphaComparison.state === 'mismatch'", section5Start);
  assert.equal(source.includes('appendixSection5.slots.selectedEinsumLabel'), false);
  assert.doesNotMatch(source, /Selected einsum:/);
  const selectedFormulaIdx = source.indexOf('<FormulaHighlighted example={example} hoveredLabels={null} />', section5Start);
  assert(selectedFormulaIdx < presetsIdx, 'preset jump row should follow the selected einsum line');
  assert(presetsIdx < mismatchIdx, 'preset jump row should appear before the branch-specific explanation block');
});

test('section 6 explains the unified output-orbit accumulation count', () => {
  // Task-9 rewrite: section 6 dropped the storage-savings table, the
  // alpha_engine vs alpha_storage comparison, the Model 1/2/3 framing, and
  // the buildStorageSavingsRows/savingsTableRows machinery. It now renders
  // the intro slot followed by the footer band — both describe how H is
  // induced from G_pt and why the legacy G_out / storage-only model is
  // subsumed by the unified metric.
  assert.match(source, /anchorId="appendix-section-6"[\s\S]*title=\{appendixSection6\.title\}/);
  assert.match(source, /anchorId="appendix-section-6"[\s\S]*deckClassName="max-w-none"/);
  assert.match(source, /anchorId="appendix-section-6"[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection6\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /anchorId="appendix-section-6"[\s\S]*appendixSection6\.slots\.intro/);
  assert.match(source, /anchorId="appendix-section-6"[\s\S]*appendixSection6\.slots\.footer/);

  // Removed in Task 9.
  assert.doesNotMatch(source, /import \{ buildStorageSavingsRows \}/);
  assert.doesNotMatch(source, /buildStorageSavingsRows\(EXAMPLES, 3\)/);
  assert.doesNotMatch(source, /const savingsTableRows = useMemo/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*appendixSection6\.slots\.footnote/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*appendixSection6\.slots\.tableNote/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*appendixSection6\.slots\.scopeLabel/);
  assert.doesNotMatch(source, /Accumulation representatives/);
  assert.doesNotMatch(source, /Output-orbit representatives/);
  assert.doesNotMatch(source, /Storage-only saving/);
  assert.doesNotMatch(source, /\\alpha_\{\\text\{engine\}\}/);
  assert.doesNotMatch(source, /\\alpha_\{\\text\{storage\}\}/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*\{r\.alphaEngine\}/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*\{r\.alphaStorage\}/);
});

test('appendix avoids raw unthemed math literals for key semantic symbols and example algebra', () => {
  assert.doesNotMatch(source, /<Latex math=\{'S\(W\)'\} \/>/);
  assert.doesNotMatch(source, /String\.raw`\\\$G_\{\\text\{f\}\} = G_\{\\mathrm\{out\}\} \\times S\(W\)\\\$`/);
  assert.doesNotMatch(source, /Rule\. Use \$G_\{\\\\text\{pt\}\}\$ for accumulation\./);
  assert.match(source, /notationColoredLatex\('s_w_summed', 'S\(W\)'\)/);
  assert.match(source, /WorkedExampleTensorProduct/);
  assert.match(source, /WorkedExampleDisplayEquation/);
});

test('section 6 model prefixes are emphasized through copy plus InlineMathText strong overrides', () => {
  assert.match(inlineMathTextSource, /export function renderTooltipInlineText\(text, keyPrefix, options = \{\}\)/);
  assert.match(inlineMathTextSource, /const strongClassName = options\.strongClassName \?\? 'font-semibold text-current';/);
  assert.match(inlineMathTextSource, /className=\{strongClassName\}/);
});

test('appendix hover surfaces and shared typography registers remain intact', () => {
  assert.match(source, /import \{ createPortal \} from 'react-dom';/);
  assert.match(source, /function AppendixEinsumHoverCell\(/);
  assert.match(source, /function AppendixPresetHoverLabel\(/);
  assert.match(source, /createPortal\(/);
  assert.match(source, /pointer-events-none fixed z-\[9999\]/);
  // APPENDIX_PROSE_CLASS and APPENDIX_MONO_LEDGER_CLASS were hoisted into
  // the shared workedExample module in Task 1 of the orbit-rep-matrix
  // redesign so OrbitDetailCard and similar surfaces can reuse them. Verify the
  // canonical declarations there, and confirm ExpressionLevelModal pulls
  // them in via the import. The remaining APPENDIX_* class constants
  // (justified prose, article lane, app text, etc.) still live alongside
  // the modal's local helpers.
  {
    const workedExampleSource = fs.readFileSync(
      new URL('./components/symmetry-aware-einsum-contractions/components/workedExample/index.jsx', import.meta.url),
      'utf8',
    );
    assert.match(workedExampleSource, /export const APPENDIX_PROSE_CLASS = 'font-serif text-\[17px\] leading-\[1\.75\] text-gray-900';/);
    assert.match(workedExampleSource, /export const APPENDIX_MONO_LEDGER_CLASS = 'font-mono text-\[13px\] leading-relaxed text-gray-900';/);
    assert.match(source, /APPENDIX_PROSE_CLASS,/);
    assert.match(source, /APPENDIX_MONO_LEDGER_CLASS,/);
  }
  assert.match(source, /const APPENDIX_PROSE_JUSTIFIED_CLASS = `\$\{APPENDIX_PROSE_CLASS\} text-justify`;/);
  assert.match(source, /const APPENDIX_ARTICLE_LANE_CLASS = 'max-w-\[78ch\] space-y-4 \[\&_p\]:text-justify';/);
  assert.match(source, /const APPENDIX_APP_TEXT_CLASS = 'text-\[13px\] leading-\[1\.55\] text-gray-700';/);
  assert.match(source, /const APPENDIX_SMALL_TEXT_CLASS = 'text-\[12px\] leading-5 text-gray-600';/);
  assert.match(source, /const APPENDIX_KICKER_CLASS = 'text-\[10px\] font-semibold uppercase tracking-\[0\.16em\] text-gray-400';/);
  assert.match(source, /const APPENDIX_FOOTNOTE_CLASS = 'text-\[11px\] italic text-muted-foreground';/);
});

test('appendix callout cards reuse the main-page "Where symmetry enters" shell', () => {
  assert.match(editorialCalloutSource, /rounded-2xl border border-primary\/20 bg-accent\/40 px-5 py-5/);
  assert.match(editorialCalloutSource, /font-sans text-\[11px\] font-semibold uppercase tracking-\[0\.16em\] text-coral/);
  assert.match(source, /import EditorialCallout from '\.\/EditorialCallout\.jsx';/);
  assert.match(preambleSource, /import EditorialCallout from '\.\/EditorialCallout\.jsx';/);
  assert.match(source, /function AppendixTakeaway\(/);
  assert.match(source, /function AppendixDefinitionPanel\(/);
  assert.doesNotMatch(source, /anchorId="appendix-section-5"[\s\S]*<EditorialCallout[\s\S]*label="Rule"/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*<EditorialCallout[\s\S]*label="Scope"/);
  assert.doesNotMatch(source, /alphaComparison\.state === 'coincident' \|\| alphaComparison\.state === 'none'[\s\S]*<EditorialCallout[\s\S]*label="Try presets with a visible gap"/);
  assert.doesNotMatch(source, /label=\{label\}/);
  assert.doesNotMatch(source, /label=\{title\}/);
  assert.doesNotMatch(source, /function AppendixTakeaway[\s\S]*rounded-lg border border-gray-200 bg-gray-50 px-5 py-4/);
  assert.doesNotMatch(source, /function AppendixDefinitionPanel[\s\S]*rounded-xl border border-gray-200 bg-white px-5 py-4 shadow-sm/);
  assert.doesNotMatch(source, /anchorId="appendix-section-5"[\s\S]*rounded-lg border border-gray-900\/10 bg-stone-50 px-5 py-4/);
  assert.doesNotMatch(source, /anchorId="appendix-section-6"[\s\S]*rounded-lg border border-gray-200 bg-gray-50 px-5 py-4/);
  assert.doesNotMatch(source, /function AppendixCalloutShell\(/);
  assert.doesNotMatch(source, /APPENDIX_CALLOUT_SHELL_CLASS/);
  assert.doesNotMatch(source, /APPENDIX_CALLOUT_KICKER_CLASS/);
  assert.doesNotMatch(source, /rounded-2xl border border-primary\/20 bg-accent\/40 px-5 py-5/);
  assert.match(preambleSource, /<EditorialCallout[\s\S]*Where symmetry enters/);
  assert.doesNotMatch(preambleSource, /<div id="where-symmetry-enters" className="rounded-2xl border border-primary\/20 bg-accent\/40 px-5 py-5 scroll-mt-24">/);
});

test('appendix roadmap cards are formula-first with white background and black borders', () => {
  const roadmapBlock = source.slice(
    source.indexOf('function AppendixRoadmap()'),
    source.indexOf('const appendixRailClass'),
  );
  assert.match(source, /function AppendixRoadmap\(/);
  assert.match(source, /rounded-lg border border-black bg-white px-4 py-3/);
  assert.doesNotMatch(roadmapBlock, /Pointwise group/);
  assert.doesNotMatch(roadmapBlock, /Output group/);
  assert.doesNotMatch(roadmapBlock, /Dummy group/);
  assert.doesNotMatch(roadmapBlock, /Formal group/);
  assert.ok(roadmapBlock.includes('H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V'));
  assert.match(roadmapBlock, /the main accumulation count: stored output representatives in/);
  assert.match(source, /same-domain dummy renamings of bound summation variables/);
  assert.match(source, /<Latex math=\{String\.raw`G_\{\\text\{f\}\} = H \\times \\prod_d S\(W_d\)`\} \/>/);
});
