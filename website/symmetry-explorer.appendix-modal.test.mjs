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
  assert.match(source, /relative w-full max-w-\[(1460px|var\(--content-max\))\] rounded-lg border border-gray-200 bg-white shadow-2xl/);
  assert.match(source, /appendixRailClass = 'mx-auto w-full max-w-\[(1460px|var\(--content-max\))\] px-6 md:px-8 lg:px-10'/);
  assert.match(source, /id="expr-modal-heading"[\s\S]*>\s*Why expression symmetry is not the cost symmetry\s*<span style=\{\{ color: 'var\(--coral\)' \}\}>/);
  assert.match(source, /<SectionReferenceLink href="#cost-savings" beforeNavigate=\{onClose\}>Section 5<\/SectionReferenceLink>/);
  assert.match(source, /on the main page computed a symmetry-aware accumulation count/);
  assert.match(source, /href="#cost-savings"/);
  assert.match(source, /pointwise symmetry for accumulation, formal symmetry for the completed expression, and output symmetry for storage/);
  assert.match(source, /max-w-\[min\(100%,980px\)\]/);
  assert.doesNotMatch(source, /Act 5 computed a symmetry-aware accumulation count\./);
  assert.doesNotMatch(source, /This appendix has two parts\./);
  assert.doesNotMatch(source, /Expression-level symmetry and Symmetry-aware storage/);
  assert.doesNotMatch(source, /if \(!block\) return null;/);
});

test('appendix uses an editorial spine with asymmetric support shelves', () => {
  const sectionBlock = (n) => {
    const start = source.indexOf(`n={${n}}`);
    assert.notEqual(start, -1, `missing section marker n={${n}}`);
    const next = source.indexOf('<AppendixSection', start + 1);
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

  assert.match(section4, /AppendixSupportSplit/);
  assert.match(section4, /renderAppendixSingleBlock\(appendixSection4\.slots\.constructionTitle, 0\)/);
  assert.match(section4, /<VSubSwConstruction[\s\S]*showHeading=\{false\}/);
  assert.match(section4, /VSubSwConstruction/);

  assert.match(section5, /AppendixSupportSplit/);
  assert.match(section5, /supportClassName="space-y-5 xl:pt-1"/);
  assert.doesNotMatch(section5, /supportClassName=\{APPENDIX_SUPPORT_SHELF_CLASS\}/);

  assert.doesNotMatch(section6, /AppendixSupportSplit/);
  assert.match(section6, /Storage-only saving/);
  assert.match(section6, /deckClassName="max-w-none"/);
  assert.match(section6, /-mx-6 -mb-10 mt-8 border-t border-stone-200\/70 bg-gray-50 px-6 py-4 md:-mx-8 md:px-8 lg:-mx-10 lg:px-10/);
  assert.match(section6, /appendixSection6\.slots\.scopeLabel/);
  assert.match(section6, /text-\[12\.5px\] leading-6 text-stone-700/);
  assert.doesNotMatch(section6, /max-w-\[78ch\]/);
  assert(section6.indexOf('Storage-only saving') < section6.indexOf('-mx-6 -mb-10 mt-8 border-t border-stone-200/70 bg-gray-50 px-6 py-4 md:-mx-8 md:px-8 lg:-mx-10 lg:px-10'), 'footer band should come after the table');
  assert(section6.indexOf('appendixSection6.slots.tableNote') > section6.indexOf('Storage-only saving'), 'storage explanatory paragraph should move below the table');

  assert.match(appendixSectionSource, /anchorId = ''/);
  assert.match(appendixSectionSource, /deckClassName = ''/);
  assert.match(appendixSectionSource, /<section id=\{anchorId \|\| undefined\}/);
  assert.match(appendixSectionSource, /<SectionEyebrow n=\{n\} label=\{label\} anchorId=\{anchorId\} \/>/);
  assert.match(appendixSectionSource, /className=\{\['mt-3 font-serif text-\[17px\] leading-\[1\.75\] text-gray-700', deckClassName \|\| 'max-w-\[70ch\]'\]\.filter\(Boolean\)\.join\(' '\)\}/);
  assert.match(source, /anchorId="appendix-section-1"/);
  assert.match(source, /anchorId="appendix-section-2"/);
  assert.match(source, /anchorId="appendix-section-3"/);
  assert.match(source, /anchorId="appendix-section-4"/);
  assert.match(source, /anchorId="appendix-section-5"/);
  assert.match(source, /anchorId="appendix-section-6"/);
});

test('section 1 explains G_pt first and moves the σ-loop ledger into an audit block', () => {
  assert.match(source, /n=\{1\}[\s\S]*title=\{appendixSection1\.title\}/);
  assert.match(source, /n=\{1\}[\s\S]*deck=\{appendixSection1\.deck\}/);
  assert.match(source, /n=\{1\}[\s\S]*appendixSection1\.slots\.intro/);
  assert.match(source, /n=\{1\}[\s\S]*appendixSection1\.slots\.takeaway/);
  assert.match(source, /n=\{1\}[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.auditIntro, 0\)/);
  assert.match(source, /n=\{1\}[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.auditIntro, 1\)/);
  assert.match(source, /n=\{1\}[\s\S]*appendixSection1\.slots\.columnGuide/);
  assert.match(source, /n=\{1\}[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 0\)/);
  assert.match(source, /n=\{1\}[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 1\)/);
  assert.match(source, /n=\{1\}[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 2\)/);
  assert.match(source, /n=\{1\}[\s\S]*renderAppendixSingleBlock\(appendixSection1\.slots\.followups, 3\)/);
  assert.match(source, /data-takeaway=\{String\.raw`G_\{\\mathrm\{pt\}\} is a cost group: it is valid for direct product\/update compression because it identifies equal indexed products under the declared equality model\.`\}/);
  assert.match(source, /n=\{1\}[\s\S]*className=\{APPENDIX_PROSE_JUSTIFIED_CLASS\}/);
  assert.match(source, /appendixSection1\.slots\.auditIntro/);
  assert.match(source, /n=\{1\}[\s\S]*appendixSection1\.slots\.columnGuide/);
  assert.match(source, /n=\{1\}[\s\S]*<Latex math="\|G_\{\\mathrm\{wreath\}\}\|" \/>/);
  assert.match(source, /n=\{1\}[\s\S]*<Latex math="\|G_\{\\text\{pt\}\}\|" \/>/);
  assert.doesNotMatch(source, /`\$\|G_\{\\mathrm\{wreath\}\}\|\$ counts candidate row moves before filtering\.[\s\S]*\$\|G_\{\\text\{pt\}\}\|\$ is the final detected pointwise group size\.`/);
  assert.match(source, /appendixSection1\.slots\.followups/);
  assert.match(source, /trace-product/);
  assert.match(source, /triangle/);
  assert.match(source, /young-s3/);
  assert.doesNotMatch(source, /n=\{1\}[\s\S]*Let \$\$\{notationLatex\('l_labels'\)\}\$ be the set of all labels/);
  assert.doesNotMatch(source, /n=\{1\}[\s\S]*When such a bijection exists, it induces a label relabeling/);
  assert.doesNotMatch(source, /<AppendixAuditBlock/);
  assert.doesNotMatch(source, /Audit detail/);
  assert.doesNotMatch(source, /Takeaway\.\s*<\/span>\s*<Latex math=\{String\.raw`G_\\\{\\text\\\{pt\\\}\\\}`/);
});

test('sections 2 through 4 introduce same-domain dummy renaming, then G_out, then G_f in dependency order', () => {
  assert.match(source, /n=\{2\}[\s\S]*title=\{appendixSection2\.title\}/);
  assert.match(source, /n=\{2\}[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection2\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /n=\{2\}[\s\S]*appendixSection2\.slots\.intro/);
  assert.match(source, /n=\{2\}[\s\S]*appendixSection2\.slots\.takeaway/);
  assert.match(source, /n=\{2\}[\s\S]*appendixSection2\.slots\.runningExampleLabelPrefix/);
  assert.match(source, /n=\{2\}[\s\S]*appendixSection2\.slots\.runningExamplePresetLabel/);
  assert.match(source, /n=\{2\}[\s\S]*appendixSection2\.slots\.runningExampleLead/);
  assert.match(source, /n=\{2\}[\s\S]*<FormulaHighlighted example=\{runningExamplePreset\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /const runningExampleExpandedEquation = useMemo\(/);
  assert.match(source, /buildExpandedEinsumEquation\(runningExamplePreset\)/);
  assert.match(source, /n=\{2\}[\s\S]*<Latex math=\{runningExampleExpandedEquation\} \/>/);
  assert.match(source, /n=\{2\}[\s\S]*A\[i,l\] \\cdot A\[j,k\]/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*This gives a second group, \$\$\{notationLatex\('s_w_summed'\)\}\$/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*Rename dummy variables back/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*The swap [\s\S]* preserves the double sum as a formal expression/);
  assert.match(source, /n=\{2\}[\s\S]*className=\{APPENDIX_PROSE_JUSTIFIED_CLASS\}/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*rounded-lg border border-gray-200 bg-white px-5 py-4 shadow-sm/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*4 \\neq 6/);
  assert.match(source, /n=\{3\}[\s\S]*title=\{appendixSection3\.title\}/);
  assert.match(source, /n=\{3\}[\s\S]*deck=\{appendixSection3\.deck\}/);
  assert.match(source, /n=\{3\}[\s\S]*appendixSection3\.slots\.definitionLead/);
  assert.match(source, /n=\{3\}[\s\S]*appendixSection3\.slots\.takeaway/);
  assert.match(source, /n=\{3\}[\s\S]*appendixSection3\.slots\.workedExampleLead/);
  assert.match(source, /n=\{3\}[\s\S]*appendixSection3\.slots\.workedExampleNote/);
  assert.match(source, /n=\{3\}[\s\S]*data-reader-facing-formula=/);
  assert.match(source, /const APPENDIX_G_OUT_DEFINITION_LATEX/);
  assert.match(source, /APPENDIX_PI_RESTRICT_V_LATEX/);
  assert.match(source, /n=\{3\}[\s\S]*appendixSection3\.slots\.workedExampleLead/);
  assert.match(source, /n=\{3\}[\s\S]*\\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.doesNotMatch(source, /n=\{3\}[\s\S]*Among the elements of \$\$\{notationLatex\('g_pointwise'\)\}\$/);
  assert.doesNotMatch(source, /n=\{3\}[\s\S]*supportClassName=\{`\$\{APPENDIX_SUPPORT_SHELF_CLASS\} xl:pt-5`\}/);
  assert.match(source, /n=\{4\}[\s\S]*title=\{appendixSection4\.title\}/);
  assert.match(source, /n=\{4\}[\s\S]*deck=\{appendixSection4\.deck\}/);
  assert.match(source, /n=\{4\}[\s\S]*appendixSection4\.slots\.intro/);
  assert.match(source, /n=\{4\}[\s\S]*appendixSection4\.slots\.takeaway/);
  assert.match(source, /n=\{4\}[\s\S]*renderAppendixSingleBlock\(appendixSection4\.slots\.constructionTitle, 0\)/);
  assert.match(source, /n=\{4\}[\s\S]*appendixSection4\.slots\.constructionNote/);
  assert.match(source, /n=\{4\}[\s\S]*<Latex math=\{String\.raw`G_\{\\text\{f\}\} = G_\{\\mathrm\{out\}\} \\times \\prod_d S\(W_d\)`\} \/>/);
  assert.match(source, /n=\{4\}[\s\S]*<VSubSwConstruction/);
  assert.doesNotMatch(source, /n=\{4\}[\s\S]*We can now name the label-renaming formal group considered in this appendix\./);
  assert.doesNotMatch(source, /n=\{4\}[\s\S]*rounded-xl border border-gray-200 bg-gray-50 px-5 py-4 text-center/);
  assert.doesNotMatch(source, /n=\{4\}[\s\S]*supportClassName=\{`\$\{APPENDIX_SUPPORT_SHELF_CLASS\} xl:pt-1`\}/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('s_w_summed'\)\}\$/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('g_output'\)\}\$/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('g_formal'\)\}\$/);
});

test('section 5 uses alphaComparison branches and the bilinear witness to reject Burnside on G_f for cost', () => {
  assert.match(source, /n=\{5\}[\s\S]*title=\{appendixSection5\.title\}/);
  assert.match(source, /n=\{5\}[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection5\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.intro/);
  assert.match(source, /n=\{5\}[\s\S]*renderAppendixSingleBlock\(appendixSection5\.slots\.rule, 0\)/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.presetPickerLabel/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.mismatchLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.coincidentLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.noneLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.workedExampleLabelPrefix/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.workedExampleLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.assignmentLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.bilinearOrbitLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.directOrbitLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.mixedOrbitLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.genericOrbitLead/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.genericAssignmentTemplate/);
  assert.match(source, /n=\{5\}[\s\S]*appendixSection5\.slots\.genericNoteTemplate/);
  assert.match(source, /function WorkedExampleIndex\(/);
  assert.match(source, /function WorkedExampleCoords\(/);
  assert.match(source, /function WorkedExampleTensorRef\(/);
  assert.match(source, /function WorkedExampleTensorProduct\(/);
  assert.match(source, /function WorkedExampleDisplayEquation\(/);
  assert.match(source, /function buildWorkedExampleFactors\(/);
  assert.match(source, /alphaComparison\.state === 'mismatch'/);
  assert.match(source, /alphaComparison\.state === 'coincident'/);
  assert.match(source, /alphaComparison\.state === 'none'/);
  assert.doesNotMatch(source, /n=\{5\}[\s\S]*The main page’s \$\\alpha\$ counts accumulation representatives/);
  assert.doesNotMatch(source, /n=\{5\}[\s\S]*A naive formal count using \$\$\{notationLatex\('g_formal'\)\}\$ gives \$\\alpha_\{\\text\{formal\}\}/);
  assert.doesNotMatch(source, /n=\{5\}[\s\S]*The pointwise accumulation count used by the engine is \$\\alpha_\{\\text\{engine\}\}/);
  assert.match(source, /\\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.doesNotMatch(source, /A = \[\[1, 2\], \[3, 4\]\]/);
  assert.match(source, /n=\{5\}[\s\S]*<WorkedExampleDisplayEquation/);
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
  const section5Start = source.indexOf('n={5}');
  const presetsIdx = source.indexOf('appendixSection5.slots.presetPickerLabel', section5Start);
  const mismatchIdx = source.indexOf("alphaComparison.state === 'mismatch'", section5Start);
  assert.equal(source.includes('appendixSection5.slots.selectedEinsumLabel'), false);
  assert.doesNotMatch(source, /Selected einsum:/);
  const selectedFormulaIdx = source.indexOf('<FormulaHighlighted example={example} hoveredLabels={null} />', section5Start);
  assert(selectedFormulaIdx < presetsIdx, 'preset jump row should follow the selected einsum line');
  assert(presetsIdx < mismatchIdx, 'preset jump row should appear before the branch-specific explanation block');
});

test('section 6 frames storage as a separate optimization axis with α_engine and α_storage', () => {
  assert.match(source, /n=\{6\}[\s\S]*title=\{appendixSection6\.title\}/);
  assert.match(source, /n=\{6\}[\s\S]*deckClassName="max-w-none"/);
  assert.match(source, /n=\{6\}[\s\S]*deck=\{\s*<InlineMathText>\s*\{normalizeAppendixDisplayText\(appendixSection6\.deck\)\}\s*<\/InlineMathText>\s*\}/);
  assert.match(source, /n=\{6\}[\s\S]*appendixSection6\.slots\.intro/);
  assert.match(source, /n=\{6\}[\s\S]*renderAppendixSingleBlock\(appendixSection6\.slots\.footnote, 0\)/);
  assert.match(source, /n=\{6\}[\s\S]*renderAppendixSingleBlock\(appendixSection6\.slots\.tableNote, 0\)/);
  assert.match(source, /n=\{6\}[\s\S]*renderAppendixSingleBlock\(appendixSection6\.slots\.scopeLabel, 0\)/);
  assert.match(source, /n=\{6\}[\s\S]*appendixSection6\.slots\.footer/);
  assert.match(source, /n=\{6\}[\s\S]*const operandChips = operands\.map\(\(operand\) => \(\{/);
  assert.match(source, /n=\{6\}[\s\S]*chipName: operand\.count > 1 \? `\$\{operand\.name\}×\$\{operand\.count\}` : operand\.name/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="flex flex-wrap gap-1\.5">\s*\{operandChips\.map\(\(operand\) => \(/);
  assert.match(source, /n=\{6\}[\s\S]*<SymmetryChip key=\{`\$\{operand\.name\}-\$\{operand\.sym\}`\} name=\{operand\.chipName\} symmetry=\{operand\.sym\} \/>/);
  assert.match(source, /import \{ buildStorageSavingsRows \} from '\.\.\/engine\/storageSavings\.js';/);
  assert.match(source, /const savingsTableRows = useMemo\(\s*\(\) => buildStorageSavingsRows\(EXAMPLES, 3\),\s*\[\],\s*\)/);
  assert.doesNotMatch(source, /const SAVINGS_TABLE_ROWS = \[/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*<span className="font-mono font-semibold">\{o\.name\}<\/span>/);
  assert.match(source, /Accumulation representatives/);
  assert.match(source, /Storage-aware output updates/);
  assert.match(source, /Storage-only saving/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="text-\[13px\] font-semibold text-gray-900">\s*<Latex math="\\alpha_\{\\text\{engine\}\}" \/>/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="mt-1 text-\[11px\] font-normal leading-5 text-gray-500">Accumulation representatives<\/div>/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="text-\[13px\] font-semibold text-gray-900">\s*<Latex math="\\alpha_\{\\text\{storage\}\}" \/>/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="mt-1 text-\[11px\] font-normal leading-5 text-gray-500">Storage-aware output updates<\/div>/);
  assert.match(source, /n=\{6\}[\s\S]*r\.vLatex === '\\\\varnothing' \? '\\\\varnothing' : `\\\\\{\$\{r\.vLatex\}\\\\\}`/);
  assert.match(source, /n=\{6\}[\s\S]*<Latex math=\{r\.vSubLatex\} \/>/);
  assert.match(source, /n=\{6\}[\s\S]*\{r\.alphaEngine\}/);
  assert.match(source, /n=\{6\}[\s\S]*\{r\.alphaStorage\}/);
  assert.match(source, /n=\{6\}[\s\S]*\$\{r\.saving\} \(\$\{r\.savingPct\}%\)/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*Accumulation is governed by \$\$\{notationLatex\('g_pointwise'\)\}\$/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*Output storage is governed by \$\$\{notationLatex\('g_output'\)\}\$/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*The dummy-label group \$\$\{notationLatex\('s_w_summed'\)\}\$ contributes nothing to output storage/);
  assert.doesNotMatch(source, /Appendix note/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*SCOPE/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*The \$\\alpha\$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model with \$\$\{notationLatex\('g_pointwise'\)\}\$ as the summand-value equivalence relation\./);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*Output-tensor storage collapse, algebraic restructuring such as factoring \$R = v v\^\\top\$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured on the main page\./);
  assert.doesNotMatch(source, /This appendix separates three optimization questions/);
  assert.doesNotMatch(source, /The formal group \$\$\{notationLatex\('g_formal'\)\}\$ is essential for explaining expression-level label-renaming symmetry/);
  assert.doesNotMatch(source, /space-y-3 text-\[13px\] leading-6 text-gray-700/);
});

test('appendix avoids raw unthemed math literals for key semantic symbols and example algebra', () => {
  assert.doesNotMatch(source, /<Latex math=\{'S\(W\)'\} \/>/);
  assert.doesNotMatch(source, /String\.raw`\\\$G_\{\\text\{f\}\} = G_\{\\mathrm\{out\}\} \\times S\(W\)\\\$`/);
  assert.doesNotMatch(source, /Rule\. Use \$G_\{\\\\text\{pt\}\}\$ for accumulation\./);
  assert.match(source, /notationColoredLatex\('s_w_summed', 'S\(W\)'\)/);
  assert.match(source, /WorkedExampleTensorProduct/);
  assert.match(source, /WorkedExampleDisplayEquation/);
});

test('appendix hover surfaces and shared typography registers remain intact', () => {
  assert.match(source, /import \{ createPortal \} from 'react-dom';/);
  assert.match(source, /function AppendixEinsumHoverCell\(/);
  assert.match(source, /function AppendixPresetHoverLabel\(/);
  assert.match(source, /createPortal\(/);
  assert.match(source, /pointer-events-none fixed z-\[9999\]/);
  assert.match(source, /const APPENDIX_PROSE_CLASS = 'font-serif text-\[17px\] leading-\[1\.75\] text-gray-900';/);
  assert.match(source, /const APPENDIX_PROSE_JUSTIFIED_CLASS = `\$\{APPENDIX_PROSE_CLASS\} text-justify`;/);
  assert.match(source, /const APPENDIX_ARTICLE_LANE_CLASS = 'max-w-\[78ch\] space-y-4 \[\&_p\]:text-justify';/);
  assert.match(source, /const APPENDIX_APP_TEXT_CLASS = 'text-\[13px\] leading-\[1\.55\] text-gray-700';/);
  assert.match(source, /const APPENDIX_SMALL_TEXT_CLASS = 'text-\[12px\] leading-5 text-gray-600';/);
  assert.match(source, /const APPENDIX_MONO_LEDGER_CLASS = 'font-mono text-\[13px\] leading-relaxed text-gray-900';/);
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
  assert.doesNotMatch(source, /n=\{5\}[\s\S]*<EditorialCallout[\s\S]*label="Rule"/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*<EditorialCallout[\s\S]*label="Scope"/);
  assert.doesNotMatch(source, /alphaComparison\.state === 'coincident' \|\| alphaComparison\.state === 'none'[\s\S]*<EditorialCallout[\s\S]*label="Try presets with a visible gap"/);
  assert.doesNotMatch(source, /label=\{label\}/);
  assert.doesNotMatch(source, /label=\{title\}/);
  assert.doesNotMatch(source, /function AppendixTakeaway[\s\S]*rounded-lg border border-gray-200 bg-gray-50 px-5 py-4/);
  assert.doesNotMatch(source, /function AppendixDefinitionPanel[\s\S]*rounded-xl border border-gray-200 bg-white px-5 py-4 shadow-sm/);
  assert.doesNotMatch(source, /n=\{5\}[\s\S]*rounded-lg border border-gray-900\/10 bg-stone-50 px-5 py-4/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*rounded-lg border border-gray-200 bg-gray-50 px-5 py-4/);
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
  assert.match(source, /The restriction <Latex math=\{String\.raw`G_\{\\text\{pt\}\}\\|_V`\} \/> to output labels/);
  assert.match(source, /same-domain dummy renamings of bound summation variables/);
  assert.match(source, /<Latex math=\{String\.raw`G_\{\\text\{f\}\} = G_\{\\mathrm\{out\}\} \\times \\prod_d S\(W_d\)`\} \/>/);
});
