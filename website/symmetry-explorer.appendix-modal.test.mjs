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
  assert.match(section1, /σ-loop audit/);

  assert.match(section2, /AppendixSupportSplit/);
  assert.match(section2, /className="lg:grid-cols-\[0\.95fr_1\.25fr\]"/);
  assert.match(section2, /Running example —\{' '\}[\s\S]*bilinear trace/);

  assert.match(section3, /AppendixSupportSplit/);
  assert.match(section3, /Worked example — bilinear trace/);

  assert.match(section4, /AppendixSupportSplit/);
  assert.match(section4, /<h4 className="font-heading text-\[18px\] font-semibold text-gray-900">\s*Formal-group construction\s*<\/h4>/);
  assert.match(section4, /<VSubSwConstruction[\s\S]*showHeading=\{false\}/);
  assert.match(section4, /VSubSwConstruction/);

  assert.match(section5, /AppendixSupportSplit/);
  assert.match(section5, /Selected einsum/);
  assert.match(section5, /supportClassName="space-y-5 xl:pt-1"/);
  assert.doesNotMatch(section5, /supportClassName=\{APPENDIX_SUPPORT_SHELF_CLASS\}/);

  assert.doesNotMatch(section6, /AppendixSupportSplit/);
  assert.match(section6, /Storage-only saving/);
  assert.match(section6, /deckClassName="max-w-none"/);
  assert.match(section6, /-mx-6 -mb-10 mt-8 border-t border-stone-200\/70 bg-gray-50 px-6 py-4 md:-mx-8 md:px-8 lg:-mx-10 lg:px-10/);
  assert.match(section6, /SCOPE/);
  assert.match(section6, /text-\[12\.5px\] leading-6 text-stone-700/);
  assert.doesNotMatch(section6, /max-w-\[78ch\]/);
  assert(section6.indexOf('Storage-only saving') < section6.indexOf('-mx-6 -mb-10 mt-8 border-t border-stone-200/70 bg-gray-50 px-6 py-4 md:-mx-8 md:px-8 lg:-mx-10 lg:px-10'), 'footer band should come after the table');
  assert(section6.indexOf('The table below records the storage-only savings available') > section6.indexOf('Storage-only saving'), 'storage explanatory paragraph should move below the table');

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
  assert.match(source, /n=\{1\}[\s\S]*Let \$\$\{notationLatex\('l_labels'\)\}\$ be the set of all labels/);
  assert.match(source, /n=\{1\}[\s\S]*APPENDIX_SHORT_V_LATEX/);
  assert.match(source, /n=\{1\}[\s\S]*APPENDIX_M_SIGMA_LATEX/);
  assert.match(source, /n=\{1\}[\s\S]*APPENDIX_PI_SIGMA_LATEX/);
  assert.match(source, /n=\{1\}[\s\S]*The group generated by the admissible relabelings is \$\$\{notationLatex\('g_pointwise'\)\}\$/);
  assert.match(source, /data-takeaway=\{String\.raw`G_\{\\text\{pt\}\} is a cost group: it is valid for accumulation because it identifies genuinely equal indexed products\.`\}/);
  assert.match(source, /n=\{1\}[\s\S]*className=\{APPENDIX_PROSE_JUSTIFIED_CLASS\}/);
  assert.match(source, /σ-loop audit/);
  assert.match(source, /The table below is an implementation audit of the \$\\sigma\$-loop for representative presets\./);
  assert.match(source, /n=\{1\}[\s\S]*Columns\./);
  assert.match(source, /n=\{1\}[\s\S]*counts candidate row moves before filtering/);
  assert.match(source, /n=\{1\}[\s\S]*Recorded counts admissible non-identity relabelings/);
  assert.match(source, /n=\{1\}[\s\S]*final detected pointwise group size/);
  assert.match(source, /n=\{1\}[\s\S]*<Latex math=\{String\.raw`\|G_\{\\mathrm\{wreath\}\}\|`\} \/>/);
  assert.match(source, /n=\{1\}[\s\S]*<Latex math=\{String\.raw`\|G_\{\\text\{pt\}\}\|`\} \/>/);
  assert.doesNotMatch(source, /`\$\|G_\{\\mathrm\{wreath\}\}\|\$ counts candidate row moves before filtering\.[\s\S]*\$\|G_\{\\text\{pt\}\}\|\$ is the final detected pointwise group size\.`/);
  assert.match(source, /Frobenius/);
  assert.match(source, /trace-product/);
  assert.match(source, /triangle/);
  assert.match(source, /young-s3/);
  assert.doesNotMatch(source, /<AppendixAuditBlock/);
  assert.doesNotMatch(source, /Audit detail/);
  assert.doesNotMatch(source, /Takeaway\.\s*<\/span>\s*<Latex math=\{String\.raw`G_\\\{\\text\\\{pt\\\}\\\}`/);
});

test('sections 2 through 4 introduce S(W), then G_out, then G_f in dependency order', () => {
  assert.match(source, /n=\{2\}[\s\S]*This gives a second group, \$\$\{notationLatex\('s_w_summed'\)\}\$/);
  assert.match(source, /n=\{2\}[\s\S]*Running example —/);
  assert.match(source, /n=\{2\}[\s\S]*<FormulaHighlighted example=\{runningExamplePreset\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /const runningExampleExpandedEquation = useMemo\(/);
  assert.match(source, /buildExpandedEinsumEquation\(runningExamplePreset\)/);
  assert.match(source, /n=\{2\}[\s\S]*<Latex math=\{runningExampleExpandedEquation\} \/>/);
  assert.match(source, /n=\{2\}[\s\S]*A\[i,l\] \\cdot A\[j,k\]/);
  assert.match(source, /n=\{2\}[\s\S]*Rename dummy variables back/);
  assert.match(source, /n=\{2\}[\s\S]*APPENDIX_SHORT_W_LATEX/);
  assert.match(source, /n=\{2\}[\s\S]*The swap [\s\S]* preserves the double sum as a formal expression/);
  assert.match(source, /n=\{2\}[\s\S]*className=\{APPENDIX_PROSE_JUSTIFIED_CLASS\}/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*rounded-lg border border-gray-200 bg-white px-5 py-4 shadow-sm/);
  assert.doesNotMatch(source, /n=\{2\}[\s\S]*4 \\neq 6/);
  assert.match(source, /n=\{3\}[\s\S]*data-reader-facing-formula=/);
  assert.match(source, /const APPENDIX_G_OUT_DEFINITION_LATEX/);
  assert.match(source, /APPENDIX_PI_RESTRICT_V_LATEX/);
  assert.match(source, /n=\{3\}[\s\S]*R\[i,j\] = R\[j,i\]/);
  assert.match(source, /n=\{3\}[\s\S]*R\[0,1\] = R\[1,0\]/);
  assert.match(source, /n=\{3\}[\s\S]*\\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.doesNotMatch(source, /n=\{3\}[\s\S]*supportClassName=\{`\$\{APPENDIX_SUPPORT_SHELF_CLASS\} xl:pt-5`\}/);
  assert.match(source, /n=\{4\}[\s\S]*APPENDIX_SHORT_S_W_LATEX/);
  assert.match(source, /n=\{4\}[\s\S]*<Latex math=\{`\$\{notationLatex\('g_formal'\)\} = \$\{notationLatex\('g_output'\)\} \\\\times \$\{APPENDIX_SHORT_S_W_LATEX\(\)\}`\} \/>/);
  assert.match(source, /n=\{4\}[\s\S]*<VSubSwConstruction/);
  assert.match(source, /n=\{4\}[\s\S]*The construction above enumerates the two factors and their product for the selected preset\./);
  assert.doesNotMatch(source, /n=\{4\}[\s\S]*rounded-xl border border-gray-200 bg-gray-50 px-5 py-4 text-center/);
  assert.doesNotMatch(source, /n=\{4\}[\s\S]*supportClassName=\{`\$\{APPENDIX_SUPPORT_SHELF_CLASS\} xl:pt-1`\}/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('s_w_summed'\)\}\$/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('g_output'\)\}\$/);
  assert.doesNotMatch(source, /Takeaway\.\s*\\\$?\$\$\{notationLatex\('g_formal'\)\}\$/);
});

test('section 5 uses alphaComparison branches and the bilinear witness to reject Burnside on G_f for cost', () => {
  assert.match(source, /n=\{5\}[\s\S]*The main page’s \$\\alpha\$ counts accumulation representatives/);
  assert.match(source, /function WorkedExampleIndex\(/);
  assert.match(source, /function WorkedExampleCoords\(/);
  assert.match(source, /function WorkedExampleTensorRef\(/);
  assert.match(source, /function WorkedExampleTensorProduct\(/);
  assert.match(source, /function WorkedExampleDisplayEquation\(/);
  assert.match(source, /function buildWorkedExampleFactors\(/);
  assert.match(source, /alphaComparison\.state === 'mismatch'/);
  assert.match(source, /alphaComparison\.state === 'coincident'/);
  assert.match(source, /alphaComparison\.state === 'none'/);
  assert.match(source, /A naive formal count using \$\$\{notationLatex\('g_formal'\)\}\$ gives \$\\alpha_\{\\text\{formal\}\}/);
  assert.match(source, /The pointwise accumulation count used by the engine is \$\\alpha_\{\\text\{engine\}\}/);
  assert.match(source, /\\begin\{bmatrix\} 1 & 2 \\\\ 3 & 4 \\end\{bmatrix\}/);
  assert.doesNotMatch(source, /A = \[\[1, 2\], \[3, 4\]\]/);
  assert.match(source, /Then the expanded form of the einsum is:/);
  assert.match(source, /n=\{5\}[\s\S]*<WorkedExampleDisplayEquation/);
  assert.match(source, /Now set <\/span>[\s\S]*<Latex math=\{String\.raw`i = 0,\\; j = 1`\} \/>/);
  assert.match(source, /outputCoords=\{\[0, 1\]\}/);
  assert.match(source, /sumCoords=\{\['k', 'l'\]\}/);
  assert.match(source, /same formal orbit/);
  assert.match(source, /scalarValues=\{\[1, 4\]\}/);
  assert.match(source, /scalarValues=\{\[2, 3\]\}/);
  assert.match(source, /const showDirectS2C3FormalOrbitExample =/);
  assert.match(source, /const showMixedChainFormalOrbitExample =/);
  assert.match(source, /example\?\.id === 'direct-s2-c3'/);
  assert.match(source, /example\?\.id === 'mixed-chain'/);
  assert.match(source, /T\[0,0,0,1,2\] = 1/);
  assert.match(source, /T\[0,0,0,2,1\] = 2/);
  assert.match(source, /sumCoords=\{\['c', 'd', 'e'\]\}/);
  assert.match(source, /The transposition \$\(d\\,e\)\$ is allowed as a dummy relabeling in \$\$\{notationLatex\('g_formal'\)\}\$/);
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
  assert.match(source, /Both assignments contribute to the same \$\{formalOrbitExample\?\.outputTargetNoun \?\? 'output entry'\}/);
  assert.doesNotMatch(source, /Rule\./);
  assert.equal(source.includes("Use $${notationLatex('g_pointwise')}$ for accumulation."), true);
  assert.equal(source.includes("Use $${notationLatex('g_formal')}$ to describe formal symmetry of the completed expression."), true);
  assert.equal(source.includes("Do not use the dummy-label factor $${notationLatex('s_w_summed')}$ to remove summand computations."), true);
  assert.match(source, /Selected einsum:/);
  assert.match(source, /font-mono text-\[13px\] leading-6 text-stone-900/);
  assert.match(source, /<FormulaHighlighted example=\{example\} hoveredLabels=\{null\} \/>/);
  assert.match(source, /selectedOperandItems = useMemo/);
  assert.match(source, /selectedOperandItems\.map/);
  assert.match(source, /<SymmetryChip key=\{`\$\{operand\.name\}-\$\{operand\.sym\}`\} name=\{operand\.chipName\} symmetry=\{operand\.sym\} \/>/);
  assert.doesNotMatch(source, /Selected einsum[\s\S]*rounded-xl border border-stone-200 bg-white px-3 py-2\.5 shadow-sm/);
  assert.doesNotMatch(source, /alphaComparison\.state === 'coincident' \|\| alphaComparison\.state === 'none'[\s\S]*<EditorialCallout/);
  assert.match(source, /Presets with a visible mismatch:/);
  assert.doesNotMatch(source, /alphaComparison\.state === 'coincident' \|\| alphaComparison\.state === 'none'/);
  assert.doesNotMatch(source, /formatWitnessTuple\(alphaComparison\.witness\.tupleA\)/);
  assert.doesNotMatch(source, /formatWitnessTuple\(alphaComparison\.witness\.tupleB\)/);
  assert.doesNotMatch(source, /The following two assignments lie in the same formal orbit and contribute to the same output entry, but they belong to different pointwise orbits and produce different products\./);
  const section5Start = source.indexOf('n={5}');
  const presetsIdx = source.indexOf('Presets with a visible mismatch:', section5Start);
  const mismatchIdx = source.indexOf("alphaComparison.state === 'mismatch'", section5Start);
  assert(source.indexOf('Selected einsum:', section5Start) < presetsIdx, 'preset jump row should follow the selected einsum header');
  assert(presetsIdx < mismatchIdx, 'preset jump row should appear before the branch-specific explanation block');
});

test('section 6 frames storage as a separate optimization axis with α_engine and α_storage', () => {
  assert.match(source, /n=\{6\}[\s\S]*Accumulation is governed by \$\$\{notationLatex\('g_pointwise'\)\}\$/);
  assert.match(source, /n=\{6\}[\s\S]*Output storage is governed by \$\$\{notationLatex\('g_output'\)\}\$/);
  assert.match(source, /n=\{6\}[\s\S]*The dummy-label group \$\$\{notationLatex\('s_w_summed'\)\}\$ contributes nothing to output storage/);
  assert.match(source, /n=\{6\}[\s\S]*const operandChips = operands\.map\(\(operand\) => \(\{/);
  assert.match(source, /n=\{6\}[\s\S]*chipName: operand\.count > 1 \? `\$\{operand\.name\}×\$\{operand\.count\}` : operand\.name/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="flex flex-wrap gap-1\.5">\s*\{operandChips\.map\(\(operand\) => \(/);
  assert.match(source, /n=\{6\}[\s\S]*<SymmetryChip key=\{`\$\{operand\.name\}-\$\{operand\.sym\}`\} name=\{operand\.chipName\} symmetry=\{operand\.sym\} \/>/);
  assert.doesNotMatch(source, /n=\{6\}[\s\S]*<span className="font-mono font-semibold">\{o\.name\}<\/span>/);
  assert.match(source, /Accumulation representatives/);
  assert.match(source, /Output-storage representatives/);
  assert.match(source, /Storage-only saving/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="text-\[13px\] font-semibold text-gray-900">\s*<Latex math="\\alpha_\{\\text\{engine\}\}" \/>/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="mt-1 text-\[11px\] font-normal leading-5 text-gray-500">Accumulation representatives<\/div>/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="text-\[13px\] font-semibold text-gray-900">\s*<Latex math="\\alpha_\{\\text\{storage\}\}" \/>/);
  assert.match(source, /n=\{6\}[\s\S]*<div className="mt-1 text-\[11px\] font-normal leading-5 text-gray-500">Output-storage representatives<\/div>/);
  assert.match(source, /All rows are computed at <Latex math="n = 3" \/>/);
  assert.match(source, /The table below records the storage-only savings available for the presets at \$n = 3\$/);
  assert.equal(source.includes('The column $\\\\alpha_{\\\\text{engine}}$ is the accumulation representative count used by the main page.'), true);
  assert.equal(source.includes("The column $\\\\alpha_{\\\\text{storage}}$ is the number of output-storage representatives after grouping output cells into $${notationLatex('g_output')}$-orbits."), true);
  assert.equal(source.includes('These are different quantities; $\\\\alpha_{\\\\text{storage}}$ is not a replacement for the accumulation cost.'), true);
  assert.doesNotMatch(source, /Appendix note/);
  assert.match(source, /SCOPE/);
  assert.match(source, /The \$\\alpha\$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model with \$\$\{notationLatex\('g_pointwise'\)\}\$ as the summand-value equivalence relation\./);
  assert.match(source, /Output-tensor storage collapse, algebraic restructuring such as factoring \$R = v v\^\\top\$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured on the main page\./);
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
  assert.match(source, /alpha-renamings of bound summation variables/);
});
