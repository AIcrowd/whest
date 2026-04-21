import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('Act 1 uses a desktop preset rail and a mobile preset fallback', () => {
  const appSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');
  const chooserSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx', import.meta.url), 'utf8');
  const sidebarSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx', import.meta.url), 'utf8');

  assert.match(appSource, /PresetSidebar/);
  assert.match(appSource, /selectedPresetIdx=\{selectedPresetIdx\}/);
  assert.match(appSource, /getPresetControlSelection\(exampleIdx, isDirty\)/);
  assert.match(appSource, /mx-auto mt-8 w-full max-w-\[1460px\] px-6 md:px-8 lg:px-10/);
  assert.match(appSource, /<PresetSidebar[\s\S]*<main className="min-w-0 flex-1">/);
  assert.match(sidebarSource, /aria-label="Preset examples"/);
  assert.match(sidebarSource, /ExplorerSidebarItem/);
  assert.match(sidebarSource, /worked contractions/);
  assert.match(sidebarSource, /expectedGroup/);
  assert.match(sidebarSource, /summary\.caseIds\?\.map/);
  assert.match(chooserSource, /aria-label="Mobile preset examples"/);
  assert.match(chooserSource, /overflow-hidden rounded-lg border border-gray-200 bg-white/);
  assert.match(chooserSource, /text-\[10px\] font-semibold uppercase tracking-\[0\.2em\] text-gray-400/);
  assert.match(chooserSource, /glyph=\{summary\.glyph\}/);
  assert.match(chooserSource, /formula=\{summary\.formula\}/);
  assert.match(chooserSource, /activePresetIdx === idx \? 'bg-coral-light\/50' : 'hover:bg-gray-50'/);
  // 10px is legitimate for design-system kickers (.w-kicker is 10/0.2em/
  // gray-400 in colors_and_type.css). The builder's 'VARIABLES' /
  // 'subscripts' / 'output' / 'operands' labels are kickers at spec size.
  // 11px was and still is disallowed here — reserved for the PresetSidebar
  // kicker under the paper-register register.
  assert.doesNotMatch(chooserSource, /text-\[11px\]/);
  assert.match(chooserSource, /expectedGroup/);
  assert.match(chooserSource, /summary\.caseIds\?\.map/);
});

test('ExampleChooser uses the shared Python code block and current builder primitives', () => {
  const chooserSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx', import.meta.url), 'utf8');
  const codeBlockSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PythonCodeBlock.jsx', import.meta.url), 'utf8');

  assert.match(chooserSource, /PythonCodeBlock/);
  assert.match(chooserSource, /ExplorerField/);
  assert.match(chooserSource, /Button/);
  assert.match(chooserSource, /Input/);
  assert.match(codeBlockSource, /navigator\.clipboard\.writeText/);
  assert.match(codeBlockSource, /function highlightPython/);
  assert.match(codeBlockSource, /const PRIMARY_FUNCTIONS = new Set\(\['randn', 'einsum_path'\]\)/);
  assert.match(codeBlockSource, /ExplorerSectionCard/);
  assert.match(codeBlockSource, /Button/);
  assert.match(codeBlockSource, /hl-fn-primary/);
  assert.match(codeBlockSource, /\[&_\.hl-kw\]:font-semibold/);
  assert.match(codeBlockSource, /bg-white/);
  assert.match(codeBlockSource, /\[&_\.hl-str\]:text-emerald-700/);
  assert.match(codeBlockSource, /text-\[#ef5a4c\]/);
});

test('ExampleChooser increments added variable names spreadsheet-style', () => {
  const chooserSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx', import.meta.url), 'utf8');

  assert.match(chooserSource, /function nextVariableName/);
  assert.match(chooserSource, /while \(idx >= 0 && chars\[idx\] === 'Z'\)/);
  assert.match(chooserSource, /chars\.unshift\('A'\)/);
  assert.match(chooserSource, /name: nextVariableName\(lastName\)/);
});

test('PresetSidebar matches the design-system preset-list spec (flat container, 10px gray kicker, canonical padding)', () => {
  const sidebarSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx', import.meta.url), 'utf8');
  const presetSelectionSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/lib/presetSelection.js', import.meta.url), 'utf8');
  assert.match(sidebarSource, /w-\[18rem\][\s\S]*xl:w-\[20rem\]/);
  assert.match(sidebarSource, /border-b border-gray-100/);
  assert.match(sidebarSource, /glyph="⚙"/);
  assert.match(sidebarSource, /formula="— build below —"/);
  assert.match(sidebarSource, /Keep the current builder state/);
  assert.match(sidebarSource, /text-\[10px\] font-semibold uppercase tracking-\[0\.2em\] text-gray-400/);
  assert.match(sidebarSource, /overflow-hidden border-x border-gray-200 bg-white/);
  assert.match(sidebarSource, /border-b border-gray-100 px-4 py-4/);
  assert.match(sidebarSource, /divide-y divide-gray-100/);
  assert.match(sidebarSource, /glyph=\{summary\.glyph\}/);
  assert.match(sidebarSource, /description=\{summary\.description\}/);
  assert.match(sidebarSource, /formula=\{summary\.formula\}/);
  // 11px body text is still disallowed — the kicker is the one exception
  // and uses text-[10px] (spec-specified for eyebrows).
  assert.match(sidebarSource, /CaseBadge/);
  assert.match(sidebarSource, /summary\.expectedGroup/);
  assert.match(sidebarSource, /summary\.caseIds\?\.map/);
  // Active rail is always coral — regime identity lives on CaseBadge, not
  // the rail (was: `style={{ backgroundColor: summary.color }}`).
  assert.match(sidebarSource, /rounded-\[2px\] bg-coral/);
  assert.doesNotMatch(sidebarSource, /backgroundColor: summary\.color/);
  assert.match(presetSelectionSource, /PRESET_GLYPHS_BY_CLASSIFICATION/);
  assert.match(presetSelectionSource, /trivial: '·'/);
  assert.match(presetSelectionSource, /allVisible: '◌'/);
  assert.match(presetSelectionSource, /allSummed: '∑'/);
  assert.match(presetSelectionSource, /mixed: '⟡'/);
  assert.match(presetSelectionSource, /singleton: '①'/);
  assert.match(presetSelectionSource, /directProduct: '⊗'/);
  assert.match(presetSelectionSource, /young: 'Y'/);
  assert.match(presetSelectionSource, /bruteForceOrbit: '◎'/);
  assert.doesNotMatch(presetSelectionSource, /expectedGroup.*includes/);
});

test('CaseBadge compact variant uses the shared xs scale instead of micro text sizes', () => {
  const badgeSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/CaseBadge.jsx', import.meta.url), 'utf8');
  assert.match(badgeSource, /variant === 'compact'/);
  assert.match(badgeSource, /size === 'xs'[\s\S]*h-5 w-5 justify-center rounded-full px-0\.5 py-0 leading-none text-\[11px\] font-bold/);
  assert.doesNotMatch(badgeSource, /text-\[9px\]/);
});
