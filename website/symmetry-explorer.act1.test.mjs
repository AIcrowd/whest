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
  assert.match(appSource, /<PresetSidebar[\s\S]*<main className="min-w-0 flex-1">/);
  assert.match(sidebarSource, /aria-label="Preset examples"/);
  assert.match(sidebarSource, /ExplorerSidebarItem/);
  assert.match(sidebarSource, /Badge/);
  assert.match(sidebarSource, /expectedGroup/);
  assert.match(chooserSource, /aria-label="Mobile preset examples"/);
  assert.match(chooserSource, /<Button[\s\S]*variant="outline"[\s\S]*h-auto[\s\S]*items-start[\s\S]*justify-start/);
  assert.match(chooserSource, /<span className="flex items-center gap-2">/);
  assert.match(
    chooserSource,
    /activePresetIdx === idx[\s\S]*border-coral bg-coral-light\/50 ring-2 ring-coral\/30[\s\S]*border-gray-200 hover:border-gray-300/,
  );
  assert.match(chooserSource, /gap-3 px-4 py-3/);
  assert.match(chooserSource, /text-sm text-gray-500/);
  assert.match(chooserSource, /text-sm text-gray-400/);
  assert.doesNotMatch(chooserSource, /text-\[10px\]/);
  assert.doesNotMatch(chooserSource, /text-\[11px\]/);
  assert.match(chooserSource, /expectedGroup/);
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
  assert.match(codeBlockSource, /ExplorerSectionCard/);
  assert.match(codeBlockSource, /Button/);
  assert.match(codeBlockSource, /\[&_\.hl-kw\]:font-semibold/);
  assert.match(codeBlockSource, /\[&_\.hl-str\]:text-emerald-300/);
});

test('PresetSidebar matches the design-system preset-list spec (flat container, 10px gray kicker, canonical padding)', () => {
  const sidebarSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx', import.meta.url), 'utf8');
  assert.match(sidebarSource, /w-\[18rem\]/);
  // Spec padding: items sit inside an outer gray-200 rounded container,
  // `pl-5` leaves room for the 4px coral left rail on active items.
  assert.match(sidebarSource, /px-4 py-3 pl-5/);
  assert.match(sidebarSource, /Define your own contraction/);
  assert.match(sidebarSource, /Keep the current builder state/);
  // Kicker follows the `--text-10` / gray-400 / 0.2em tracking spec from
  // design-system/colors_and_type.css (`.w-kicker` default register).
  assert.match(sidebarSource, /text-\[10px\] font-semibold uppercase tracking-\[0\.2em\] text-gray-400/);
  // Outer flat container wrapping the preset rows.
  assert.match(sidebarSource, /divide-y divide-gray-100[\s\S]*rounded-lg border border-gray-200/);
  assert.match(sidebarSource, /text-sm text-gray-500/);
  assert.match(sidebarSource, /text-sm text-gray-400/);
  // 11px body text is still disallowed — the kicker is the one exception
  // and uses text-[10px] (spec-specified for eyebrows).
  assert.doesNotMatch(sidebarSource, /text-\[11px\]/);
  assert.match(sidebarSource, /CaseBadge/);
  assert.match(sidebarSource, /summary\.expectedGroup/);
  // Active rail is always coral — regime identity lives on CaseBadge, not
  // the rail (was: `style={{ backgroundColor: summary.color }}`).
  assert.match(sidebarSource, /rounded-\[2px\] bg-coral/);
  assert.doesNotMatch(sidebarSource, /backgroundColor: summary\.color/);
});

test('CaseBadge compact variant uses the shared xs scale instead of micro text sizes', () => {
  const badgeSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/CaseBadge.jsx', import.meta.url), 'utf8');
  assert.match(badgeSource, /variant === 'compact'/);
  assert.match(badgeSource, /size === 'xs'[\s\S]*h-5 w-5 justify-center rounded-full px-0\.5 py-0 leading-none text-\[11px\] font-bold/);
  assert.doesNotMatch(badgeSource, /text-\[9px\]/);
});
