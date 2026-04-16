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
  assert.match(chooserSource, /checkpointItems\.length > 0/);
  assert.match(chooserSource, /Reference Code/);
  assert.match(codeBlockSource, /navigator\.clipboard\.writeText/);
  assert.match(codeBlockSource, /function highlightPython/);
  assert.match(codeBlockSource, /ExplorerSectionCard/);
  assert.match(codeBlockSource, /Button/);
  assert.match(codeBlockSource, /\[&_\.hl-kw\]:font-semibold/);
  assert.match(codeBlockSource, /\[&_\.hl-str\]:text-emerald-300/);
});

test('PresetSidebar widens the rail and uses shared text sizing while still showing the output symmetry', () => {
  const sidebarSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx', import.meta.url), 'utf8');
  assert.match(sidebarSource, /w-\[18rem\]/);
  assert.match(sidebarSource, /px-3\.5 py-3/);
  assert.match(sidebarSource, /Define your own contraction/);
  assert.match(sidebarSource, /Keep the current builder state/);
  assert.match(sidebarSource, /text-xs font-semibold uppercase tracking-\[0\.18em\] text-primary\/75/);
  assert.match(sidebarSource, /text-sm text-gray-500/);
  assert.match(sidebarSource, /text-sm text-gray-400/);
  assert.doesNotMatch(sidebarSource, /text-\[10px\]/);
  assert.doesNotMatch(sidebarSource, /text-\[11px\]/);
  assert.match(sidebarSource, /CaseBadge/);
  assert.match(sidebarSource, /summary\.expectedGroup/);
});

test('CaseBadge compact variant uses the shared xs scale instead of micro text sizes', () => {
  const badgeSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/CaseBadge.jsx', import.meta.url), 'utf8');
  assert.match(badgeSource, /variant === 'compact'/);
  assert.match(badgeSource, /size === 'xs'[\s\S]*h-5 w-5 justify-center rounded-full px-0\.5 py-0 leading-none text-\[11px\] font-bold/);
  assert.doesNotMatch(badgeSource, /text-\[9px\]/);
});
