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
  assert.match(sidebarSource, /aria-label="Preset examples"/);
  assert.match(sidebarSource, /expectedGroup/);
  assert.match(chooserSource, /aria-label="Mobile preset examples"/);
  assert.match(chooserSource, /expectedGroup/);
});

test('ExampleChooser uses the shared Python code block and the wider 70\\/30 row', () => {
  const chooserSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx', import.meta.url), 'utf8');
  const codeBlockSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PythonCodeBlock.jsx', import.meta.url), 'utf8');

  assert.match(chooserSource, /PythonCodeBlock/);
  assert.match(chooserSource, /lg:grid-cols-\[minmax\(0,7fr\)_minmax\(320px,3fr\)\]/);
  assert.match(chooserSource, /checkpointItems\.length > 0/);
  assert.match(chooserSource, /Reference Code/);
  assert.match(codeBlockSource, /navigator\.clipboard\.writeText/);
  assert.match(codeBlockSource, /function highlightPython/);
});

test('PresetSidebar keeps preset rows compact but still shows the output symmetry', () => {
  const sidebarSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx', import.meta.url), 'utf8');
  assert.match(sidebarSource, /CaseBadge/);
  assert.match(sidebarSource, /summary\.expectedGroup/);
});
