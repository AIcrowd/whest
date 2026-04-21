import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import katex from 'katex';
import { getRegimePresentation } from './components/symmetry-aware-einsum-contractions/components/regimePresentation.js';
import {
  colorizeNotationLatex,
  notationColor,
  notationLatex,
  resetActiveExplorerTheme,
  setActiveExplorerTheme,
} from './components/symmetry-aware-einsum-contractions/lib/notationSystem.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('Latex threads themeOverride into notation colorization and memoization', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/Latex.jsx');

  assert.match(src, /export default function Latex\(\{ math, display = false, colorize = true, themeOverride = null \}\)/);
  assert.match(src, /colorize \? colorizeNotationLatex\(math, themeOverride\) : math/);
  assert.match(src, /\[math, display, colorize, activeExplorerThemeId, themeOverride\]/);
});

test('InlineMathText accepts themeOverride and forwards it to nested Latex', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/InlineMathText.jsx');

  assert.match(src, /export default function InlineMathText\(\{ children, themeOverride = null \}\)/);
  assert.match(src, /<Latex key=\{`math-\$\{i\}`\} math=\{part\.slice\(1, -1\)\} themeOverride=\{themeOverride\} \/>/);
});

test('GlossaryProse accepts themeOverride and forwards it to nested Latex', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/GlossaryProse.jsx');

  assert.match(src, /export default function GlossaryProse\(\{ text, themeOverride = null \}\)/);
  assert.match(src, /<Latex key=\{part\.key\} math=\{part\.value\} themeOverride=\{themeOverride\} \/>/);
});

test('Section 5 override path applies editorial-noir-math colors at runtime', (t) => {
  setActiveExplorerTheme('editorial-noir');
  t.after(() => resetActiveExplorerTheme());

  const defaultYoungPresentation = getRegimePresentation('young');
  const overrideYoungPresentation = getRegimePresentation('young', 'editorial-noir-math');
  const sectionFiveMath = [
    notationLatex('alpha_component'),
    notationLatex('n_l'),
    notationLatex('g_component'),
  ].join(' + ');
  const defaultMarkup = katex.renderToString(colorizeNotationLatex(sectionFiveMath), {
    throwOnError: false,
    trust: true,
  });
  const overrideMarkup = katex.renderToString(colorizeNotationLatex(sectionFiveMath, 'editorial-noir-math'), {
    throwOnError: false,
    trust: true,
  });

  assert.equal(defaultYoungPresentation.color, notationColor('alpha_total', 'editorial-noir'));
  assert.equal(overrideYoungPresentation.color, notationColor('alpha_total', 'editorial-noir-math'));
  assert.notEqual(defaultYoungPresentation.color, overrideYoungPresentation.color);

  assert.ok(defaultMarkup.includes(notationColor('alpha_component', 'editorial-noir')));
  assert.ok(!defaultMarkup.includes(notationColor('alpha_component', 'editorial-noir-math')));
  assert.ok(overrideMarkup.includes(notationColor('alpha_component', 'editorial-noir-math')));
  assert.ok(overrideMarkup.includes(notationColor('n_l', 'editorial-noir-math')));
  assert.ok(overrideMarkup.includes(notationColor('g_component', 'editorial-noir-math')));
  assert.ok(!overrideMarkup.includes(notationColor('alpha_component', 'editorial-noir')));
});
