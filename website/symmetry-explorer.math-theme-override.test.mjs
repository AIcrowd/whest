import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

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
