import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) { return readFileSync(resolve(__dirname, rel), 'utf-8'); }

test('section4 intro is exactly 2 paragraphs after relocation', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/section4.js');
  const introMatch = src.match(/intro:\s*\[([\s\S]*?)\],\s*produces:/);
  assert.ok(introMatch, 'intro slot should be present');
  const paragraphCount = (introMatch[1].match(/p\(/g) ?? []).length;
  assert.equal(paragraphCount, 2, `expected 2 paragraphs in intro, got ${paragraphCount}`);
});

test('section4 intro keeps ¶1 and ¶2 verbatim and drops ¶3 and ¶4', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/section4.js');
  assert.match(src, /The group \$G_\{\\\\text\{pt\}\}\$ acts on the full assignment grid \$X\$/);
  assert.match(src, /Accumulation is the part that is easy to over-compress/);
  assert.doesNotMatch(src, /A product-orbit representative can contain many full index assignments/);
  assert.doesNotMatch(src, /When projection branches, the explorer can count exactly without expanding the full assignment grid/);
});
