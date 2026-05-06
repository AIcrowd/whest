import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

// V3.1 lens preset IDs and their expected names.
const LENS_PRESETS = [
  { id: 'cross-s2',         name: 'Cross S₂' },
  { id: 'triple-outer',     name: 'Triple outer (S3)' },
  { id: 'triangle',         name: 'Directed triangle' },
  { id: 'cross-c3-partial', name: 'Cross C₃ (partial group)' },
  { id: 'bilinear-trace',   name: 'Bilinear trace' },
];

test('all 5 V3.1 lens presets are defined in the EXAMPLES registry', () => {
  for (const { id } of LENS_PRESETS) {
    const found = EXAMPLES.find((e) => e.id === id);
    assert.ok(found, `Lens preset '${id}' is missing from EXAMPLES`);
  }
});

test('each lens preset carries the expected declared symmetry structure', () => {
  const crossS2 = EXAMPLES.find((e) => e.id === 'cross-s2');
  assert.ok(crossS2, 'cross-s2 not found');
  const aVar = crossS2.variables.find((v) => v.name === 'A');
  assert.ok(aVar, 'cross-s2 must have an A variable');
  assert.equal(aVar.symmetry, 'symmetric', 'cross-s2 A must be symmetric');

  const tripleOuter = EXAMPLES.find((e) => e.id === 'triple-outer');
  assert.ok(tripleOuter, 'triple-outer not found');
  assert.equal(tripleOuter.expression.subscripts, 'ia,ib,ic', 'triple-outer subscripts mismatch');

  const triangle = EXAMPLES.find((e) => e.id === 'triangle');
  assert.ok(triangle, 'triangle not found');
  assert.equal(triangle.expression.subscripts, 'ij,jk,ki', 'triangle subscripts mismatch');

  const crossC3 = EXAMPLES.find((e) => e.id === 'cross-c3-partial');
  assert.ok(crossC3, 'cross-c3-partial not found');
  const tVar = crossC3.variables.find((v) => v.name === 'T');
  assert.ok(tVar, 'cross-c3-partial must have a T variable');
  assert.equal(tVar.symmetry, 'cyclic', 'cross-c3-partial T must be cyclic');

  const bilinear = EXAMPLES.find((e) => e.id === 'bilinear-trace');
  assert.ok(bilinear, 'bilinear-trace not found');
  assert.equal(bilinear.expression.subscripts, 'ik,jl', 'bilinear-trace subscripts mismatch');
});

test('default preset in SymmetryAwareEinsumContractionsApp is cross-s2', () => {
  const src = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  assert.match(src, /DEFAULT_EXAMPLE_ID = 'cross-s2'/, "DEFAULT_EXAMPLE_ID must be 'cross-s2'");
});

test('PresetSidebar renders a "Pedagogical lenses" group heading', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx');
  assert.match(src, /Pedagogical lenses/, 'PresetSidebar must contain "Pedagogical lenses" heading');
});

test('PresetSidebar renders a "Reference presets" group heading', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx');
  assert.match(src, /Reference presets/, 'PresetSidebar must contain "Reference presets" heading');
});

test('PresetSidebar LENS_PRESET_IDS contains all 5 lens presets in the correct order', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx');
  const lensIds = ['cross-s2', 'triple-outer', 'triangle', 'cross-c3-partial', 'bilinear-trace'];
  for (const id of lensIds) {
    assert.match(src, new RegExp(`'${id}'`), `LENS_PRESET_IDS must include '${id}'`);
  }
  // Verify ordering: cross-s2 appears before triple-outer in source
  const crossS2Pos = src.indexOf("'cross-s2'");
  const tripleOuterPos = src.indexOf("'triple-outer'");
  assert.ok(crossS2Pos < tripleOuterPos, 'cross-s2 must appear before triple-outer in LENS_PRESET_IDS');
});

test('cross-s2 is the first lens preset (index 0 in LENS_PRESET_IDS ordering)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx');
  // The LENS_PRESET_IDS array literal should start with cross-s2
  assert.match(src, /LENS_PRESET_IDS = \[\s*'cross-s2'/, 'cross-s2 must be first in LENS_PRESET_IDS');
});
