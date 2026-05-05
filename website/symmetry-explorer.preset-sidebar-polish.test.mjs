import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const SIDEBAR = 'components/symmetry-aware-einsum-contractions/components/PresetSidebar.jsx';

test('PresetSidebar renders a search/filter input', () => {
  const src = read(SIDEBAR);
  // The search input must exist with type="search" and an onChange handler
  // wired to a filterQuery state setter.
  assert.match(src, /type="search"/);
  assert.match(src, /placeholder="Search presets or tags/);
  assert.match(src, /setFilterQuery/);
  assert.match(src, /useState\(\s*''\s*\)/);
});

test('PresetSidebar search input carries an aria-label for accessibility', () => {
  const src = read(SIDEBAR);
  assert.match(src, /aria-label="Search presets by name or tag"/);
});

test('PresetSidebar renders tag pills with data-preset-tag attribute on each card', () => {
  const src = read(SIDEBAR);
  // The TagPill component must mark each rendered tag with data-preset-tag,
  // and the PRESET_TAGS map must define tags for the V3.1 lens presets.
  assert.match(src, /data-preset-tag=\{tag\}/);
  assert.match(src, /export const PRESET_TAGS/);
  assert.match(src, /'cross-s2':\s*\[[^\]]*'branching'[^\]]*\]/);
  assert.match(src, /'triple-outer':\s*\[[^\]]*'all-visible'[^\]]*\]/);
  assert.match(src, /'triangle':\s*\[[^\]]*'certification'[^\]]*\]/);
  assert.match(src, /'cross-c3-partial':\s*\[[^\]]*'partition'[^\]]*\]/);
  assert.match(src, /'bilinear-trace':\s*\[[^\]]*'formal'[^\]]*\]/);
});

test('Preset cards keep mini-preview metadata without rendering hover tooltips', () => {
  const src = read(SIDEBAR);
  // The signature text remains available as metadata, but the floating hover
  // tooltip was removed because it interfered with neighboring preset rows.
  assert.match(src, /data-mini-preview-text=\{previewText \?\? undefined\}/);
  assert.match(src, /export const PRESET_MINI_PREVIEW/);
  assert.match(src, /'cross-s2':[^,}]*'mini O→Q row with two filled cells'/);
  assert.match(src, /'triple-outer':[^,}]*'row count equals filled-cell count'/);
  assert.match(src, /'triangle':[^,}]*'accepted\/rejected sigma badges'/);
  assert.match(src, /'cross-c3-partial':[^,}]*'equality-pattern chips'/);
  assert.match(src, /'bilinear-trace':[^,}]*'formal-symmetry mismatch'/);
  assert.doesNotMatch(src, /role="tooltip"/);
  assert.doesNotMatch(src, /data-mini-preview="true"/);
});

test('PresetSidebar tracks the previous preset id with a ref + state', () => {
  const src = read(SIDEBAR);
  // The "previous preset" memory uses both a ref (stable across renders) and
  // a state setter (so the breadcrumb can re-render when it changes).
  assert.match(src, /prevPresetIdRef/);
  assert.match(src, /useRef\(null\)/);
  assert.match(src, /setPreviousPresetId/);
  assert.match(src, /lastSelectedIdxRef/);
  // The effect must compare current vs last selected idx and capture the prior.
  assert.match(src, /useEffect/);
  assert.match(src, /selectedPresetIdx !== lastSelectedIdxRef\.current/);
});

test('"Return to previous" breadcrumb renders conditionally when a prior preset exists', () => {
  const src = read(SIDEBAR);
  // The breadcrumb is only rendered when previousPresetEntry is non-null,
  // and the button is marked with data-preset-return for test/QA hooks.
  assert.match(src, /previousPresetEntry \?/);
  assert.match(src, /data-preset-return="true"/);
  assert.match(src, /Return to \{previousPresetEntry\.summary\.name\}/);
  // Clicking the breadcrumb must call onSelect with the prior idx.
  assert.match(src, /onSelect\(previousPresetEntry\.idx\)/);
});

test('PresetSidebar exposes tag + mini-preview maps for downstream callers', () => {
  const src = read(SIDEBAR);
  // Both maps are exported (named exports) so other components — e.g. §2
  // narrative spotlight links — can consult them without duplicating.
  assert.match(src, /export const PRESET_TAGS/);
  assert.match(src, /export const PRESET_MINI_PREVIEW/);
});

test('All five V3.1 lens presets receive at least one tag and a mini-preview signature', () => {
  const src = read(SIDEBAR);
  for (const id of ['cross-s2', 'triple-outer', 'triangle', 'cross-c3-partial', 'bilinear-trace']) {
    // Each id appears as a key in both maps.
    const tagPattern = new RegExp(`'${id}':\\s*\\[`);
    const previewPattern = new RegExp(`'${id}':\\s*'`);
    assert.match(src, tagPattern, `${id} missing in PRESET_TAGS`);
    assert.match(src, previewPattern, `${id} missing in PRESET_MINI_PREVIEW`);
  }
});

test('Reference presets do not show generic TBD mini-preview copy', () => {
  const src = read(SIDEBAR);
  assert.match(src, /PRESET_MINI_PREVIEW\[summary\.id\] \?\? null/);
  assert.match(src, /data-mini-preview-text=\{previewText \?\? undefined\}/);
  assert.doesNotMatch(src, /setHoveredIdx/);
  assert.doesNotMatch(src, /visible=\{hasPreview && isHovered\}/);
  assert.doesNotMatch(src, /pedagogical signature TBD/i);
  assert.doesNotMatch(src, /GENERIC_MINI_PREVIEW/);
});
