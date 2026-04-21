import test from 'node:test';
import assert from 'node:assert/strict';

import {
  EXPLORER_THEME_PRESETS,
  EXPLORER_THEME_RECOMMENDED_ID,
  getExplorerThemePreset,
} from './components/symmetry-aware-einsum-contractions/lib/explorerTheme.js';

test('explorer theme registry exposes the approved presets', () => {
  assert.deepEqual(
    EXPLORER_THEME_PRESETS.map((preset) => preset.id),
    ['strict-editorial', 'editorial-balance', 'teaching-calm'],
  );
  assert.equal(EXPLORER_THEME_RECOMMENDED_ID, 'editorial-balance');

  const editorialBalance = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance');
  assert.ok(editorialBalance);
  assert.equal(editorialBalance.roles.freeSide, '#F0524D');
  assert.equal(editorialBalance.roles.ink, '#292C2D');
  assert.equal(editorialBalance.roles.quantity, '#0B6D7A');
  assert.equal(EXPLORER_THEME_PRESETS[0].roles.symmetryObject, '#334155');
  assert.equal(EXPLORER_THEME_PRESETS[0].roles.quantity, '#292C2D');
  assert.equal(EXPLORER_THEME_PRESETS[2].roles.symmetryObject, '#2959C4');
  assert.equal(EXPLORER_THEME_PRESETS[2].roles.action, '#FA9E33');
  assert.equal(getExplorerThemePreset('missing-id').id, 'editorial-balance');
});
