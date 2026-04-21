import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  explorerThemeColor,
  EXPLORER_THEME_PRESETS,
  EXPLORER_THEME_RECOMMENDED_ID,
  getActiveExplorerThemeId,
  getActiveExplorerThemeRoles,
  getExplorerThemeCssVariables,
  getExplorerThemePreset,
  resetActiveExplorerTheme,
  setActiveExplorerTheme,
  subscribeActiveExplorerTheme,
} from './components/symmetry-aware-einsum-contractions/lib/explorerTheme.js';

function read(relativePath) {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf8');
}

test('explorer theme registry exposes the approved presets', () => {
  assert.deepEqual(
    EXPLORER_THEME_PRESETS.map((preset) => preset.id),
    [
      'strict-editorial',
      'editorial-balance',
      'editorial-balance-slate',
      'editorial-balance-warm',
      'teaching-calm',
      'quiet-ledger',
      'soft-coral',
      'quiet-info',
      'deep-info-ledger',
      'warm-exception',
      'muted-amber',
      'whestbench-axis',
      'whestbench-axis-blue',
      'whestbench-sampling',
      'whestbench-cov-prop',
      'whestbench-diverging',
      'whestbench-verdict',
      'whestbench-scorecard',
      'whestbench-sage',
      'coral-slate-contrast',
      'coral-slate-split',
      'coral-slate-hardline',
      'ink-authority',
      'editorial-noir',
      'mean-prop-led',
      'cool-proof',
      'blue-ledger',
      'blue-margin',
      'warm-margin',
      'cov-prop-editorial',
    ],
  );
  assert.equal(EXPLORER_THEME_RECOMMENDED_ID, 'editorial-balance');

  const editorialBalance = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance');
  const editorialBalanceSlate = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance-slate');
  const editorialBalanceWarm = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance-warm');
  const teachingCalm = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'teaching-calm');
  const whestbenchSampling = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-sampling');
  const whestbenchCovProp = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-cov-prop');
  const whestbenchVerdict = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-verdict');
  const whestbenchSage = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-sage');
  const whestbenchAxisBlue = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-axis-blue');
  const coralSlateContrast = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'coral-slate-contrast');
  const coralSlateHardline = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'coral-slate-hardline');
  const inkAuthority = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'ink-authority');
  const meanPropLed = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'mean-prop-led');
  const blueLedger = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'blue-ledger');
  const blueMargin = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'blue-margin');
  const warmMargin = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'warm-margin');
  const covPropEditorial = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'cov-prop-editorial');
  assert.ok(editorialBalance);
  assert.ok(editorialBalanceSlate);
  assert.ok(editorialBalanceWarm);
  assert.ok(teachingCalm);
  assert.ok(whestbenchSampling);
  assert.ok(whestbenchCovProp);
  assert.ok(whestbenchVerdict);
  assert.ok(whestbenchSage);
  assert.ok(whestbenchAxisBlue);
  assert.ok(coralSlateContrast);
  assert.ok(coralSlateHardline);
  assert.ok(inkAuthority);
  assert.ok(meanPropLed);
  assert.ok(blueLedger);
  assert.ok(blueMargin);
  assert.ok(warmMargin);
  assert.ok(covPropEditorial);
  assert.equal(editorialBalance.roles.freeSide, '#F0524D');
  assert.equal(editorialBalance.roles.ink, '#292C2D');
  assert.equal(editorialBalance.roles.quantity, '#0B6D7A');
  assert.equal(editorialBalanceSlate.roles.quantity, '#334155');
  assert.equal(editorialBalanceWarm.roles.quantity, '#B29F9E');
  assert.equal(EXPLORER_THEME_PRESETS[0].roles.symmetryObject, '#334155');
  assert.equal(EXPLORER_THEME_PRESETS[0].roles.quantity, '#292C2D');
  assert.equal(teachingCalm.roles.symmetryObject, '#2959C4');
  assert.equal(teachingCalm.roles.action, '#FA9E33');
  assert.equal(whestbenchSampling.roles.quantity, '#D23934');
  assert.equal(whestbenchCovProp.roles.action, '#B29F9E');
  assert.equal(whestbenchAxisBlue.roles.summedSide, '#334155');
  assert.equal(whestbenchAxisBlue.roles.action, '#B29F9E');
  assert.equal(whestbenchAxisBlue.roles.quantity, '#334155');
  assert.equal(whestbenchVerdict.roles.quantity, '#B29F9E');
  assert.equal(whestbenchSage.roles.quantity, '#94A3B8');
  assert.equal(whestbenchVerdict.roles.statusSuccess, '#B29F9E');
  assert.equal(coralSlateContrast.roles.quantity, '#334155');
  assert.equal(coralSlateHardline.roles.quantity, '#292C2D');
  assert.equal(inkAuthority.roles.symmetryObject, '#292C2D');
  assert.equal(meanPropLed.roles.action, '#2959C4');
  assert.equal(blueLedger.roles.quantity, '#2959C4');
  assert.equal(blueMargin.roles.symmetryObject, '#2959C4');
  assert.equal(blueMargin.roles.quantity, '#334155');
  assert.equal(blueMargin.roles.editorialAccent, '#B29F9E');
  assert.equal(warmMargin.roles.quantity, '#B29F9E');
  assert.equal(covPropEditorial.roles.statusSuccess, '#B29F9E');
  assert.equal(getExplorerThemePreset('missing-id').id, 'editorial-balance');
});

test('explorerThemeColor resolves shared role colors from ids or theme objects', () => {
  const teachingCalm = getExplorerThemePreset('teaching-calm');

  assert.equal(explorerThemeColor('strict-editorial', 'summedSide'), '#334155');
  assert.equal(explorerThemeColor(teachingCalm, 'symmetryObject'), '#2959C4');
  assert.equal(explorerThemeColor(teachingCalm, 'quantity'), '#0B6D7A');
  assert.equal(
    explorerThemeColor('missing-id', 'hero'),
    getExplorerThemePreset(EXPLORER_THEME_RECOMMENDED_ID).roles.hero,
  );
});

test('explorerTheme owns the active theme store', () => {
  const seen = [];
  const unsubscribe = subscribeActiveExplorerTheme(() => {
    seen.push(getActiveExplorerThemeId());
  });

  resetActiveExplorerTheme();
  assert.equal(getActiveExplorerThemeId(), EXPLORER_THEME_RECOMMENDED_ID);

  setActiveExplorerTheme('teaching-calm');
  assert.equal(getActiveExplorerThemeId(), 'teaching-calm');
  assert.equal(
    getActiveExplorerThemeRoles().symmetryObject,
    getExplorerThemePreset('teaching-calm').roles.symmetryObject,
  );

  unsubscribe();
  resetActiveExplorerTheme();
  assert.deepEqual(seen, ['teaching-calm']);
  assert.equal(getActiveExplorerThemeId(), EXPLORER_THEME_RECOMMENDED_ID);
});

test('explorer theme CSS variables expose status tokens for the scoped alias layer', () => {
  const teachingCalm = getExplorerThemePreset('teaching-calm');
  const cssVars = getExplorerThemeCssVariables(teachingCalm);

  assert.equal(cssVars['--status-success'], teachingCalm.roles.statusSuccess);
  assert.equal(cssVars['--status-warning'], teachingCalm.roles.statusWarning);
  assert.equal(cssVars['--success'], teachingCalm.roles.quantity);
  assert.equal(cssVars['--warning'], teachingCalm.roles.action);
  assert.equal(cssVars['--editorial-accent'], teachingCalm.roles.editorialAccent);
});

test('styles.css defines the explorer-scoped role variable aliases', () => {
  const appSrc = read('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const stylesSrc = read('./components/symmetry-aware-einsum-contractions/styles.css');
  const aliasBlock = stylesSrc.match(/\.symmetry-aware-einsum-explorer\s*\{[\s\S]*?\n\}/)?.[0] ?? '';

  assert.match(appSrc, /className="[^"]*\bsymmetry-aware-einsum-explorer\b[^"]*"/);
  assert.match(aliasBlock, /--explorer-hero:\s*var\(--coral\);/);
  assert.match(aliasBlock, /--explorer-hero-muted:\s*var\(--coral-hover\);/);
  assert.match(aliasBlock, /--explorer-ink:\s*var\(--foreground\);/);
  assert.match(aliasBlock, /--explorer-muted:\s*var\(--muted-foreground\);/);
  assert.match(aliasBlock, /--explorer-border:\s*var\(--border\);/);
  assert.match(aliasBlock, /--explorer-surface:\s*var\(--card\);/);
  assert.match(aliasBlock, /--explorer-surface-inset:\s*var\(--muted\);/);
  assert.match(aliasBlock, /--explorer-free-side:\s*var\(--ein-v\);/);
  assert.match(aliasBlock, /--explorer-summed-side:\s*var\(--ein-w\);/);
  assert.match(aliasBlock, /--explorer-symmetry-object:\s*var\(--info\);/);
  assert.match(aliasBlock, /--explorer-action:\s*var\(--warning\);/);
  assert.match(aliasBlock, /--explorer-quantity:\s*var\(--success\);/);
  assert.match(aliasBlock, /--explorer-status-success:\s*var\(--status-success\);/);
  assert.match(aliasBlock, /--explorer-status-warning:\s*var\(--status-warning\);/);
  assert.match(aliasBlock, /--explorer-editorial-accent:\s*var\(--editorial-accent\);/);
});
