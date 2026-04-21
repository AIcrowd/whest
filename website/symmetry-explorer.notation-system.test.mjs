import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import katex from 'katex';

import {
  NOTATION_HOST_FILES,
  NOTATION_REGISTRY,
  colorizeNotationLatex,
  getActiveExplorerThemeId,
  getActiveExplorerThemeRoles,
  getActiveNotationGrammarId,
  resetActiveExplorerTheme,
  resetActiveNotationPalette,
  setActiveNotationGrammar,
  setActiveExplorerTheme,
  notationColor,
  notationColoredLatex,
  notationLatex,
  notationText,
} from './components/symmetry-aware-einsum-contractions/lib/notationSystem.js';
import {
  EXPLORER_THEME_PRESETS,
  EXPLORER_THEME_RECOMMENDED_ID,
} from './components/symmetry-aware-einsum-contractions/lib/explorerTheme.js';
import {
  NOTATION_GRAMMAR_PRESETS,
  NOTATION_GRAMMAR_RECOMMENDED_ID,
  getNotationGrammarPreset,
} from './components/symmetry-aware-einsum-contractions/lib/notationGrammar.js';
import { EXPLORER_ACTS } from './components/symmetry-aware-einsum-contractions/components/explorerNarrative.js';

const WEBSITE_ROOT = path.resolve(
  path.dirname(new URL(import.meta.url).pathname),
);

const REQUIRED_HOSTS = [
  'components/symmetry-aware-einsum-contractions/components/explorerNarrative.js',
  'components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx',
  'components/symmetry-aware-einsum-contractions/components/ComponentView.jsx',
  'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx',
  'components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx',
  'components/symmetry-aware-einsum-contractions/components/DiminoView.jsx',
  'components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx',
  'components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx',
  'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx',
  'components/symmetry-aware-einsum-contractions/components/VSubSwConstruction.jsx',
  'components/symmetry-aware-einsum-contractions/components/WreathStructureView.jsx',
  'components/symmetry-aware-einsum-contractions/engine/regimeSpec.js',
  'components/symmetry-aware-einsum-contractions/engine/shapeSpec.js',
];

const LEGACY_NOTATION_PATTERNS = [
  /V \(free\)/,
  /W \(summed\)/,
  /free label \(V\)/,
  /summed label \(W\)/,
  /\\pi_V\(O\)/,
];

test('notation registry defines text, latex, and the semantic grammar anchors', () => {
  const entries = Object.entries(NOTATION_REGISTRY);
  const recommendedTheme = EXPLORER_THEME_PRESETS.find(
    (preset) => preset.id === EXPLORER_THEME_RECOMMENDED_ID,
  );
  assert.ok(entries.length > 12, 'expected a real notation inventory');

  for (const [id, entry] of entries) {
    assert.equal(typeof entry.text, 'string', `${id} needs text`);
    assert.equal(typeof entry.latex, 'string', `${id} needs latex`);
    assert.match(entry.color, /^#[0-9A-F]{6}$/i, `${id} needs hex color`);
  }

  assert.equal(notationText('v_free'), 'V_free');
  assert.equal(notationLatex('v_free'), 'V_{\\mathrm{free}}');
  assert.equal(notationColor('v_free'), recommendedTheme.roles.freeSide);
  assert.equal(notationText('w_summed'), 'W_summed');
  assert.equal(notationLatex('w_summed'), 'W_{\\mathrm{summed}}');
  assert.equal(notationColor('w_summed'), recommendedTheme.roles.summedSide);
  assert.equal(notationColor('g_detected'), recommendedTheme.roles.symmetryObject);
  assert.equal(notationColor('sigma_row_move'), recommendedTheme.roles.action);
  assert.equal(notationColor('alpha_total'), recommendedTheme.roles.quantity);
  assert.equal(notationColor('m_incidence'), '#292C2D');
  assert.equal(notationColor('l_labels'), '#5D5F60');
  assert.equal(notationColor('g_v_factor'), notationColor('v_free'));
  assert.equal(notationColor('g_w_factor'), notationColor('w_summed'));
  assert.equal(notationColor('projection_pi_v_free'), notationColor('v_free'));
  assert.equal(notationColor('c_omega_cycles'), recommendedTheme.roles.quantity);
  assert.equal(notationText('c_omega_cycles'), 'c_Ω(g)');
  assert.equal(notationLatex('c_omega_cycles'), 'c_\\Omega(g)');
  assert.equal(
    notationColoredLatex('g_wreath'),
    `\\textcolor{${notationColor('g_wreath')}}{${notationLatex('g_wreath')}}`,
  );
});

test('explorer themes expose the approved page-wide alternatives with a stable recommendation', () => {
  const presetIds = EXPLORER_THEME_PRESETS.map((preset) => preset.id);

  assert.deepEqual(
    presetIds,
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
  assert.match(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance').summary,
    /recommended balance/i,
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance-slate').roles.quantity,
    '#334155',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-balance-warm').roles.quantity,
    '#B29F9E',
  );
  assert.match(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-axis').summary,
    /whestbench coral↔slate axis/i,
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-axis-blue').roles.action,
    '#B29F9E',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-cov-prop').roles.quantity,
    '#B29F9E',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-scorecard').roles.quantity,
    '#D23934',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'whestbench-sage').roles.statusSuccess,
    '#94A3B8',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'coral-slate-contrast').roles.symmetryObject,
    '#334155',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-noir').roles.quantity,
    '#292C2D',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'cool-proof').roles.symmetryObject,
    '#2959C4',
  );
  assert.equal(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'blue-ledger').roles.statusSuccess,
    '#2959C4',
  );
});

test('legacy notation grammar presets are a compatibility shim over explorer themes', () => {
  assert.equal(NOTATION_GRAMMAR_RECOMMENDED_ID, 'split-cost');
  assert.ok(NOTATION_GRAMMAR_PRESETS.length > 0);

  for (const preset of NOTATION_GRAMMAR_PRESETS) {
    assert.equal(typeof preset.explorerThemeId, 'string');
    assert.ok(
      EXPLORER_THEME_PRESETS.some((theme) => theme.id === preset.explorerThemeId),
      `${preset.id} should map to a live explorer theme`,
    );
    assert.deepEqual(
      preset.palette ?? {},
      {},
      `${preset.id} should not carry authority palettes anymore`,
    );
  }

  assert.equal(getNotationGrammarPreset('missing-id').id, 'current');
});

test('active explorer theme overrides notation colors globally until reset', () => {
  const teachingCalm = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'teaching-calm');
  const recommendedTheme = EXPLORER_THEME_PRESETS.find(
    (preset) => preset.id === EXPLORER_THEME_RECOMMENDED_ID,
  );

  resetActiveExplorerTheme();
  assert.equal(getActiveExplorerThemeId(), EXPLORER_THEME_RECOMMENDED_ID);
  assert.equal(getActiveExplorerThemeRoles().hero, recommendedTheme.roles.hero);
  assert.equal(notationColor('alpha_total'), recommendedTheme.roles.quantity);

  setActiveExplorerTheme(teachingCalm.id, teachingCalm.roles);
  assert.equal(getActiveExplorerThemeId(), 'teaching-calm');
  assert.equal(getActiveExplorerThemeRoles().symmetryObject, teachingCalm.roles.symmetryObject);
  assert.equal(notationColor('v_free'), teachingCalm.roles.freeSide);
  assert.equal(notationColor('w_summed'), teachingCalm.roles.summedSide);
  assert.equal(notationColor('g_component'), teachingCalm.roles.symmetryObject);
  assert.equal(notationColor('g_element'), teachingCalm.roles.action);
  assert.equal(notationColor('m_component'), teachingCalm.roles.quantity);
  assert.equal(notationColor('alpha_total'), teachingCalm.roles.quantity);
  assert.equal(
    colorizeNotationLatex(String.raw`\mu = (k-1)\prod_a M_a`).includes(teachingCalm.roles.quantity),
    true,
  );

  resetActiveExplorerTheme();
  assert.equal(getActiveExplorerThemeId(), EXPLORER_THEME_RECOMMENDED_ID);
  assert.equal(notationColor('alpha_total'), recommendedTheme.roles.quantity);
});

test('legacy notation grammar setters are compatibility-only and do not override explorer theme colors', () => {
  const teachingCalm = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'teaching-calm');

  resetActiveExplorerTheme();
  resetActiveNotationPalette();
  setActiveExplorerTheme(teachingCalm.id, teachingCalm.roles);

  const before = notationColor('alpha_total');
  setActiveNotationGrammar('split-cost', { alpha_total: '#111111' });

  assert.equal(getActiveNotationGrammarId(), 'split-cost');
  assert.equal(notationColor('alpha_total'), before);

  resetActiveNotationPalette();
  assert.equal(getActiveNotationGrammarId(), 'current');
  assert.equal(notationColor('alpha_total'), before);
});

test('shared latex colorizer colors representative raw formulas before render', () => {
  const appendixFormula = String.raw`G_{\text{f}} = G_{\text{pt}}\big|_{V_{\mathrm{free}}} \times S(W_{\mathrm{summed}})`;
  const youngFormula = String.raw`\alpha = n_L^{|V_{\mathrm{free}}|} \cdot \binom{n_L + |W_{\mathrm{summed}}| - 1}{|W_{\mathrm{summed}}|}`;
  const precoloredFormula = String.raw`\textcolor{#2F855A}{\prod_c n_c} + \textcolor{#C05621}{n_\Omega}`;
  const singletonFormula = String.raw`\alpha = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega - 1)^{c_\Omega(g)}\right)`;
  const matrixWitnessFormula = String.raw`M_\sigma = M`;
  const wreathMembership = String.raw`\sigma \in G_{\mathrm{wreath}}`;
  const wreathEquation = String.raw`G_{\mathrm{wreath}} = \prod_i (H_i \wr S_{m_i})`;
  const appendixExponentIssue = String.raw`t \in [n]^L`;

  const colorizedAppendix = colorizeNotationLatex(appendixFormula);
  const colorizedYoung = colorizeNotationLatex(youngFormula);
  const preservedPrecolored = colorizeNotationLatex(precoloredFormula);
  const colorizedSingleton = colorizeNotationLatex(singletonFormula);
  const colorizedMatrixWitness = colorizeNotationLatex(matrixWitnessFormula);
  const colorizedWreathMembership = colorizeNotationLatex(wreathMembership);
  const colorizedWreathEquation = colorizeNotationLatex(wreathEquation);
  const colorizedAppendixExponentIssue = colorizeNotationLatex(appendixExponentIssue);

  assert.equal(colorizedAppendix.includes(notationColoredLatex('g_formal')), true);
  assert.equal(colorizedAppendix.includes(notationColoredLatex('g_pointwise_restricted_v')), true);
  assert.equal(colorizedAppendix.includes(notationColoredLatex('s_w_summed')), true);

  assert.equal(colorizedYoung.includes(notationColoredLatex('alpha_total')), true);
  assert.equal(colorizedYoung.includes(notationColoredLatex('v_free')), true);
  assert.equal(colorizedYoung.includes(notationColoredLatex('w_summed')), true);
  assert.equal(colorizedSingleton.includes(notationColoredLatex('c_omega_cycles')), true);
  assert.equal(preservedPrecolored, precoloredFormula);
  assert.doesNotMatch(colorizedSingleton, /c_\\textcolor/);
  assert.doesNotMatch(colorizedMatrixWitness, /M_\\textcolor/);
  assert.equal(colorizedWreathMembership.includes(String.raw`\in`), true);
  assert.equal(colorizedWreathMembership.includes(notationColoredLatex('sigma_row_move')), true);
  assert.equal(colorizedWreathMembership.includes(notationColoredLatex('g_wreath')), true);
  assert.doesNotMatch(colorizedWreathMembership, /\\textcolor\{#[0-9A-Fa-f]{6}\}\{\\sigma \\in/);
  assert.equal(colorizedWreathEquation.includes(notationColoredLatex('g_wreath')), true);
  assert.equal(colorizedWreathEquation.includes(notationColoredLatex('h_family')), true);
  assert.doesNotMatch(colorizedWreathEquation, /\\textcolor\{#[0-9A-Fa-f]{6}\}\{G_\{\\mathrm\{wreath\}\}\s*=/);
  assert.equal(
    colorizedAppendixExponentIssue,
    `t \\in [n]^{\\textcolor{${notationColor('l_labels')}}{L}}`,
  );
  assert.equal(
    katex.renderToString(colorizedAppendixExponentIssue, { throwOnError: false, trust: true }).includes('katex-error'),
    false,
  );
  assert.equal(
    katex.renderToString(colorizedSingleton, { throwOnError: false, trust: true }).includes('katex-error'),
    false,
  );
  assert.equal(
    katex.renderToString(colorizedMatrixWitness, { throwOnError: false, trust: true }).includes('katex-error'),
    false,
  );
});

test('notation host inventory covers the main notation-bearing explorer surfaces', () => {
  assert.deepEqual(
    REQUIRED_HOSTS.filter((host) => !NOTATION_HOST_FILES.includes(host)),
    [],
  );

  for (const relativePath of NOTATION_HOST_FILES) {
    const absolutePath = path.join(WEBSITE_ROOT, relativePath);
    assert.equal(fs.existsSync(absolutePath), true, `${relativePath} missing from inventory`);
  }
});

test('notation system no longer imports the legacy notation grammar registry', () => {
  const notationSystemSource = fs.readFileSync(
    path.join(
      WEBSITE_ROOT,
      'components/symmetry-aware-einsum-contractions/lib/notationSystem.js',
    ),
    'utf8',
  );

  assert.doesNotMatch(notationSystemSource, /from '\.\/notationGrammar\.js'/);
  assert.doesNotMatch(notationSystemSource, /getNotationGrammarPreset/);
  assert.doesNotMatch(notationSystemSource, /let activeExplorerThemeId =/);
  assert.doesNotMatch(notationSystemSource, /let activeExplorerThemeRoles =/);
  assert.doesNotMatch(notationSystemSource, /notationColorWithPalette/);
  assert.doesNotMatch(notationSystemSource, /notationColoredLatexWithPalette/);
  assert.doesNotMatch(notationSystemSource, /colorizeNotationLatexWithPalette/);
});

test('notation host files do not use legacy V/W shorthand or raw notation colors', () => {
  const notationColors = new Set(Object.values(NOTATION_REGISTRY).map((entry) => entry.color));

  for (const relativePath of NOTATION_HOST_FILES) {
    const source = fs.readFileSync(path.join(WEBSITE_ROOT, relativePath), 'utf8');

    for (const pattern of LEGACY_NOTATION_PATTERNS) {
      assert.doesNotMatch(source, pattern, `${relativePath} still contains legacy notation`);
    }

    for (const color of notationColors) {
      assert.equal(
        source.includes(color),
        false,
        `${relativePath} still hardcodes notation color ${color}`,
      );
    }
  }
});

test('representative surfaces render long-form notation across narrative, price savings, and appendix math', () => {
  const narrativeText = EXPLORER_ACTS.flatMap((act) => act.introParagraphs ?? []).join(' ');
  const priceSavingsSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx'),
    'utf8',
  );
  const appendixSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx'),
    'utf8',
  );
  const constructionSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/VSubSwConstruction.jsx'),
    'utf8',
  );
  const regimeSpecSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/engine/regimeSpec.js'),
    'utf8',
  );

  assert.match(narrativeText, /V_\{\\mathrm\{free\}\}/);
  assert.match(narrativeText, /W_\{\\mathrm\{summed\}\}/);
  assert.match(narrativeText, /G_\{\\mathrm\{wreath\}\}/);
  assert.match(narrativeText, /H_i \\wr S_\{m_i\}/);
  assert.match(priceSavingsSource, /notationLatex\('v_free'\)/);
  assert.match(priceSavingsSource, /notationLatex\('w_summed'\)/);
  assert.match(priceSavingsSource, /notationLatex\('g_component'\)/);
  assert.match(constructionSource, /Formal-group construction/);
  assert.match(constructionSource, /notationLatex\('g_formal'\)/);
  assert.match(constructionSource, /notationLatex\('g_pointwise_restricted_v'\)/);
  assert.match(constructionSource, /notationLatex\('s_w_summed'\)/);
  assert.match(constructionSource, /is trivial for this einsum\./);
  assert.match(appendixSource, /The table below records the additional savings available when output storage respects those/);
  assert.match(appendixSource, /G_\{\\\\text\{pt\}\}\\\\big\|_\{V_\{\\\\mathrm\{free\}\}\}/);
  assert.match(regimeSpecSource, /\$\$\{notationLatex\('v_free'\)\}\$/);
  assert.match(regimeSpecSource, /\\mathrm\{Sym\}\(\$\{notationLatex\('w_summed'\)\}\)/);
  assert.match(regimeSpecSource, /notationLatex\('c_omega_cycles'\)/);
});

test('shared render paths colorize formulas and notation-bearing descriptions', () => {
  const latexSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/Latex.jsx'),
    'utf8',
  );
  const caseBadgeSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/CaseBadge.jsx'),
    'utf8',
  );
  const componentCostSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx'),
    'utf8',
  );
  const decisionLadderSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx'),
    'utf8',
  );

  assert.match(latexSource, /colorizeNotationLatex/);
  assert.match(latexSource, /getActiveExplorerThemeId/);
  assert.match(caseBadgeSource, /InlineMathText/);
  assert.match(caseBadgeSource, /tooltip\.body/);
  assert.match(componentCostSource, /<InlineMathText>\{text\}<\/InlineMathText>/);
  assert.match(decisionLadderSource, /notationTint\('g_detected',\s*0\.42\)/);
  assert.match(decisionLadderSource, /notationTint\('g_detected',\s*0\.1\)/);
  assert.doesNotMatch(decisionLadderSource, /border-violet-400/);
  assert.doesNotMatch(decisionLadderSource, /bg-violet-50/);
  assert.doesNotMatch(decisionLadderSource, /text-violet-700/);
});

test('editorial math copy leaves operators neutral and lets Latex auto-color individual symbols', () => {
  const narrativeSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/explorerNarrative.js'),
    'utf8',
  );
  const wreathSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/WreathStructureView.jsx'),
    'utf8',
  );

  assert.doesNotMatch(narrativeSource, /notationColoredLatex\('g_wreath',/);
  assert.doesNotMatch(narrativeSource, /notationColoredLatex\('sigma_row_move',/);
  assert.doesNotMatch(narrativeSource, /notationColoredLatex\('l_labels',/);
  assert.doesNotMatch(narrativeSource, /notationColoredLatex\('l_component',/);
  assert.doesNotMatch(wreathSource, /notationColoredLatex\('g_wreath',/);
});

test('inline math prose renderer supports bold emphasis markers used in appendix copy', () => {
  const inlineMathSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/InlineMathText.jsx'),
    'utf8',
  );
  assert.match(inlineMathSource, /font-semibold text-current/);
  assert.match(inlineMathSource, /segment\.startsWith\('\*\*'\)/);
  assert.match(inlineMathSource, /segment\.endsWith\('\*\*'\)/);
});
