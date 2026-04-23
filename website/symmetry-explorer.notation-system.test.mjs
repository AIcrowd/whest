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
  assert.equal(notationColor('v_free'), '#292C2D');
  assert.equal(notationText('w_summed'), 'W_summed');
  assert.equal(notationLatex('w_summed'), 'W_{\\mathrm{summed}}');
  assert.equal(notationColor('w_summed'), recommendedTheme.roles.summedSide);
  assert.equal(notationColor('g_detected'), recommendedTheme.roles.symmetryObject);
  assert.equal(notationText('g_output'), 'G_out');
  assert.equal(notationLatex('g_output'), 'G_{\\mathrm{out}}');
  assert.equal(notationColor('g_output'), recommendedTheme.roles.freeSide);
  assert.equal(notationColor('sigma_row_move'), recommendedTheme.roles.action);
  assert.equal(notationColor('alpha_total'), recommendedTheme.roles.quantity);
  assert.equal(notationColor('m_incidence'), '#292C2D');
  assert.equal(notationColor('l_labels'), '#5D5F60');
  assert.equal(notationColor('g_v_factor'), notationColor('v_free'));
  assert.equal(notationColor('g_w_factor'), '#334155');
  assert.equal(notationColor('projection_pi_v_free'), '#292C2D');
  assert.equal(notationColor('c_omega_cycles'), '#292C2D');
  assert.equal(notationText('c_omega_cycles'), 'c_Ω(g)');
  assert.equal(notationLatex('c_omega_cycles'), 'c_\\Omega(g)');
  assert.equal(
    notationColoredLatex('g_wreath'),
    `\\textcolor{${notationColor('g_wreath')}}{${notationLatex('g_wreath')}}`,
  );
});

test('notation registry separates product count M from multiplication cost μ', () => {
  assert.equal(notationText('m_total'), 'M');
  assert.equal(notationLatex('m_total'), 'M');
  assert.equal(notationText('mu_total'), 'μ');
  assert.equal(notationLatex('mu_total'), '\\mu');
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
      'editorial-noir-math',
      'mean-prop-led',
      'cool-proof',
      'blue-ledger',
      'blue-margin',
      'warm-margin',
      'cov-prop-editorial',
    ],
  );
  assert.equal(EXPLORER_THEME_RECOMMENDED_ID, 'editorial-noir');
  assert.match(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-noir').summary,
    /darker editorial/i,
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
  assert.match(
    EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-noir-math').summary,
    /rich math palette/i,
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
  assert.equal(notationColor('alpha_total'), '#292C2D');

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
  assert.equal(notationColor('alpha_total'), '#292C2D');
});

test('editorial-noir-math remaps approved notation families through the rich math palette', () => {
  resetActiveExplorerTheme();
  setActiveExplorerTheme('editorial-noir-math');

  assert.equal(notationColor('v_free'), '#A45F44');
  assert.equal(notationColor('w_summed'), '#334155');
  assert.equal(notationColor('g_detected'), '#4A6288');
  assert.equal(notationColor('g_w_factor'), '#4A7E9A');
  assert.equal(notationColor('g_wreath'), '#557048');
  assert.equal(notationColor('x_space'), '#6B5C92');
  assert.equal(notationColor('orbit_o'), '#7F5F78');
  assert.equal(notationColor('projection_pi_v_free'), '#3C8D86');
  assert.equal(notationColor('pi_relabeling'), '#3C8D86');
  assert.equal(notationColor('sigma_row_move'), '#975B4C');
  assert.equal(notationColor('g_element'), '#6D8770');
  assert.equal(notationColor('mu_total'), '#326B79');
  assert.equal(notationColor('m_component'), '#3C8D86');
  assert.equal(notationColor('alpha_total'), '#8F647F');
  assert.equal(notationColor('k_operands'), '#B07C5F');
  assert.equal(notationColor('n_label'), '#A8904E');
  assert.equal(notationColor('n_cycle'), '#4D8A78');
  assert.equal(notationColor('n_l'), '#8C7B44');
  assert.equal(notationColor('n_omega'), '#4A7E9A');
  assert.equal(notationColor('c_omega_cycles'), '#975B4C');
  assert.equal(notationColor('l_labels'), '#557048');
  assert.equal(notationColor('u_axis_classes'), '#557048');
  assert.equal(notationColor('r_complement'), '#557048');
  assert.equal(notationColor('m_incidence'), '#6D8770');

  resetActiveExplorerTheme();
});

test('notationColor accepts an explicit theme override without changing global state', () => {
  resetActiveExplorerTheme();
  setActiveExplorerTheme('editorial-noir');
  const noirMath = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-noir-math');

  const globalAlpha = notationColor('alpha_total');
  const overriddenAlpha = notationColor('alpha_total', 'editorial-noir-math');
  const overriddenAlphaFromPreset = notationColor('alpha_total', noirMath);

  assert.equal(globalAlpha, '#292C2D');
  assert.equal(overriddenAlpha, '#8F647F');
  assert.equal(overriddenAlphaFromPreset, '#8F647F');
  assert.equal(getActiveExplorerThemeId(), 'editorial-noir');
  assert.equal(notationColor('alpha_total'), '#292C2D');
});

test('colorizeNotationLatex accepts an explicit theme override without changing global state', () => {
  resetActiveExplorerTheme();
  setActiveExplorerTheme('editorial-noir');
  const noirMath = EXPLORER_THEME_PRESETS.find((preset) => preset.id === 'editorial-noir-math');

  const colored = colorizeNotationLatex(String.raw`\mu + \alpha_a`, 'editorial-noir-math');
  const coloredFromPreset = colorizeNotationLatex(String.raw`\mu + \alpha_a`, noirMath);

  assert.match(colored, /#326B79/);
  assert.match(colored, /#8F647F/);
  assert.match(coloredFromPreset, /#326B79/);
  assert.match(coloredFromPreset, /#8F647F/);
  assert.equal(getActiveExplorerThemeId(), 'editorial-noir');
});

test('section five formulas route visible noir-math symbols through distinct notation lanes', () => {
  resetActiveExplorerTheme();
  setActiveExplorerTheme('editorial-noir-math');

  const totalCostSource = fs.readFileSync(
    path.join(WEBSITE_ROOT, 'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx'),
    'utf8',
  );
  assert.match(totalCostSource, /sumOver\(inSet\(tc\(SYM\.element, notationLatex\('g_element'\)\), tc\(SYM\.localGroup, notationLatex\('g_component'\)\)\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.ambient, notationLatex\('x_space'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.ambient, notationLatex\('orbit_space_component'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.projection, notationLatex\('projection_pi_v_free'\)\)/);
  assert.match(totalCostSource, /notationLatex\('r_complement'\)/);
  assert.match(totalCostSource, /notationLatex\('l_labels'\)/);
  assert.match(totalCostSource, /notationLatex\('g_w_factor'\)/);
  assert.match(totalCostSource, /notationLatex\('n_omega'\)/);
  assert.match(totalCostSource, /notationLatex\('c_omega_cycles'\)/);

  const singletonColored = colorizeNotationLatex(
    String.raw`\frac{n_\Omega}{|G_a|} \sum_g \Bigl(\prod_{c \in R} n_c\Bigr)\Bigl(n_\Omega^{c_\Omega(g)} - (n_\Omega - 1)^{c_\Omega(g)}\Bigr)`,
  );

  assert.match(totalCostSource, /tc\(SYM\.ambient, 'X'\)/);
  assert.match(totalCostSource, /tc\(SYM\.wlabel, notationLatex\('w_summed'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.summedGroup, notationLatex\('g_w_factor'\)\)/);

  assert.match(singletonColored, /#4A7E9A/);
  assert.match(singletonColored, /#4A6288/);
  assert.match(singletonColored, /#557048/);
  assert.match(singletonColored, /#4D8A78/);
  assert.match(singletonColored, /#975B4C/);

  assert.match(totalCostSource, /tc\(SYM\.orbitObject, notationLatex\('orbit_o'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.ambient, notationLatex\('x_space'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.localGroup, notationLatex\('g_component'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.projection, notationLatex\('projection_pi_v_free'\)\)/);
  assert.match(totalCostSource, /tc\(SYM\.vlabel, notationLatex\('v_free'\)\)/);

  resetActiveExplorerTheme();
});

test('editorial-noir-math keeps operator scaffolding neutral when auto-colorizing latex', () => {
  resetActiveExplorerTheme();
  setActiveExplorerTheme('editorial-noir-math');

  const colored = colorizeNotationLatex(String.raw`\mu = (k - 1) \prod_a M_a + \prod_a \alpha_a`);

  assert.match(colored, /\\textcolor\{#326B79\}\{\\mu\}/);
  assert.match(colored, /\\textcolor\{#3C8D86\}\{M_a\}/);
  assert.match(colored, /\\textcolor\{#8F647F\}\{\\alpha_a\}/);
  assert.doesNotMatch(colored, /\\textcolor\{[^}]+\}\{=\}/);

  resetActiveExplorerTheme();
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
  const appendixFormula = String.raw`G_{\text{f}} = G_{\mathrm{out}} \times S(W_{\mathrm{summed}})`;
  const preciseAppendixFormula = String.raw`G_{\mathrm{out}} = G_{\text{pt}}\big|_{V_{\mathrm{free}}}`;
  const youngFormula = String.raw`\alpha = n_L^{|V_{\mathrm{free}}|} \cdot \binom{n_L + |W_{\mathrm{summed}}| - 1}{|W_{\mathrm{summed}}|}`;
  const precoloredFormula = String.raw`\textcolor{#2F855A}{\prod_c n_c} + \textcolor{#C05621}{n_\Omega}`;
  const singletonFormula = String.raw`\alpha = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega - 1)^{c_\Omega(g)}\right)`;
  const matrixWitnessFormula = String.raw`M_\sigma = M`;
  const wreathMembership = String.raw`\sigma \in G_{\mathrm{wreath}}`;
  const wreathEquation = String.raw`G_{\mathrm{wreath}} = \prod_i (H_i \wr S_{m_i})`;
  const appendixExponentIssue = String.raw`t \in [n]^L`;

  const colorizedAppendix = colorizeNotationLatex(appendixFormula);
  const colorizedPreciseAppendix = colorizeNotationLatex(preciseAppendixFormula);
  const colorizedYoung = colorizeNotationLatex(youngFormula);
  const preservedPrecolored = colorizeNotationLatex(precoloredFormula);
  const colorizedSingleton = colorizeNotationLatex(singletonFormula);
  const colorizedMatrixWitness = colorizeNotationLatex(matrixWitnessFormula);
  const colorizedWreathMembership = colorizeNotationLatex(wreathMembership);
  const colorizedWreathEquation = colorizeNotationLatex(wreathEquation);
  const colorizedAppendixExponentIssue = colorizeNotationLatex(appendixExponentIssue);

  assert.equal(colorizedAppendix.includes(notationColoredLatex('g_formal')), true);
  assert.equal(colorizedAppendix.includes(notationColoredLatex('g_output')), true);
  assert.equal(colorizedAppendix.includes(notationColoredLatex('s_w_summed')), true);
  assert.equal(colorizedPreciseAppendix.includes(notationColoredLatex('g_output')), true);
  assert.equal(colorizedPreciseAppendix.includes(notationColoredLatex('g_pointwise_restricted_v')), true);

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
  assert.match(constructionSource, /notationLatex\('g_output'\)/);
  assert.match(constructionSource, /notationColoredLatex\('v_free', 'V'\)/);
  assert.match(constructionSource, /String\.raw`\\prod_d \$\{notationColoredLatex\('s_w_summed', 'S\(W_d\)'\)\}`/);
  assert.match(constructionSource, /is trivial for this einsum\./);
  assert.match(appendixSource, /Pointwise group/);
  assert.match(appendixSource, /anchorId="appendix-section-3"/);
  assert.match(appendixSource, /The restriction <Latex math=\{String\.raw`G_\{\\text\{pt\}\}\\|_V`\} \/> to output labels/);
  assert.match(appendixSource, /notationLatex\('g_output'\)/);
  assert.match(appendixSource, /notationColoredLatex\('s_w_summed', 'S\(W\)'\)/);
  assert.match(appendixSource, /appendixSection6\.title/);
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
