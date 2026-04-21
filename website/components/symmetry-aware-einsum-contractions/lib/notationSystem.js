import {
  getActiveExplorerThemeId as getThemeStoreActiveExplorerThemeId,
  getActiveExplorerThemeRoles as getThemeStoreActiveExplorerThemeRoles,
  getExplorerThemePreset,
  resetActiveExplorerTheme as resetThemeStoreActiveExplorerTheme,
  setActiveExplorerTheme as setThemeStoreActiveExplorerTheme,
} from './explorerTheme.js';
import {
  EDITORIAL_NOIR_RICH_MATH_PALETTE_ID,
  EDITORIAL_NOIR_RICH_MATH_ROLES,
} from './editorialNoirMathPalette.js';

const makeEntry = (text, latex, color) => ({ text, latex, color });

// Baseline notation colors retained for structural tokens and as fallbacks when
// a symbol is not mapped onto an explorer-theme role. Explorer themes own the
// active page-wide color grammar.
const NOTATION_COLORS = {
  free: '#F0524D',
  summed: '#64748B',
  object: '#4A7CFF',
  action: '#FA9E33',
  quantity: '#23B761',
  structure: '#292C2D',
  structureMuted: '#5D5F60',
};

let activeNotationGrammarId = 'current';

const NOTATION_THEME_ROLE_BY_ID = {
  v_free: 'freeSide',
  v_free_component: 'freeSide',
  g_pointwise_restricted_v: 'freeSide',
  g_v_factor: 'freeSide',
  projection_pi_v_free: 'freeSide',
  w_summed: 'summedSide',
  w_summed_component: 'summedSide',
  s_w_summed: 'summedSide',
  g_w_factor: 'summedSide',
  x_w_summed: 'summedSide',
  g_wreath: 'symmetryObject',
  h_family: 'symmetryObject',
  sym_l: 'symmetryObject',
  g_detected: 'symmetryObject',
  g_component: 'symmetryObject',
  g_formal: 'symmetryObject',
  g_pointwise: 'symmetryObject',
  x_space: 'symmetryObject',
  x_component: 'symmetryObject',
  orbit_space_component: 'symmetryObject',
  orbit_o: 'symmetryObject',
  omega_orbit: 'symmetryObject',
  sigma_row_move: 'action',
  pi_relabeling: 'action',
  g_element: 'action',
  mu_total: 'quantity',
  alpha_total: 'quantity',
  alpha_component: 'quantity',
  m_total: 'quantity',
  m_component: 'quantity',
  k_operands: 'quantity',
  n_label: 'quantity',
  n_cycle: 'quantity',
  n_l: 'quantity',
  n_omega: 'quantity',
  c_omega_cycles: 'quantity',
};

const NOTATION_MATH_ROLE_BY_ID = {
  v_free: 'freeSide',
  v_free_component: 'freeSide',
  g_pointwise_restricted_v: 'freeSide',
  g_v_factor: 'freeSide',
  projection_pi_v_free: 'freeSide',
  w_summed: 'summedSide',
  w_summed_component: 'summedSide',
  s_w_summed: 'summedSide',
  g_w_factor: 'summedSide',
  x_w_summed: 'summedSide',
  g_detected: 'symmetryObject',
  g_component: 'symmetryObject',
  g_pointwise: 'symmetryObject',
  x_space: 'ambientSpace',
  x_component: 'ambientSpace',
  orbit_space_component: 'ambientSpace',
  orbit_o: 'ambientSpace',
  omega_orbit: 'ambientSpace',
  g_wreath: 'subgroup',
  h_family: 'subgroup',
  g_formal: 'subgroup',
  sym_l: 'subgroup',
  sigma_row_move: 'actionSigma',
  pi_relabeling: 'projection',
  g_element: 'actionElement',
  mu_total: 'muFamily',
  m_total: 'muFamily',
  m_component: 'mFamily',
  alpha_total: 'alphaFamily',
  alpha_component: 'alphaFamily',
  k_operands: 'countFamily',
  n_label: 'countFamily',
  n_cycle: 'countFamily',
  n_l: 'countFamily',
  n_omega: 'countFamily',
  c_omega_cycles: 'countFamily',
};

export const NOTATION_REGISTRY = {
  v_free: makeEntry('V_free', String.raw`V_{\mathrm{free}}`, NOTATION_COLORS.free),
  w_summed: makeEntry('W_summed', String.raw`W_{\mathrm{summed}}`, NOTATION_COLORS.summed),
  v_free_component: makeEntry('V_free,a', String.raw`V_{\mathrm{free},a}`, NOTATION_COLORS.free),
  w_summed_component: makeEntry('W_summed,a', String.raw`W_{\mathrm{summed},a}`, NOTATION_COLORS.summed),
  l_labels: makeEntry('L', 'L', NOTATION_COLORS.structureMuted),
  l_component: makeEntry('L_a', 'L_a', NOTATION_COLORS.structureMuted),
  u_axis_classes: makeEntry('U', 'U', NOTATION_COLORS.structureMuted),
  m_incidence: makeEntry('M', 'M', NOTATION_COLORS.structure),
  g_wreath: makeEntry('G_wreath', String.raw`G_{\mathrm{wreath}}`, NOTATION_COLORS.object),
  h_family: makeEntry('H_i', 'H_i', NOTATION_COLORS.object),
  sigma_row_move: makeEntry('σ', String.raw`\sigma`, NOTATION_COLORS.action),
  pi_relabeling: makeEntry('π', String.raw`\pi`, NOTATION_COLORS.action),
  sym_l: makeEntry('Sym(L)', String.raw`\mathrm{Sym}(L)`, NOTATION_COLORS.object),
  g_detected: makeEntry('G', 'G', NOTATION_COLORS.object),
  g_component: makeEntry('G_a', 'G_a', NOTATION_COLORS.object),
  g_formal: makeEntry('G_f', String.raw`G_{\text{f}}`, NOTATION_COLORS.object),
  g_pointwise: makeEntry('G_pt', String.raw`G_{\text{pt}}`, NOTATION_COLORS.object),
  g_pointwise_restricted_v: makeEntry('G_pt|_V_free', String.raw`G_{\text{pt}}\big|_{V_{\mathrm{free}}}`, NOTATION_COLORS.free),
  s_w_summed: makeEntry('S(W_summed)', String.raw`S(W_{\mathrm{summed}})`, NOTATION_COLORS.summed),
  g_v_factor: makeEntry('G_V_free', String.raw`G_{V_{\mathrm{free}}}`, NOTATION_COLORS.free),
  g_w_factor: makeEntry('G_W_summed', String.raw`G_{W_{\mathrm{summed}}}`, NOTATION_COLORS.summed),
  x_space: makeEntry('X', 'X', NOTATION_COLORS.object),
  x_component: makeEntry('X_a', 'X_a', NOTATION_COLORS.object),
  x_w_summed: makeEntry('X_W_summed', String.raw`X_{W_{\mathrm{summed}}}`, NOTATION_COLORS.summed),
  orbit_space_component: makeEntry('X/G_a', String.raw`X / G_a`, NOTATION_COLORS.object),
  orbit_o: makeEntry('O', 'O', NOTATION_COLORS.object),
  projection_pi_v_free: makeEntry('π_V_free(O)', String.raw`\pi_{V_{\mathrm{free}}}(O)`, NOTATION_COLORS.free),
  mu_total: makeEntry('μ', String.raw`\mu`, NOTATION_COLORS.quantity),
  alpha_total: makeEntry('α', String.raw`\alpha`, NOTATION_COLORS.quantity),
  alpha_component: makeEntry('α_a', String.raw`\alpha_a`, NOTATION_COLORS.quantity),
  m_total: makeEntry('M', 'M', NOTATION_COLORS.quantity),
  m_component: makeEntry('M_a', 'M_a', NOTATION_COLORS.quantity),
  k_operands: makeEntry('k', 'k', NOTATION_COLORS.quantity),
  g_element: makeEntry('g', 'g', NOTATION_COLORS.action),
  n_label: makeEntry('n_ℓ', String.raw`n_\ell`, NOTATION_COLORS.quantity),
  n_cycle: makeEntry('n_c', 'n_c', NOTATION_COLORS.quantity),
  n_l: makeEntry('n_L', 'n_L', NOTATION_COLORS.quantity),
  omega_orbit: makeEntry('Ω', String.raw`\Omega`, NOTATION_COLORS.object),
  n_omega: makeEntry('n_Ω', String.raw`n_\Omega`, NOTATION_COLORS.quantity),
  c_omega_cycles: makeEntry('c_Ω(g)', String.raw`c_\Omega(g)`, NOTATION_COLORS.quantity),
  r_complement: makeEntry('R', 'R', NOTATION_COLORS.structureMuted),
};

export const NOTATION_HOST_FILES = [
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
  'components/symmetry-aware-einsum-contractions/components/AccumulationHardCard.jsx',
  'components/symmetry-aware-einsum-contractions/components/explorerNarrative.js',
  'components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx',
  'components/symmetry-aware-einsum-contractions/components/CaseBadge.jsx',
  'components/symmetry-aware-einsum-contractions/components/ComponentView.jsx',
  'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx',
  'components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx',
  'components/symmetry-aware-einsum-contractions/components/DiminoView.jsx',
  'components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx',
  'components/symmetry-aware-einsum-contractions/components/GlossaryList.jsx',
  'components/symmetry-aware-einsum-contractions/components/GlossaryProse.jsx',
  'components/symmetry-aware-einsum-contractions/components/InlineMathText.jsx',
  'components/symmetry-aware-einsum-contractions/components/Latex.jsx',
  'components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx',
  'components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx',
  'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx',
  'components/symmetry-aware-einsum-contractions/components/VSubSwConstruction.jsx',
  'components/symmetry-aware-einsum-contractions/components/WreathStructureView.jsx',
  'components/symmetry-aware-einsum-contractions/engine/regimeSpec.js',
  'components/symmetry-aware-einsum-contractions/engine/shapeSpec.js',
];

const AUTO_COLORED_NOTATION_IDS = [
  'l_labels',
  'u_axis_classes',
  'm_incidence',
  'v_free',
  'w_summed',
  'v_free_component',
  'w_summed_component',
  'l_component',
  'g_wreath',
  'h_family',
  'sigma_row_move',
  'pi_relabeling',
  'sym_l',
  'g_detected',
  'g_component',
  'g_formal',
  'g_pointwise',
  'g_pointwise_restricted_v',
  's_w_summed',
  'g_v_factor',
  'g_w_factor',
  'x_w_summed',
  'projection_pi_v_free',
  'mu_total',
  'alpha_total',
  'alpha_component',
  'm_total',
  'm_component',
  'k_operands',
  'n_label',
  'n_l',
  'n_cycle',
  'omega_orbit',
  'n_omega',
  'c_omega_cycles',
];

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function buildStandaloneRegex(latex) {
  const escaped = escapeRegex(latex);
  return new RegExp(`(?<![_A-Za-z])${escaped}(?![_A-Za-z])`, 'g');
}

const AUTO_COLOR_RULES = AUTO_COLORED_NOTATION_IDS
  .map((id) => ({ id, latex: NOTATION_REGISTRY[id].latex }))
  .sort((a, b) => b.latex.length - a.latex.length)
  .map(({ id, latex }) => ({
    id,
    regex: buildStandaloneRegex(latex),
  }));

function readBraceGroup(source, openIndex) {
  if (source[openIndex] !== '{') return null;
  let depth = 0;
  for (let i = openIndex; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) {
        return {
          content: source.slice(openIndex + 1, i),
          end: i + 1,
        };
      }
    }
  }
  return null;
}

function protectExistingTextColor(source) {
  const marker = String.raw`\textcolor`;
  let masked = '';
  let cursor = 0;
  const protectedSegments = [];

  while (cursor < source.length) {
    const start = source.indexOf(marker, cursor);
    if (start === -1) {
      masked += source.slice(cursor);
      break;
    }

    masked += source.slice(cursor, start);

    const colorGroup = readBraceGroup(source, start + marker.length);
    if (!colorGroup) {
      masked += marker;
      cursor = start + marker.length;
      continue;
    }

    const bodyGroup = readBraceGroup(source, colorGroup.end);
    if (!bodyGroup) {
      masked += source.slice(start, colorGroup.end);
      cursor = colorGroup.end;
      continue;
    }

    const placeholder = `@@TEXTCOLOR_${protectedSegments.length}@@`;
    protectedSegments.push(source.slice(start, bodyGroup.end));
    masked += placeholder;
    cursor = bodyGroup.end;
  }

  return { masked, protectedSegments };
}

function notationEntry(id) {
  const entry = NOTATION_REGISTRY[id];
  if (!entry) {
    throw new Error(`Unknown notation id: ${id}`);
  }
  return entry;
}

function resolveColor(id) {
  const activeTheme = getExplorerThemePreset(getThemeStoreActiveExplorerThemeId());
  if (activeTheme.mathPaletteId === EDITORIAL_NOIR_RICH_MATH_PALETTE_ID) {
    const mathRole = NOTATION_MATH_ROLE_BY_ID[id];
    if (mathRole && Object.prototype.hasOwnProperty.call(EDITORIAL_NOIR_RICH_MATH_ROLES, mathRole)) {
      return EDITORIAL_NOIR_RICH_MATH_ROLES[mathRole];
    }
  }
  const themeRole = NOTATION_THEME_ROLE_BY_ID[id];
  const activeExplorerThemeRoles = getThemeStoreActiveExplorerThemeRoles();
  if (themeRole && Object.prototype.hasOwnProperty.call(activeExplorerThemeRoles, themeRole)) {
    return activeExplorerThemeRoles[themeRole];
  }
  return notationEntry(id).color;
}

export function setActiveExplorerTheme(themeOrId) {
  setThemeStoreActiveExplorerTheme(themeOrId);
}

export function resetActiveExplorerTheme() {
  resetThemeStoreActiveExplorerTheme();
}

export function getActiveExplorerThemeId() {
  return getThemeStoreActiveExplorerThemeId();
}

export function getActiveExplorerThemeRoles() {
  return getThemeStoreActiveExplorerThemeRoles();
}

// Legacy compatibility only. Explorer themes own notation colors.
export function setActiveNotationGrammar(grammarId, paletteOverride = null) {
  void paletteOverride;
  activeNotationGrammarId = grammarId ?? 'current';
}

export function resetActiveNotationPalette() {
  activeNotationGrammarId = 'current';
}

export function getActiveNotationGrammarId() {
  return activeNotationGrammarId;
}

export function notationText(id) {
  return notationEntry(id).text;
}

export function notationLatex(id) {
  return notationEntry(id).latex;
}

export function notationColor(id) {
  return resolveColor(id);
}

export function notationColoredLatex(
  id,
  latexOverride = notationLatex(id),
  wrapInGroup = false,
) {
  const colored = String.raw`\textcolor{${notationColor(id)}}{${latexOverride}}`;
  return wrapInGroup ? `{${colored}}` : colored;
}

export function colorizeNotationLatex(math) {
  if (typeof math !== 'string' || math.length === 0) return math;

  const { masked, protectedSegments } = protectExistingTextColor(math);
  let colorized = masked;
  const placeholders = [];

  for (const { id, regex } of AUTO_COLOR_RULES) {
    colorized = colorized.replace(regex, (match, offset) => {
      const precedingChar = colorized[offset - 1];
      const superscriptOrSubscript = precedingChar === '^' || precedingChar === '_';
      const replacement = notationColoredLatex(id, match, superscriptOrSubscript);
      const key = `@@NOTATION_${placeholders.length}@@`;
      placeholders.push(replacement);
      return key;
    });
  }

  placeholders.forEach((replacement, idx) => {
    colorized = colorized.replace(`@@NOTATION_${idx}@@`, replacement);
  });

  protectedSegments.forEach((replacement, idx) => {
    colorized = colorized.replace(`@@TEXTCOLOR_${idx}@@`, replacement);
  });

  return colorized;
}

export function notationTint(id, alpha) {
  const color = notationColor(id).slice(1);
  const r = Number.parseInt(color.slice(0, 2), 16);
  const g = Number.parseInt(color.slice(2, 4), 16);
  const b = Number.parseInt(color.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function notationIdForRole(role) {
  return role === 'w' ? 'w_summed' : 'v_free';
}
