const makeEntry = (text, latex, color) => ({ text, latex, color });

export const NOTATION_REGISTRY = {
  v_free: makeEntry('V_free', String.raw`V_{\mathrm{free}}`, '#E8555A'),
  w_summed: makeEntry('W_summed', String.raw`W_{\mathrm{summed}}`, '#64748B'),
  v_free_component: makeEntry('V_free,a', String.raw`V_{\mathrm{free},a}`, '#3B82F6'),
  w_summed_component: makeEntry('W_summed,a', String.raw`W_{\mathrm{summed},a}`, '#7C8FA3'),
  l_labels: makeEntry('L', 'L', '#52606D'),
  l_component: makeEntry('L_a', 'L_a', '#685C8E'),
  u_axis_classes: makeEntry('U', 'U', '#9A3412'),
  m_incidence: makeEntry('M', 'M', '#292C2D'),
  g_wreath: makeEntry('G_wreath', String.raw`G_{\mathrm{wreath}}`, '#9F67D2'),
  h_family: makeEntry('H_i', 'H_i', '#C2410C'),
  sigma_row_move: makeEntry('σ', String.raw`\sigma`, '#D23934'),
  pi_relabeling: makeEntry('π', String.raw`\pi`, '#0F766E'),
  sym_l: makeEntry('Sym(L)', String.raw`\mathrm{Sym}(L)`, '#0D9488'),
  g_detected: makeEntry('G', 'G', '#8B5CF6'),
  g_component: makeEntry('G_a', 'G_a', '#7E57C2'),
  g_formal: makeEntry('G_f', String.raw`G_{\text{f}}`, '#A855F7'),
  g_pointwise: makeEntry('G_pt', String.raw`G_{\text{pt}}`, '#9333EA'),
  g_pointwise_restricted_v: makeEntry('G_pt|_V_free', String.raw`G_{\text{pt}}\big|_{V_{\mathrm{free}}}`, '#6366F1'),
  s_w_summed: makeEntry('S(W_summed)', String.raw`S(W_{\mathrm{summed}})`, '#334155'),
  g_v_factor: makeEntry('G_V_free', String.raw`G_{V_{\mathrm{free}}}`, '#2952CC'),
  g_w_factor: makeEntry('G_W_summed', String.raw`G_{W_{\mathrm{summed}}}`, '#5B6B7A'),
  x_space: makeEntry('X', 'X', '#0E7490'),
  x_component: makeEntry('X_a', 'X_a', '#0284C7'),
  x_w_summed: makeEntry('X_W_summed', String.raw`X_{W_{\mathrm{summed}}}`, '#0369A1'),
  orbit_space_component: makeEntry('X/G_a', String.raw`X / G_a`, '#1D4ED8'),
  orbit_o: makeEntry('O', 'O', '#0891B2'),
  projection_pi_v_free: makeEntry('π_V_free(O)', String.raw`\pi_{V_{\mathrm{free}}}(O)`, '#1B7FDB'),
  alpha_total: makeEntry('α', String.raw`\alpha`, '#B45309'),
  alpha_component: makeEntry('α_a', String.raw`\alpha_a`, '#A16207'),
  m_total: makeEntry('M', 'M', '#059669'),
  m_component: makeEntry('M_a', 'M_a', '#047857'),
  k_operands: makeEntry('k', 'k', '#475569'),
  g_element: makeEntry('g', 'g', '#DD6B20'),
  n_label: makeEntry('n_ℓ', String.raw`n_\ell`, '#7E22CE'),
  n_cycle: makeEntry('n_c', 'n_c', '#2F855A'),
  omega_orbit: makeEntry('Ω', String.raw`\Omega`, '#C05621'),
  n_omega: makeEntry('n_Ω', String.raw`n_\Omega`, '#7C2D12'),
  r_complement: makeEntry('R', 'R', '#5B5B5B'),
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
  'alpha_total',
  'alpha_component',
  'm_component',
  'k_operands',
  'n_label',
  'n_cycle',
  'omega_orbit',
  'n_omega',
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

export function notationText(id) {
  return notationEntry(id).text;
}

export function notationLatex(id) {
  return notationEntry(id).latex;
}

export function notationColor(id) {
  return notationEntry(id).color;
}

export function notationColoredLatex(id, latexOverride = notationLatex(id)) {
  return String.raw`\textcolor{${notationColor(id)}}{${latexOverride}}`;
}

export function colorizeNotationLatex(math) {
  if (typeof math !== 'string' || math.length === 0) return math;

  const { masked, protectedSegments } = protectExistingTextColor(math);
  let colorized = masked;
  const placeholders = [];

  for (const { id, regex } of AUTO_COLOR_RULES) {
    colorized = colorized.replace(regex, (match) => {
      const key = `@@NOTATION_${placeholders.length}@@`;
      placeholders.push(notationColoredLatex(id, match));
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
