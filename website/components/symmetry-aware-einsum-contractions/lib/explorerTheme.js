const HERO = '#F0524D';
const HERO_MUTED = '#D23934';
const INK = '#292C2D';
const MUTED = '#5D5F60';
const BORDER = '#D9DCDC';
const SURFACE = '#FFFFFF';
const SURFACE_INSET = '#F8F9F9';
const SUMMED_SIDE = '#64748B';
const DEEP_SLATE = '#334155';
const MID_SLATE = '#94A3B8';
const LIGHT_SLATE = '#D1D5DB';
const INFO_DEEP = '#2959C4';
const INFO = '#4A7CFF';
const QUANTITY_TEAL = '#0B6D7A';
const WARM_EXCEPTION = '#B29F9E';
const EMBER = '#C7632F';
const BURNT_SIENNA = '#9A3412';
const DEEP_TEAL = '#0F766E';
const BRONZE = '#B45309';
const CORAL_TINT = '#F7A09D';
const MID_GRAY = '#AAACAD';
const STATUS_SUCCESS = '#23B761';
const STATUS_WARNING = '#FA9E33';

const BASE_THEME_ROLES = {
  hero: HERO,
  heroMuted: HERO_MUTED,
  ink: INK,
  body: INK,
  muted: MUTED,
  border: BORDER,
  surface: SURFACE,
  surfaceInset: SURFACE_INSET,
  freeSide: HERO,
  summedSide: SUMMED_SIDE,
  symmetryObject: DEEP_SLATE,
  action: DEEP_SLATE,
  quantity: QUANTITY_TEAL,
  statusSuccess: STATUS_SUCCESS,
  statusWarning: STATUS_WARNING,
};

function createExplorerTheme(id, label, summary, roles) {
  return {
    id,
    label,
    summary,
    roles,
  };
}

function createThemeRoles(overrides = {}) {
  return {
    ...BASE_THEME_ROLES,
    ...overrides,
  };
}

function clampAlpha(alpha) {
  return Math.max(0, Math.min(1, alpha));
}

function hexToRgb(hex) {
  const normalized = String(hex ?? '').trim().replace('#', '');
  if (!/^[0-9A-Fa-f]{6}$/.test(normalized)) return null;
  return {
    r: Number.parseInt(normalized.slice(0, 2), 16),
    g: Number.parseInt(normalized.slice(2, 4), 16),
    b: Number.parseInt(normalized.slice(4, 6), 16),
  };
}

function rgba(hex, alpha) {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${clampAlpha(alpha)})`;
}

export const EXPLORER_THEME_RECOMMENDED_ID = 'editorial-balance';

export const EXPLORER_THEME_PRESETS = [
  createExplorerTheme(
    'strict-editorial',
    'Strict editorial',
    'A restrained presentation that keeps symmetry objects grounded in slate.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: DEEP_SLATE,
      action: DEEP_SLATE,
      quantity: INK,
    }),
  ),
  createExplorerTheme(
    'editorial-balance',
    'Editorial balance',
    'The recommended balance between the coral free side and a calm blue object family.',
    createThemeRoles({
      symmetryObject: DEEP_SLATE,
      action: DEEP_SLATE,
      quantity: QUANTITY_TEAL,
    }),
  ),
  createExplorerTheme(
    'editorial-balance-slate',
    'Editorial balance slate',
    'Keeps the editorial-balance structure, but replaces the cool quantity lane with the mean-prop slate.',
    createThemeRoles({
      symmetryObject: DEEP_SLATE,
      action: DEEP_SLATE,
      quantity: DEEP_SLATE,
      statusSuccess: DEEP_SLATE,
    }),
  ),
  createExplorerTheme(
    'editorial-balance-warm',
    'Editorial balance warm',
    'A warmer editorial-balance variant that uses the restrained cov-prop accent instead of the green quantity lane.',
    createThemeRoles({
      symmetryObject: DEEP_SLATE,
      action: DEEP_SLATE,
      quantity: WARM_EXCEPTION,
      statusSuccess: WARM_EXCEPTION,
      statusWarning: HERO_MUTED,
    }),
  ),
  createExplorerTheme(
    'teaching-calm',
    'Teaching calm',
    'A softer teaching mode with a deep blue symmetry-object family.',
    createThemeRoles({
      symmetryObject: INFO_DEEP,
      action: STATUS_WARNING,
      quantity: QUANTITY_TEAL,
    }),
  ),
  createExplorerTheme(
    'quiet-ledger',
    'Quiet ledger',
    'An extra-neutral reading mode that keeps both quantities and symmetry close to slate.',
    createThemeRoles({
      symmetryObject: DEEP_SLATE,
      action: MUTED,
      quantity: DEEP_SLATE,
    }),
  ),
  createExplorerTheme(
    'soft-coral',
    'Soft coral',
    'Lets coral do a little more work while keeping the structural layer cool and pale.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: DEEP_SLATE,
      action: HERO_MUTED,
      quantity: QUANTITY_TEAL,
    }),
  ),
  createExplorerTheme(
    'quiet-info',
    'Quiet info',
    'A calmer blue-note variant where the object family deepens without becoming loud.',
    createThemeRoles({
      symmetryObject: INFO_DEEP,
      action: DEEP_SLATE,
      quantity: QUANTITY_TEAL,
    }),
  ),
  createExplorerTheme(
    'deep-info-ledger',
    'Deep info ledger',
    'Pairs the deeper info blue with slate counts for a cooler, more diagrammatic paper tone.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: INFO_DEEP,
      action: SUMMED_SIDE,
      quantity: DEEP_SLATE,
    }),
  ),
  createExplorerTheme(
    'warm-exception',
    'Warm exception',
    'Uses one sanctioned warm-gray accent in the secondary lane while the main math stays coral and slate.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: DEEP_SLATE,
      action: WARM_EXCEPTION,
      quantity: QUANTITY_TEAL,
    }),
  ),
  createExplorerTheme(
    'muted-amber',
    'Muted amber',
    'Keeps actions legible with amber, but lets the rest of the page stay mostly ink and slate.',
    createThemeRoles({
      symmetryObject: DEEP_SLATE,
      action: STATUS_WARNING,
      quantity: DEEP_SLATE,
    }),
  ),
  createExplorerTheme(
    'whestbench-axis',
    'Whestbench axis',
    'Maps the explorer onto the whestbench coral↔slate axis, with quantities staying on the cool side.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: DEEP_SLATE,
      action: MID_SLATE,
      quantity: DEEP_SLATE,
      statusSuccess: MID_SLATE,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'whestbench-axis-blue',
    'Whestbench axis blue',
    'A stricter whestbench axis: black and coral lead, mean-prop blue replaces the gray lane, and warm-gray stays a restrained accent only.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: DEEP_SLATE,
      action: WARM_EXCEPTION,
      quantity: DEEP_SLATE,
      statusSuccess: DEEP_SLATE,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'whestbench-sampling',
    'Whestbench sampling',
    'Lets the whestbench sampling coral carry quantities and key math accents while symmetry stays on slate.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: DEEP_SLATE,
      action: MID_SLATE,
      quantity: HERO_MUTED,
      statusSuccess: HERO_MUTED,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'whestbench-cov-prop',
    'Whestbench cov-prop',
    'Brings in the single warm-gray exception from whestbench for action and cost accents.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: DEEP_SLATE,
      action: WARM_EXCEPTION,
      quantity: WARM_EXCEPTION,
      statusSuccess: WARM_EXCEPTION,
      statusWarning: HERO_MUTED,
    }),
  ),
  createExplorerTheme(
    'whestbench-diverging',
    'Whestbench diverging',
    'Treats quantities as the positive end of the whestbench coral↔slate axis with pale structural surrounds.',
    createThemeRoles({
      summedSide: LIGHT_SLATE,
      symmetryObject: DEEP_SLATE,
      action: WARM_EXCEPTION,
      quantity: HERO,
      muted: MID_GRAY,
      statusSuccess: HERO_MUTED,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'whestbench-verdict',
    'Whestbench verdict',
    'Ground-truth ink leads, mean-prop slate carries structure, and the restrained warm-gray lane handles scored quantities.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: INK,
      action: DEEP_SLATE,
      quantity: WARM_EXCEPTION,
      muted: MID_GRAY,
      statusSuccess: WARM_EXCEPTION,
      statusWarning: HERO_MUTED,
    }),
  ),
  createExplorerTheme(
    'whestbench-scorecard',
    'Whestbench scorecard',
    'A scorecard-flavored editorial mode with coral accents, slate structure, and no hues outside the whestbench axis.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: DEEP_SLATE,
      action: HERO_MUTED,
      quantity: HERO_MUTED,
      muted: MID_GRAY,
      statusSuccess: HERO_MUTED,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'whestbench-sage',
    'Whestbench sage',
    'An airy white-paper variant that stays strictly on the ground-truth / sampling / mean-prop axis with pale slate support.',
    createThemeRoles({
      summedSide: LIGHT_SLATE,
      symmetryObject: DEEP_SLATE,
      action: MID_SLATE,
      quantity: MID_SLATE,
      muted: MID_GRAY,
      statusSuccess: MID_SLATE,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'coral-slate-contrast',
    'Coral slate contrast',
    'A sharp coral↔slate split that keeps the page paper-like while pushing the whestbench tension harder.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: DEEP_SLATE,
      action: MID_SLATE,
      quantity: DEEP_SLATE,
      muted: MID_GRAY,
      statusSuccess: HERO_MUTED,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'coral-slate-split',
    'Coral slate split',
    'A lighter diverging variant that keeps the high-contrast coral/slate axis but softens the secondary lane.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: DEEP_SLATE,
      action: MID_SLATE,
      quantity: MID_SLATE,
      muted: MID_GRAY,
      statusSuccess: MID_SLATE,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'coral-slate-hardline',
    'Coral slate hardline',
    'A paper-with-red-markup treatment where ink and deep slate dominate and coral lands in harder, rarer hits.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: INK,
      action: DEEP_SLATE,
      quantity: INK,
      muted: DEEP_SLATE,
      statusSuccess: HERO_MUTED,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'ink-authority',
    'Ink authority',
    'An authority-first editorial mode with ink, white, and slate carrying most of the page and coral used sparingly.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: INK,
      action: DEEP_SLATE,
      quantity: DEEP_SLATE,
      muted: MUTED,
      statusSuccess: DEEP_SLATE,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'editorial-noir',
    'Editorial noir',
    'A darker editorial variant that strengthens black/slate blocks while keeping coral as a precise accent rather than a field color.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: INK,
      action: MID_SLATE,
      quantity: INK,
      muted: DEEP_SLATE,
      statusSuccess: HERO_MUTED,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'mean-prop-led',
    'Mean-prop led',
    'Lets the mean-prop slate own the structural lane while coral stays the hero accent and black remains the authority anchor.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: DEEP_SLATE,
      action: INFO_DEEP,
      quantity: DEEP_SLATE,
      muted: MID_SLATE,
      statusSuccess: DEEP_SLATE,
      statusWarning: HERO_MUTED,
    }),
  ),
  createExplorerTheme(
    'cool-proof',
    'Cool proof',
    'A cooler analytical paper mode that gives formulas and structure a stronger blue-slate spine without losing the editorial feel.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: INFO_DEEP,
      action: DEEP_SLATE,
      quantity: DEEP_SLATE,
      muted: MID_SLATE,
      statusSuccess: DEEP_SLATE,
      statusWarning: HERO_MUTED,
    }),
  ),
  createExplorerTheme(
    'blue-ledger',
    'Blue ledger',
    'A high-contrast ledger variant with a stronger slate-blue spine and coral preserved as the singular hero accent.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: INFO_DEEP,
      action: MID_SLATE,
      quantity: INFO_DEEP,
      muted: MID_GRAY,
      statusSuccess: INFO_DEEP,
      statusWarning: WARM_EXCEPTION,
    }),
  ),
  createExplorerTheme(
    'warm-margin',
    'Warm margin',
    'A bookish high-contrast mode that keeps coral, black, and slate dominant while surfacing the warm exception more visibly.',
    createThemeRoles({
      summedSide: DEEP_SLATE,
      symmetryObject: DEEP_SLATE,
      action: WARM_EXCEPTION,
      quantity: WARM_EXCEPTION,
      muted: MID_GRAY,
      statusSuccess: WARM_EXCEPTION,
      statusWarning: HERO_MUTED,
    }),
  ),
  createExplorerTheme(
    'cov-prop-editorial',
    'Cov-prop editorial',
    'A restrained warm-exception editorial variant that replaces the old green/teal quantity lane with the cov-prop warm gray.',
    createThemeRoles({
      summedSide: MID_SLATE,
      symmetryObject: INK,
      action: WARM_EXCEPTION,
      quantity: WARM_EXCEPTION,
      muted: MID_GRAY,
      statusSuccess: WARM_EXCEPTION,
      statusWarning: HERO_MUTED,
    }),
  ),
];

let activeExplorerThemeId = EXPLORER_THEME_RECOMMENDED_ID;
const activeExplorerThemeListeners = new Set();

function notifyActiveExplorerThemeListeners() {
  activeExplorerThemeListeners.forEach((listener) => listener());
}

export function getExplorerThemeRoles(themeOrId) {
  if (themeOrId && typeof themeOrId === 'object') {
    return themeOrId.roles ?? getExplorerThemePreset(themeOrId.id).roles;
  }
  return getExplorerThemePreset(themeOrId).roles;
}

export function explorerThemeColor(themeOrId, role) {
  return getExplorerThemeRoles(themeOrId)[role];
}

export function explorerThemeTint(themeOrId, role, alpha) {
  return rgba(explorerThemeColor(themeOrId, role), alpha);
}

export function getExplorerThemeOperandPalette(themeOrId) {
  const roles = getExplorerThemeRoles(themeOrId);
  return [...new Set([
    roles.quantity,
    roles.symmetryObject,
    roles.action,
    roles.heroMuted,
    roles.summedSide,
    BURNT_SIENNA,
    EMBER,
    DEEP_TEAL,
    BRONZE,
    WARM_EXCEPTION,
    CORAL_TINT,
    INFO,
  ])];
}

export function getExplorerThemeFingerprintPalette(themeOrId) {
  const roles = getExplorerThemeRoles(themeOrId);
  return [
    roles.symmetryObject,
    roles.quantity,
    roles.action,
    roles.summedSide,
    roles.heroMuted,
  ];
}

export function getExplorerThemeCssVariables(themeOrId) {
  const roles = getExplorerThemeRoles(themeOrId);
  return {
    '--coral': roles.freeSide,
    '--coral-hover': roles.heroMuted,
    '--coral-light': rgba(roles.freeSide, 0.14),
    '--success': roles.quantity,
    '--warning': roles.action,
    '--info': roles.symmetryObject,
    '--ein-v': roles.freeSide,
    '--ein-w': roles.summedSide,
    '--gray-900': roles.ink,
    '--gray-600': roles.muted,
    '--gray-400': rgba(roles.muted, 0.72),
    '--gray-200': roles.border,
    '--gray-100': roles.surfaceInset,
    '--gray-50': roles.surfaceInset,
    '--white': roles.surface,
    '--background': roles.surface,
    '--foreground': roles.ink,
    '--card': roles.surface,
    '--muted': roles.surfaceInset,
    '--muted-foreground': roles.muted,
    '--border': roles.border,
    '--status-success': roles.statusSuccess,
    '--status-warning': roles.statusWarning,
  };
}

export function getExplorerThemePreset(id) {
  return EXPLORER_THEME_PRESETS.find((preset) => preset.id === id)
    ?? EXPLORER_THEME_PRESETS[1];
}

export function setActiveExplorerTheme(themeOrId) {
  const nextThemeId = typeof themeOrId === 'object' && themeOrId !== null
    ? getExplorerThemePreset(themeOrId.id).id
    : getExplorerThemePreset(themeOrId).id;

  if (nextThemeId === activeExplorerThemeId) return;
  activeExplorerThemeId = nextThemeId;
  notifyActiveExplorerThemeListeners();
}

export function resetActiveExplorerTheme() {
  setActiveExplorerTheme(EXPLORER_THEME_RECOMMENDED_ID);
}

export function getActiveExplorerThemeId() {
  return activeExplorerThemeId;
}

export function getActiveExplorerThemeRoles() {
  return getExplorerThemePreset(activeExplorerThemeId).roles;
}

export function subscribeActiveExplorerTheme(listener) {
  activeExplorerThemeListeners.add(listener);
  return () => {
    activeExplorerThemeListeners.delete(listener);
  };
}
