const HERO = '#F0524D';
const HERO_MUTED = '#D23934';
const INK = '#292C2D';
const MUTED = '#5D5F60';
const BORDER = '#D9DCDC';
const SURFACE = '#FFFFFF';
const SURFACE_INSET = '#F8F9F9';
const SUMMED_SIDE = '#64748B';
const STATUS_SUCCESS = '#23B761';
const STATUS_WARNING = '#FA9E33';

function createExplorerTheme(id, label, summary, roles) {
  return {
    id,
    label,
    summary,
    roles,
  };
}

export const EXPLORER_THEME_RECOMMENDED_ID = 'editorial-balance';

export const EXPLORER_THEME_PRESETS = [
  createExplorerTheme(
    'strict-editorial',
    'Strict editorial',
    'A restrained presentation that keeps symmetry objects grounded in slate.',
    {
      hero: HERO,
      heroMuted: HERO_MUTED,
      ink: INK,
      body: INK,
      muted: MUTED,
      border: BORDER,
      surface: SURFACE,
      surfaceInset: SURFACE_INSET,
      freeSide: HERO,
      summedSide: '#334155',
      symmetryObject: '#334155',
      action: '#334155',
      quantity: INK,
      statusSuccess: STATUS_SUCCESS,
      statusWarning: STATUS_WARNING,
    },
  ),
  createExplorerTheme(
    'editorial-balance',
    'Editorial balance',
    'The recommended balance between the coral free side and a calm blue object family.',
    {
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
      symmetryObject: '#334155',
      action: '#334155',
      quantity: '#0B6D7A',
      statusSuccess: STATUS_SUCCESS,
      statusWarning: STATUS_WARNING,
    },
  ),
  createExplorerTheme(
    'teaching-calm',
    'Teaching calm',
    'A softer teaching mode with a deep blue symmetry-object family.',
    {
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
      symmetryObject: '#2959C4',
      action: '#FA9E33',
      quantity: '#0B6D7A',
      statusSuccess: STATUS_SUCCESS,
      statusWarning: STATUS_WARNING,
    },
  ),
];

export function getExplorerThemePreset(id) {
  return EXPLORER_THEME_PRESETS.find((preset) => preset.id === id)
    ?? EXPLORER_THEME_PRESETS[1];
}
