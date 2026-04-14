/**
 * Color palette and symmetry icon mappings for the symmetry explorer.
 *
 * Each variable gets a unique color (cycling through PALETTE) and an icon
 * determined by its symmetry type.  The buildVariableColors() helper
 * produces the lookup map consumed by every visualisation layer.
 */

// Operand colors — chosen to avoid V-free blue (#4A7CFF) and W-summed
// slate (#64748B).  Warm/saturated tones that stay distinct from each
// other and from the V/W semantic colors.
export const PALETTE = [
  '#E85D04',   // burnt orange
  '#9B5DE5',   // vivid purple
  '#00BBF9',   // cyan
  '#F15BB5',   // hot pink
  '#00F5D4',   // mint/teal
  '#FEE440',   // bright yellow
  '#D62828',   // deep red
  '#06D6A0',   // emerald
  '#118AB2',   // ocean blue (darker, distinct from V-blue)
  '#073B4C',   // dark teal
];

export const SYMMETRY_ICONS = {
  none: '',
  symmetric: '\u25C6',   // ◆ diamond
  cyclic: '\u27F3',       // ⟳ cycle arrow
  dihedral: '\u2B22',     // ⬢ hexagon
  custom: '\u2699',       // ⚙ gear
};

/**
 * Return '#fff' or '#000' depending on which contrasts better with the
 * given hex background color (WCAG relative luminance formula).
 */
export function contrastText(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  // sRGB linearisation
  const R = r <= 0.03928 ? r / 12.92 : ((r + 0.055) / 1.055) ** 2.4;
  const G = g <= 0.03928 ? g / 12.92 : ((g + 0.055) / 1.055) ** 2.4;
  const B = b <= 0.03928 ? b / 12.92 : ((b + 0.055) / 1.055) ** 2.4;
  const L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
  return L > 0.4 ? '#000' : '#fff';
}

/**
 * Derive a human-readable symmetry label.
 *
 * - none      → 'dense'
 * - symmetric → 'S<k>'  (k = symAxes.length or rank)
 * - cyclic    → 'C<k>'
 * - dihedral  → 'D<k>'
 * - custom    → 'custom'
 */
function symmetryLabel(symmetry, rank, symAxes) {
  const k = (symAxes && symAxes.length) || rank;
  switch (symmetry) {
    case 'none':
      return 'dense';
    case 'symmetric':
      return `S${k}`;
    case 'cyclic':
      return `C${k}`;
    case 'dihedral':
      return `D${k}`;
    case 'custom':
      return 'custom';
    default:
      return 'dense';
  }
}

/**
 * Build a color / icon map keyed by variable name.
 *
 * @param {Array<{name: string, rank: number, symmetry: string, symAxes: number[]}>} variables
 * @returns {Object.<string, {color: string, icon: string, symmetryLabel: string, symmetry: string}>}
 */
export function buildVariableColors(variables) {
  const map = {};
  let idx = 0;

  for (const v of variables) {
    if (map[v.name] !== undefined) continue;

    map[v.name] = {
      color: PALETTE[idx % PALETTE.length],
      icon: SYMMETRY_ICONS[v.symmetry] ?? '',
      symmetryLabel: symmetryLabel(v.symmetry, v.rank, v.symAxes),
      symmetry: v.symmetry,
    };
    idx++;
  }

  return map;
}
