/**
 * Color palette and symmetry icon mappings for the symmetry explorer.
 *
 * Each variable gets a unique color (cycling through PALETTE) and an icon
 * determined by its symmetry type.  The buildVariableColors() helper
 * produces the lookup map consumed by every visualisation layer.
 */

export const PALETTE = [
  '#4a7cff',
  '#ffb74d',
  '#bb86fc',
  '#ec4899',
  '#22c55e',
  '#94a3b8',
  '#ef4444',
  '#06b6d4',
  '#f59e0b',
  '#8b5cf6',
];

export const SYMMETRY_ICONS = {
  none: '',
  symmetric: '\u25C6',   // ◆ diamond
  cyclic: '\u27F3',       // ⟳ cycle arrow
  dihedral: '\u2B22',     // ⬢ hexagon
  custom: '\u2699',       // ⚙ gear
};

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
