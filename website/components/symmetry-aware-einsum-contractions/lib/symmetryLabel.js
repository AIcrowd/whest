// website/components/symmetry-aware-einsum-contractions/lib/symmetryLabel.js
//
// Shared short-form label for an operand's declared axis symmetry. Produces
// the same text the main-page builder and sidebar use — `dense`, `S3`, `C4`,
// `D2`, `custom (2 gens)` — so the appendix modal's per-preset savings table
// matches the vocabulary the reader has already internalized from the rest
// of the page.

import { parseCycleNotation } from '../engine/cycleParser.js';

/**
 * Compact, page-consistent label for a single operand's declared symmetry.
 *
 * Output forms:
 *   - 'dense'             — no declared symmetry (variable.symmetry === 'none')
 *   - 'S3' / 'S4' / ...   — symmetric group on k axes (k = |symAxes| or rank)
 *   - 'C3' / 'C4' / ...   — cyclic group on k axes
 *   - 'D3' / 'D4' / ...   — dihedral group on k axes
 *   - 'custom'            — custom generators, unparseable / empty
 *   - 'custom (2 gens)'   — custom generators, successfully parsed
 *
 * Mirrors the `badgeLabel` helper local to ExampleChooser.jsx (which now
 * delegates here), so the two renderings cannot drift.
 */
export function variableSymmetryLabel(variable) {
  if (!variable) return 'dense';
  const { symmetry, rank, symAxes, generators } = variable;
  if (symmetry === 'none') return 'dense';
  if (symmetry === 'custom') {
    if (!generators || !generators.trim()) return 'custom';
    const parsed = parseCycleNotation(generators);
    if (parsed.error || !parsed.generators) return 'custom';
    const n = parsed.generators.length;
    return `custom (${n} gen${n !== 1 ? 's' : ''})`;
  }
  const k = (symAxes && symAxes.length) || rank;
  const prefix = symmetry === 'symmetric' ? 'S' : symmetry === 'cyclic' ? 'C' : 'D';
  return `${prefix}${k}`;
}
