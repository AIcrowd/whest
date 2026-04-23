// website/components/symmetry-aware-einsum-contractions/lib/symmetryLabel.js
//
// Shared short-form label for an operand's declared axis symmetry. Produces
// the same text the main-page builder, sticky bar, and appendix use —
// `dense`, `S3`, `C4`, `D2`, `⟨(0 1), (2 3)⟩` — so every symmetry surface in
// the explorer uses one vocabulary.

import { parseCycleNotation } from '../engine/cycleParser.js';

function stringifyGenerator(generator) {
  return generator.map((cycle) => `(${cycle.join(' ')})`).join('');
}

export function formatGeneratorNotation(generatorsOrString) {
  if (!generatorsOrString) return null;

  let generators = null;
  if (typeof generatorsOrString === 'string') {
    if (!generatorsOrString.trim()) return null;
    const parsed = parseCycleNotation(generatorsOrString);
    if (parsed.error || !parsed.generators?.length) return null;
    generators = parsed.generators;
  } else if (Array.isArray(generatorsOrString) && generatorsOrString.length > 0) {
    generators = generatorsOrString;
  } else {
    return null;
  }

  const text = generators.map((generator) => stringifyGenerator(generator)).join(', ');
  return text ? `⟨${text}⟩` : null;
}

/**
 * Compact, page-consistent label for a single operand's declared symmetry.
 *
 * Output forms:
 *   - 'dense'             — no declared symmetry (variable.symmetry === 'none')
 *   - 'S3' / 'S4' / ...   — symmetric group on k axes (k = |symAxes| or rank)
 *   - 'C3' / 'C4' / ...   — cyclic group on k axes
 *   - 'D3' / 'D4' / ...   — dihedral group on k axes
 *   - 'custom'            — custom generators, unparseable / empty
 *   - '⟨(0 1), (2 3)⟩'    — custom generators in canonical cycle notation
 *
 * Mirrors the `badgeLabel` helper local to ExampleChooser.jsx (which now
 * delegates here), so the two renderings cannot drift.
 */
export function variableSymmetryLabel(variable) {
  if (!variable) return 'dense';
  const { symmetry, rank, symAxes, generators } = variable;
  if (symmetry === 'none') return 'dense';
  if (symmetry === 'custom') {
    return formatGeneratorNotation(generators) ?? 'custom';
  }
  const k = (symAxes && symAxes.length) || rank;
  const prefix = symmetry === 'symmetric' ? 'S' : symmetry === 'cyclic' ? 'C' : 'D';
  return `${prefix}${k}`;
}
