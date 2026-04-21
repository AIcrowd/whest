// website/components/symmetry-aware-einsum-contractions/components/regimePresentation.js
import { REGIME_SPEC } from '../engine/regimeSpec.js';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import { notationColor } from '../lib/notationSystem.js';

/**
 * Rendering metadata for each regime or shape id.
 *
 * Entries for regimes (singleton, directProduct, bruteForceOrbit) come from REGIME_SPEC.
 * Entries for shapes (trivial, allVisible, allSummed, mixed) come from SHAPE_SPEC.
 */

function regimePresentationFromSpec(id) {
  const spec = REGIME_SPEC[id] || SHAPE_SPEC[id];
  if (!spec) return null;
  return {
    id,
    label: spec.label,
    shortLabel: spec.shortLabel,
    color: spec.colorId ? notationColor(spec.colorId) : (spec.color ?? '#94A3B8'),
    tooltip: {
      title: spec.label,
      body: spec.description,
      whenText: spec.when ?? null,
      latex: spec.latex ?? null,
      glossary: spec.glossary ?? null,
    },
  };
}

export function getRegimePresentation(id) {
  const key = String(id ?? '?');
  return (
    regimePresentationFromSpec(key)
    || {
      id: key,
      label: `Regime ${key}`,
      shortLabel: key,
      tooltip: null,
    }
  );
}

export const REGIME_PRESENTATION = Object.fromEntries(
  Object.keys(REGIME_SPEC).map((id) => [id, regimePresentationFromSpec(id)]),
);
