// website/components/symmetry-aware-einsum-contractions/components/regimePresentation.js
import { REGIME_SPEC } from '../engine/regimeSpec.js';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import {
  explorerThemeColor,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor } from '../lib/notationSystem.js';

/**
 * Rendering metadata for each regime or shape id.
 *
 * Entries for regimes (singleton, directProduct, bruteForceOrbit) come from REGIME_SPEC.
 * Entries for shapes (trivial, allVisible, allSummed, mixed) come from SHAPE_SPEC.
 */

function presentationColorFromSpec(spec, themeOverride = null) {
  const themeRef = themeOverride ?? getActiveExplorerThemeId();
  if (spec.themeRole) {
    const themedColor = explorerThemeColor(themeRef, spec.themeRole);
    if (themedColor) return themedColor;
  }
  if (spec.colorId) return notationColor(spec.colorId, themeOverride);
  return spec.color ?? '#94A3B8';
}

function regimePresentationFromSpec(id, themeOverride = null) {
  const spec = REGIME_SPEC[id] || SHAPE_SPEC[id];
  if (!spec) return null;
  return {
    id,
    label: spec.label,
    shortLabel: spec.shortLabel,
    color: presentationColorFromSpec(spec, themeOverride),
    tooltip: {
      title: spec.label,
      body: spec.description,
      whenText: spec.when ?? null,
      latex: spec.latex ?? null,
      glossary: spec.glossary ?? null,
    },
  };
}

export function getRegimePresentation(id, themeOverride = null) {
  const key = String(id ?? '?');
  return (
    regimePresentationFromSpec(key, themeOverride)
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
