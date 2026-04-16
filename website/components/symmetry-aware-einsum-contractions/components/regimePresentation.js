// website/components/symmetry-aware-einsum-contractions/components/regimePresentation.js
import { REGIME_SPEC } from '../engine/regimeSpec.js';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import { CLASSIFICATION_LEAVES } from '../engine/classificationSpec.js';
import { CASE_META } from '../engine/componentDecomposition.js';

/**
 * Rendering metadata for each regime or shape id.
 *
 * Entries for regimes (singleton, fullSymmetric, ...) come from REGIME_SPEC.
 * Entries for shapes (trivial, allVisible, allSummed, mixed) come from SHAPE_SPEC.
 * Legacy case types (A, B, C, D, E, trivial) map via CLASSIFICATION_LEAVES for
 * backwards compatibility — still used by some tests and preset metadata.
 */

const LEGACY_CASE_COLORS = {
  trivial: '#94A3B8',
  A: '#4A7CFF',
  B: '#64748B',
  C: '#FA9E33',
  D: '#23B761',
  E: '#F0524D',
};

function buildLegacyPresentation(caseType) {
  for (const leaf of Object.values(CLASSIFICATION_LEAVES)) {
    if (leaf.caseType !== caseType) continue;
    return {
      id: leaf.id,
      label: CASE_META[caseType]?.label ?? leaf.label,
      shortLabel: leaf.shortLabel,
      methodLabel: leaf.methodLabel ?? leaf.label,
      humanName: leaf.humanName ?? null,
      color: LEGACY_CASE_COLORS[caseType] ?? '#94A3B8',
      tooltip: {
        title: leaf.label,
        body: leaf.description,
        latex: leaf.latex ?? null,
        glossary: leaf.glossary ?? null,
      },
    };
  }
  return null;
}

const LEGACY_CASE_PRESENTATION = Object.fromEntries(
  ['trivial', 'A', 'B', 'C', 'D', 'E']
    .map((caseType) => [caseType, buildLegacyPresentation(caseType)])
    .filter(([, p]) => p !== null),
);

function regimePresentationFromSpec(id) {
  if (REGIME_SPEC[id]) {
    const spec = REGIME_SPEC[id];
    return {
      id,
      label: spec.label,
      shortLabel: spec.shortLabel,
      methodLabel: spec.label,
      humanName: spec.description,
      color: spec.color ?? '#94A3B8',
      tooltip: {
        title: spec.label,
        body: spec.description,
        whenText: spec.when ?? null,
        latex: spec.latex ?? null,
        glossary: null,
      },
    };
  }
  if (SHAPE_SPEC[id]) {
    const spec = SHAPE_SPEC[id];
    return {
      id,
      label: spec.label,
      shortLabel: spec.shortLabel,
      methodLabel: spec.label,
      humanName: spec.description,
      color: spec.color ?? '#94A3B8',
      tooltip: {
        title: spec.label,
        body: spec.description,
        whenText: spec.when ?? null,
        latex: spec.latex ?? null,
        glossary: null,
      },
    };
  }
  return null;
}

export function getRegimePresentation(id) {
  const key = String(id ?? '?');
  return (
    regimePresentationFromSpec(key)
    || LEGACY_CASE_PRESENTATION[key]
    || {
      id: key,
      label: `Regime ${key}`,
      shortLabel: key,
      methodLabel: 'Orbit enumeration',
      humanName: null,
      tooltip: null,
    }
  );
}

// Backwards-compatible alias consumed by older call sites.
export function getCasePresentation(caseType) {
  return getRegimePresentation(caseType);
}

export const CASE_PRESENTATION = LEGACY_CASE_PRESENTATION;
export const REGIME_PRESENTATION = Object.fromEntries(
  Object.keys(REGIME_SPEC).map((id) => [id, regimePresentationFromSpec(id)]),
);
