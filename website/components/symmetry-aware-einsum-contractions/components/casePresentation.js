import { CASE_META } from '../engine/componentDecomposition.js';
import { CLASSIFICATION_LEAVES } from '../engine/classificationSpec.js';

/**
 * Rendering metadata for each caseType (trivial, A, B, C, D, E).
 *
 * The labels, descriptions, and formulas are pulled from the classification
 * spec (CLASSIFICATION_LEAVES), so the case badges and tooltips stay in sync
 * with whatever the engine actually routes components to.
 */
function leafByCaseType(caseType) {
  for (const leaf of Object.values(CLASSIFICATION_LEAVES)) {
    if (leaf.caseType === caseType) return leaf;
  }
  return null;
}

function buildPresentation(caseType) {
  const leaf = leafByCaseType(caseType);
  if (!leaf) return null;
  return {
    label: CASE_META[caseType]?.label ?? leaf.label,
    shortLabel: leaf.shortLabel,
    methodLabel: leaf.methodLabel ?? leaf.label,
    humanName: leaf.humanName ?? null,
    tooltip: {
      title: leaf.label,
      body: leaf.description,
      latex: leaf.latex ?? null,
      glossary: leaf.glossary ?? null,
    },
  };
}

export const CASE_PRESENTATION = Object.fromEntries(
  ['trivial', 'A', 'B', 'C', 'D', 'E']
    .map((caseType) => [caseType, buildPresentation(caseType)])
    .filter(([, presentation]) => presentation !== null),
);

export function getCasePresentation(caseType) {
  const key = String(caseType ?? '?');
  return CASE_PRESENTATION[key] ?? {
    label: `Case ${key}`,
    shortLabel: key,
    methodLabel: 'Orbit enumeration',
    humanName: null,
    tooltip: null,
  };
}
