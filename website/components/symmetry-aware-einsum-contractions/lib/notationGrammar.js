import {
  EXPLORER_THEME_RECOMMENDED_ID,
} from './explorerTheme.js';

// Compatibility shim only.
// Explorer themes are the page-wide source of truth for notation colors now.
// Legacy notation grammar ids remain exportable so stale imports can still
// resolve metadata, but they no longer carry authoritative palettes.
function createLegacyGrammar(id, label, summary, explorerThemeId = EXPLORER_THEME_RECOMMENDED_ID) {
  return {
    id,
    label,
    summary,
    explorerThemeId,
    palette: {},
  };
}

export const NOTATION_GRAMMAR_RECOMMENDED_ID = 'split-cost';

export const NOTATION_GRAMMAR_PRESETS = [
  createLegacyGrammar(
    'current',
    'Current grammar',
    'Legacy alias for the recommended explorer theme.',
  ),
  createLegacyGrammar(
    'brand-neutral',
    'Brand-neutral',
    'Legacy compatibility preset that now defers to explorer themes.',
  ),
  createLegacyGrammar(
    'split-cost',
    'Split cost families',
    'Legacy recommended grammar id retained as an alias for the recommended explorer theme.',
  ),
  createLegacyGrammar(
    'side-first',
    'Side-first',
    'Legacy compatibility preset that now defers to explorer themes.',
  ),
  createLegacyGrammar(
    'coral-universal',
    'Coral + Universal',
    'Legacy compatibility preset that now defers to explorer themes.',
  ),
  createLegacyGrammar(
    'coral-deep-info',
    'Coral + Deep Info',
    'Legacy compatibility preset that now maps to the teaching-calm explorer theme.',
    'teaching-calm',
  ),
  createLegacyGrammar(
    'coral-slate-discipline',
    'Coral + Slate Discipline',
    'Legacy compatibility preset that now maps to the strict-editorial explorer theme.',
    'strict-editorial',
  ),
  createLegacyGrammar(
    'coral-cost-split',
    'Coral + Cost Split',
    'Legacy compatibility preset that now defers to explorer themes.',
  ),
  createLegacyGrammar(
    'coral-teal-symmetry',
    'Coral + Teal Symmetry',
    'Legacy compatibility preset that now maps to the teaching-calm explorer theme.',
    'teaching-calm',
  ),
  createLegacyGrammar(
    'coral-slate-objects',
    'Coral + Slate Objects',
    'Legacy compatibility preset that now maps to the strict-editorial explorer theme.',
    'strict-editorial',
  ),
  createLegacyGrammar(
    'coral-blue-amber-split',
    'Coral + Blue Amber Split',
    'Legacy compatibility preset that now maps to the teaching-calm explorer theme.',
    'teaching-calm',
  ),
  createLegacyGrammar(
    'coral-deep-blue-symmetry',
    'Coral + Deep Blue Symmetry',
    'Legacy compatibility preset that now maps to the teaching-calm explorer theme.',
    'teaching-calm',
  ),
  createLegacyGrammar(
    'coral-monochrome-objects',
    'Coral + Monochrome Objects',
    'Legacy compatibility preset that now maps to the strict-editorial explorer theme.',
    'strict-editorial',
  ),
];

export function getNotationGrammarPreset(id) {
  return NOTATION_GRAMMAR_PRESETS.find((preset) => preset.id === id) ?? NOTATION_GRAMMAR_PRESETS[0];
}
