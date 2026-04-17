// Single source of truth for the five recurring math symbols (V, W, G, μ, α)
// across the symmetry-aware einsum explorer. Distill-style convention: a
// symbol has the same colour everywhere it appears — in prose, in formulas,
// in diagrams — so the reader's eye ties them together without re-reading.
//
// The mapping below is anchored to concepts the rest of the site already
// colours:
//   - summed labels live in coral (--primary); W is the summed axis,
//     and μ counts the multiplications we do on the summed side.
//   - free labels live in muted stone; V is the free axis.
//   - G is the group itself — structural, neutral foreground.
//   - α is the accumulation count; amber-700 is the page's "write" colour
//     (matches the Mental Framework Step 2 rule).
//
// Export BOTH hex strings (for KaTeX \textcolor, which needs a hex) and
// Tailwind class names (for inline prose spans). Any component that renders
// one of these symbols should import from here rather than reach for its
// own colour literal.

export const MATH_COLOR_HEX = {
  V: '#5D5F60',      // --muted-foreground (stone-600)
  W: '#F0524D',      // --primary (coral)
  G: '#0F172A',      // slate-900 (neutral foreground)
  mu: '#F0524D',     // coral — mult shares the summed-side palette
  alpha: '#B45309',  // amber-700 — matches Step 2 accumulation rule
};

export const MATH_COLOR_CLASS = {
  V: 'text-muted-foreground',
  W: 'text-primary',
  G: 'text-foreground',
  mu: 'text-primary',
  alpha: 'text-amber-700',
};

// Known single-character tokens we recognise in prose and re-colour. Order
// matters only for the regex: longer tokens before shorter to avoid greedy
// mismatches (not currently an issue since all are 1 char).
export const MATH_TOKEN_ORDER = ['V', 'W', 'G', 'μ', 'α'];

const TOKEN_TO_KEY = {
  V: 'V',
  W: 'W',
  G: 'G',
  μ: 'mu',
  α: 'alpha',
};

/**
 * Lookup hex colour for a raw token character.
 * Returns undefined for unknown tokens so callers can no-op cleanly.
 */
export function hexForToken(token) {
  return MATH_COLOR_HEX[TOKEN_TO_KEY[token]];
}

/**
 * Lookup Tailwind class for a raw token character.
 */
export function classForToken(token) {
  return MATH_COLOR_CLASS[TOKEN_TO_KEY[token]];
}

/**
 * Word-boundary regex that matches any of the five known tokens in prose.
 * Used by the MathText renderer to split a description string into plain
 * runs + coloured tokens.
 *
 * `u` flag so the Greek letters are matched as single characters and not
 * as surrogate-pair fragments.
 */
export const MATH_TOKEN_REGEX = /(\bV\b|\bW\b|\bG\b|μ|α)/gu;
