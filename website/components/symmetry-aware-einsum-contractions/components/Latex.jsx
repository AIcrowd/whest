import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { colorizeNotationLatex, getActiveExplorerThemeId } from '../lib/notationSystem.js';

// KaTeX renders mathematical glyphs (including Greek letters like α, U+03B1)
// as regular HTML text nodes. If an ancestor element sets
// `text-transform: uppercase` (for a caption/eyebrow style, Tailwind's
// `uppercase` utility, etc.), the browser will dutifully transform those
// Greek letters into their capital counterparts — U+03B1 (α) becomes U+0391
// (Α), which is visually indistinguishable from Latin capital A. The same
// trap applies to `capitalize` and `lowercase`. Setting `text-transform:
// none` on the KaTeX-bearing wrapper element shields its content, so
// `<Latex math="\alpha" />` always renders a lowercase α regardless of
// ancestor typography rules. This is the correct default: mathematical
// notation is a distinct typographic layer and should never be
// case-transformed by surrounding prose styles.
const MATH_WRAPPER_STYLE = { textTransform: 'none' };

/**
 * Renders a LaTeX expression using KaTeX.
 * @param {string} math - LaTeX expression
 * @param {boolean} display - true for display mode (centered block), false for inline
 * @param {boolean} colorize - true to apply shared notation colors, false for neutral math
 * @param {object|string|null} themeOverride - optional explorer theme override for notation colors
 */
export default function Latex({ math, display = false, colorize = true, themeOverride = null }) {
  const activeExplorerThemeId = getActiveExplorerThemeId();
  const html = useMemo(() => {
    try {
      return katex.renderToString(colorize ? colorizeNotationLatex(math, themeOverride) : math, {
        displayMode: display,
        throwOnError: false,
        trust: true,
      });
    } catch {
      return math;
    }
  }, [math, display, colorize, activeExplorerThemeId, themeOverride]);

  return display
    ? <div style={MATH_WRAPPER_STYLE} dangerouslySetInnerHTML={{ __html: html }} />
    : <span className="mx-[0.08em]" style={MATH_WRAPPER_STYLE} dangerouslySetInnerHTML={{ __html: html }} />;
}
