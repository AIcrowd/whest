import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

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
 */
export default function Latex({ math, display = false }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: display,
        throwOnError: false,
        trust: true,
      });
    } catch {
      return math;
    }
  }, [math, display]);

  return display
    ? <div style={MATH_WRAPPER_STYLE} dangerouslySetInnerHTML={{ __html: html }} />
    : <span style={MATH_WRAPPER_STYLE} dangerouslySetInnerHTML={{ __html: html }} />;
}
