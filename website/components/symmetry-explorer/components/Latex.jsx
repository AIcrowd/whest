import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

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
    ? <div dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}
