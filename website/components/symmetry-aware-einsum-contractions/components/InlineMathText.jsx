import Latex from './Latex.jsx';

export function renderTooltipInlineText(text, keyPrefix) {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g)
    .filter(Boolean)
    .map((segment, index) => {
      if (segment.startsWith('**') && segment.endsWith('**') && segment.length > 4) {
        return (
          <strong key={`${keyPrefix}-bold-${index}`} className="font-semibold text-current">
            {segment.slice(2, -2)}
          </strong>
        );
      }
      if (
        segment.startsWith('*')
        && segment.endsWith('*')
        && !segment.startsWith('**')
        && !segment.endsWith('**')
        && segment.length > 2
      ) {
        return (
          <em key={`${keyPrefix}-italic-${index}`} className="italic text-current">
            {segment.slice(1, -1)}
          </em>
        );
      }
      if (segment.startsWith('`') && segment.endsWith('`') && segment.length > 2) {
        return (
          <code
            key={`${keyPrefix}-code-${index}`}
            className="rounded bg-gray-100 px-1.5 py-[1px] font-mono text-[0.92em] text-gray-700"
          >
            {segment.slice(1, -1)}
          </code>
        );
      }
      return <span key={`${keyPrefix}-text-${index}`}>{segment}</span>;
    });
}

/**
 * Render a string that embeds inline LaTeX segments wrapped in single `$`.
 *
 * Example: "The per-tuple group $G_{\\text{pt}}$ drives compression."
 *          ↦ "The per-tuple group [G_pt rendered via KaTeX] drives compression."
 *
 * - A `$$` literal is preserved (paired double-dollar reduces to single-dollar
 *   text; we keep it verbatim to avoid swallowing intentional prose dollars).
 * - Mismatched singles (odd count of `$` in the string) fall back to plain
 *   text for the trailing segment, to avoid losing content.
 * - Non-string children pass through unchanged, so callers can interleave
 *   JSX nodes with strings without special-casing.
 */
export default function InlineMathText({ children, themeOverride = null }) {
  if (children == null) return null;
  if (typeof children !== 'string') return children;

  // Split into alternating text / math segments. The regex captures the
  // delimiter so the split preserves math spans in the array.
  const parts = children.split(/(\$[^$]+\$)/g);
  return (
    <>
      {parts.flatMap((part, i) => {
        if (part.startsWith('$') && part.endsWith('$') && part.length >= 3) {
          return [<Latex key={`math-${i}`} math={part.slice(1, -1)} themeOverride={themeOverride} />];
        }
        return renderTooltipInlineText(part, `segment-${i}`);
      })}
    </>
  );
}
