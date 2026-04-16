import Latex from './Latex.jsx';

/**
 * Renders a glossary string where `$...$` segments are rendered as inline KaTeX
 * (so math symbols in the prose visually match the math font of the displayed
 * formula above). Everything outside the markers renders as plain prose text.
 *
 * This is the distill-style convention: running prose with inline math segments
 * that share the same typesetting as the displayed equation.
 */
export default function GlossaryProse({ text }) {
  if (!text) return null;

  const parts = [];
  let cursor = 0;
  let keyCounter = 0;

  while (cursor < text.length) {
    const start = text.indexOf('$', cursor);
    if (start === -1) {
      parts.push({ kind: 'text', value: text.slice(cursor), key: keyCounter++ });
      break;
    }
    if (start > cursor) {
      parts.push({ kind: 'text', value: text.slice(cursor, start), key: keyCounter++ });
    }
    const end = text.indexOf('$', start + 1);
    if (end === -1) {
      // Unbalanced — treat the remainder as literal text (don't silently eat it).
      parts.push({ kind: 'text', value: text.slice(start), key: keyCounter++ });
      break;
    }
    const math = text.slice(start + 1, end);
    if (math.length > 0) {
      parts.push({ kind: 'math', value: math, key: keyCounter++ });
    }
    cursor = end + 1;
  }

  return (
    <>
      {parts.map((part) =>
        part.kind === 'math'
          ? <Latex key={part.key} math={part.value} />
          : <span key={part.key}>{part.value}</span>,
      )}
    </>
  );
}
