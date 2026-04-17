import { Fragment } from 'react';
import { MATH_TOKEN_REGEX, classForToken } from '../lib/mathColors.js';

/**
 * Distill-style prose renderer that re-colours the shared math tokens
 * (V, W, G, μ, α) wherever they appear in a plain-text string. Everything
 * else renders as-is.
 *
 *   <MathText>Factor V and W — generators split cleanly.</MathText>
 *
 * renders "V" in stone, "W" in coral, and the rest in the inherited
 * foreground colour.
 *
 * Tokens are matched on word boundaries (for Latin letters) or as direct
 * characters (for Greek); see `MATH_TOKEN_REGEX`.
 */
export default function MathText({ children, className = '' }) {
  if (typeof children !== 'string') {
    return <span className={className}>{children}</span>;
  }

  const parts = children.split(MATH_TOKEN_REGEX);

  return (
    <span className={className}>
      {parts.map((part, idx) => {
        const tokenClass = classForToken(part);
        if (tokenClass) {
          return (
            <span key={idx} className={`${tokenClass} font-mono`}>
              {part}
            </span>
          );
        }
        return <Fragment key={idx}>{part}</Fragment>;
      })}
    </span>
  );
}
