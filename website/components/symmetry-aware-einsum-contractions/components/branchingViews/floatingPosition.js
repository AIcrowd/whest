// Pure-function viewport-edge flip math for tooltip-style floating cards.
// No React, no DOM — easily unit-tested.
//
// Used by:
//   - components/branchingViews/OrbitDetailCard.jsx (large pin card)
//   - components/branchingViews/MatrixHoverTooltip.jsx (small hover gloss,
//     uses a simplified call site that always passes a small cardW/cardH)

/**
 * Compute the top-left position for a floating card anchored at a click
 * point. Default placement is below + to the right of the click; flip
 * to the opposite side when the card would overflow the viewport.
 *
 * If the card is bigger than the viewport on either axis, clamp to the
 * viewport with `padding` margin (the card itself should provide its own
 * internal scroll in that degenerate case).
 *
 * @param {object} args
 * @param {number} args.clickX — pointer-event clientX
 * @param {number} args.clickY — pointer-event clientY
 * @param {number} args.cardW — card outer width (px)
 * @param {number} args.cardH — card outer height (px)
 * @param {number} args.viewportW — window.innerWidth
 * @param {number} args.viewportH — window.innerHeight
 * @param {number} args.padding — gap between click point and card edge (px)
 * @returns {{ left: number, top: number }}
 */
export function flipPosition({ clickX, clickY, cardW, cardH, viewportW, viewportH, padding }) {
  // Default: below + right.
  let left = clickX + padding;
  let top = clickY + padding;

  // Flip horizontally if card would overflow right edge.
  if (left + cardW > viewportW - padding) {
    left = clickX - padding - cardW;
  }
  // Clamp horizontally if still off (card wider than viewport).
  if (left < padding) {
    left = padding;
  }

  // Flip vertically if card would overflow bottom edge.
  if (top + cardH > viewportH - padding) {
    top = clickY - padding - cardH;
  }
  // Clamp vertically if still off (card taller than viewport).
  if (top < padding) {
    top = padding;
  }

  return { left, top };
}
