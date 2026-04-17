// Per-component identity colors. Used by the Interaction Graph hulls
// (Act 4) and the Aggregation Explainer (Act 5), so that a reader can
// trace one einsum component visually from its hull in the graph into
// the matching M_a / α_a factor in the combine formula.
export const COMPONENT_COLORS = ['#4A7CFF', '#23B761', '#FA9E33', '#7C3AED', '#F0524D'];

export function componentColor(idx) {
  return COMPONENT_COLORS[idx % COMPONENT_COLORS.length];
}
