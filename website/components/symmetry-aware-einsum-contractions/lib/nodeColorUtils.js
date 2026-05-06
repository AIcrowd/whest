// Pure colour-mixing helpers for graph nodes (DecisionLadder leaves,
// OrbitProjectionGraph members + reps, future xyflow nodes). Pulled out of
// DecisionLadder so they can be reused without dragging the whole
// classification-tree component into other graph views.
//
// All inputs are hex strings of the form "#RRGGBB". The mix helpers return
// `rgb(r, g, b)` strings ready to drop into a style attribute.

export function mixWithWhite(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mix = (c) => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

export function mixWithBlack(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mix = (c) => Math.round(c * (1 - amount));
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

// WCAG-style relative luminance — used to decide whether a tinted background
// needs light or dark text for legible contrast.
export function relativeLuminance(hex) {
  const rgb = [hex.slice(1, 3), hex.slice(3, 5), hex.slice(5, 7)]
    .map((part) => parseInt(part, 16) / 255)
    .map((channel) => (
      channel <= 0.03928
        ? channel / 12.92
        : ((channel + 0.055) / 1.055) ** 2.4
    ));
  return (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2]);
}

export function isDarkColor(hex) {
  return relativeLuminance(hex) < 0.28;
}

// readableTextOn picks a foreground hex appropriate for the given background.
// Light foreground "#F8FAFC" matches DecisionLadder LeafNode's dark-on-dark
// rule; dark foreground "#132228" matches its light-on-light rule. Pulled
// here so node renderers across the page agree on contrast.
export function readableTextOn(hex) {
  return isDarkColor(hex) ? '#F8FAFC' : '#132228';
}
