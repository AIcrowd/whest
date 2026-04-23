// website/components/symmetry-aware-einsum-contractions/engine/shapeLayer.js

export const SHAPES = ['trivial', 'allVisible', 'allSummed', 'mixed'];

export function detectShape({ va, wa, elements }) {
  if (!elements || elements.length <= 1) return { kind: 'trivial' };
  if (wa.length === 0) return { kind: 'allVisible' };
  if (va.length === 0) return { kind: 'allSummed' };
  return { kind: 'mixed' };
}
