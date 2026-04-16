// website/components/symmetry-aware-einsum-contractions/engine/regimes/bruteForceOrbit.js
import { withinBruteForceBudget, bruteForceEstimate } from '../budget.js';

function allAssignments(sizes) {
  const out = [];
  const current = new Array(sizes.length).fill(0);
  function visit(idx) {
    if (idx === sizes.length) { out.push([...current]); return; }
    for (let v = 0; v < sizes[idx]; v += 1) { current[idx] = v; visit(idx + 1); }
  }
  if (sizes.length === 0) out.push([]); else visit(0);
  return out;
}

function applyPerm(tuple, perm) {
  const out = new Array(tuple.length);
  for (let i = 0; i < tuple.length; i += 1) out[perm.arr[i]] = tuple[i];
  return out;
}

function key(t) { return t.join('|'); }

export const bruteForceOrbitRegime = {
  id: 'bruteForceOrbit',
  recognize(ctx) {
    if (withinBruteForceBudget(ctx.sizes, ctx.elements.length)) {
      return { fired: true, reason: `within budget (estimate = ${bruteForceEstimate(ctx.sizes, ctx.elements.length).toLocaleString()})` };
    }
    return {
      fired: false,
      reason: `brute-force estimate ${bruteForceEstimate(ctx.sizes, ctx.elements.length).toLocaleString()} exceeds budget`,
    };
  },
  compute({ elements, sizes, visiblePositions }) {
    const remaining = new Map();
    for (const a of allAssignments(sizes)) remaining.set(key(a), a);
    let total = 0;
    while (remaining.size > 0) {
      const [, rep] = remaining.entries().next().value;
      const orbit = new Map();
      for (const g of elements) {
        const moved = applyPerm(rep, g);
        const k = key(moved);
        if (!orbit.has(k)) orbit.set(k, moved);
        remaining.delete(k);
      }
      const projected = new Set();
      for (const m of orbit.values()) projected.add(visiblePositions.map((p) => m[p]).join('|'));
      total += projected.size;
    }
    return {
      count: total,
      latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
      latexSymbolic: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
      subTrace: undefined,
    };
  },
};
