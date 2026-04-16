// website/components/symmetry-aware-einsum-contractions/engine/regimes/vSetwiseStable.js
import { sizeAwareBurnside } from '../sizeAware/burnside.js';
import { Permutation } from '../permutation.js';

function labelIndex(labels) {
  const m = Object.create(null);
  labels.forEach((l, i) => { m[l] = i; });
  return m;
}

function preservesV(elements, vSet) {
  for (const g of elements) {
    const moved = new Set();
    for (const v of vSet) moved.add(g.arr[v]);
    if (moved.size !== vSet.size) return false;
    for (const m of moved) if (!vSet.has(m)) return false;
  }
  return true;
}

function allVisibleAssignments(vSizes) {
  const out = [];
  const current = new Array(vSizes.length).fill(0);
  function visit(idx) {
    if (idx === vSizes.length) { out.push([...current]); return; }
    for (let v = 0; v < vSizes[idx]; v += 1) { current[idx] = v; visit(idx + 1); }
  }
  if (vSizes.length === 0) out.push([]); else visit(0);
  return out;
}

function keyOf(arr) { return arr.join('|'); }

/**
 * Apply g to a visible assignment u. When g preserves V setwise, it induces
 * a permutation on [n]^V: new u'[j] = old u[i] where v_j = g(v_i), i.e.
 * the position-labeled-by-v_i now has its old value at position-labeled-by-g(v_i).
 */
function applyOnV(g, u, vPositions) {
  const out = new Array(u.length);
  for (let i = 0; i < u.length; i += 1) {
    const target = g.arr[vPositions[i]];
    const j = vPositions.indexOf(target);
    out[j] = u[i];
  }
  return out;
}

/**
 * Restrict a permutation to W positions, returning a Permutation acting on
 * local positions 0..|W|-1. Assumes V is G-setwise-stable (so W is too).
 */
function restrictToW(g, wPositions) {
  const arr = wPositions.map((p) => wPositions.indexOf(g.arr[p]));
  return new Permutation(arr);
}

export const vSetwiseStableRegime = {
  id: 'vSetwiseStable',
  recognize({ labels, va, elements }) {
    const idx = labelIndex(labels);
    const vSet = new Set(va.map((l) => idx[l]));
    if (!preservesV(elements, vSet)) {
      return { fired: false, reason: 'V is not setwise stable under G' };
    }
    return { fired: true, reason: 'every g ∈ G preserves V setwise' };
  },
  compute({ labels, va, wa, elements, sizes }) {
    const idx = labelIndex(labels);
    const vPositions = va.map((l) => idx[l]);
    const wPositions = wa.map((l) => idx[l]);
    const vSizes = vPositions.map((p) => sizes[p]);
    const wSizes = wPositions.map((p) => sizes[p]);

    // Build kernel K = elements fixing every V-label pointwise.
    const K = elements.filter((g) => vPositions.every((p) => g.arr[p] === p));
    const subTrace = [];
    if (K.length > 1) {
      subTrace.push({
        regimeId: 'kernelReduction',
        decision: 'fired',
        reason: `|K| = ${K.length} > 1 — kernel elements fix V pointwise`,
      });
    }

    const visible = allVisibleAssignments(vSizes);
    const remaining = new Map();
    for (const u of visible) remaining.set(keyOf(u), u);

    let total = 0;
    while (remaining.size > 0) {
      const [, rep] = remaining.entries().next().value;
      const orbit = new Map();
      const stabilizer = [];
      for (const g of elements) {
        const gU = applyOnV(g, rep, vPositions);
        const k = keyOf(gU);
        if (!orbit.has(k)) orbit.set(k, gU);
        if (k === keyOf(rep)) stabilizer.push(g);
      }
      for (const k of orbit.keys()) remaining.delete(k);
      const wRestricted = stabilizer.map((g) => restrictToW(g, wPositions));
      const hiddenOrbits = wSizes.length > 0 ? sizeAwareBurnside(wRestricted, wSizes) : 1;
      total += orbit.size * hiddenOrbits;
    }

    return {
      count: total,
      latex: String.raw`A = \sum_{[u] \in [n]^V / H} |H \cdot u| \cdot |[n]^W / G_u|`,
      latexSymbolic: String.raw`A = \sum_{[u]} |H \cdot u| \cdot |[n]^W / G_u|`,
      subTrace: subTrace.length ? subTrace : undefined,
    };
  },
};
