// website/components/symmetry-aware-einsum-contractions/engine/regimes/directProduct.js
import { sizeAwareBurnside } from '../sizeAware/burnside.js';
import { Permutation } from '../permutation.js';

function labelIndices(labels) {
  const idx = Object.create(null);
  labels.forEach((l, i) => { idx[l] = i; });
  return idx;
}

/**
 * Restrict a permutation acting on global positions to a Permutation acting
 * on the local positions wPos (which must be W-invariant for this regime).
 * Returns a new Permutation whose arr has length = wPos.length.
 */
function restrictToW(g, wPos) {
  const localIndex = new Map();
  wPos.forEach((p, i) => { localIndex.set(p, i); });
  const arr = new Array(wPos.length);
  for (let i = 0; i < wPos.length; i += 1) {
    const globalImage = g.arr[wPos[i]];
    const localImage = localIndex.get(globalImage);
    if (localImage === undefined) {
      // Should never happen: elements accepted by recognize() never move W to non-W.
      throw new Error('directProduct.restrictToW: element moves W out of W');
    }
    arr[i] = localImage;
  }
  return new Permutation(arr);
}

export const directProductRegime = {
  id: 'directProduct',
  /**
   * Fire iff (a) every post-Dimino element preserves V/W setwise (no cross),
   * (b) |G| = |G_V| · |G_W| (F-check), and (c) both projections non-trivial
   * (meaningfulness guard: V- or W-trivial reduces to allVisible/allSummed).
   */
  recognize({ labels, va, wa, elements = [] }) {
    if (!elements.length) return { fired: false, reason: 'no elements provided' };
    const idx = labelIndices(labels);
    const vPos = va.map((l) => idx[l]);
    const wPos = wa.map((l) => idx[l]);
    const vSet = new Set(vPos);

    // Every element must preserve V/W setwise — no cross-V/W elements.
    for (const g of elements) {
      for (const pos of vPos) {
        if (!vSet.has(g.arr[pos])) {
          return { fired: false, reason: 'element crosses V→W' };
        }
      }
    }

    // Compute V- and W-projection cardinalities.
    const gVKeys = new Set();
    const gWKeys = new Set();
    for (const g of elements) {
      gVKeys.add(vPos.map((p) => g.arr[p]).join(','));
      gWKeys.add(wPos.map((p) => g.arr[p]).join(','));
    }

    // Meaningfulness guard — both projections must be non-trivial.
    if (gVKeys.size <= 1 || gWKeys.size <= 1) {
      return { fired: false, reason: 'V- or W-projection is trivial' };
    }

    // F-check.
    const ok = elements.length === gVKeys.size * gWKeys.size;
    return ok
      ? { fired: true, reason: `|G|=${elements.length} = |G_V|·|G_W|=${gVKeys.size}·${gWKeys.size}` }
      : { fired: false, reason: `|G|=${elements.length} ≠ |G_V|·|G_W|=${gVKeys.size}·${gWKeys.size}` };
  },
  compute({ labels, va, wa, elements, sizes }) {
    const idx = labelIndices(labels);
    const vPos = va.map((l) => idx[l]);
    const wPos = wa.map((l) => idx[l]);
    let pvSizes = 1;
    for (const p of vPos) pvSizes *= sizes[p];
    const wElements = elements.map((g) => restrictToW(g, wPos));
    const wSizes = wPos.map((p) => sizes[p]);
    const mW = sizeAwareBurnside(wElements, wSizes);
    const count = pvSizes * mW;
    return {
      count,
      latex: String.raw`A = \left(\prod_{\ell \in V} n_\ell\right) \cdot |[n]^W / G_W|`,
      latexSymbolic: String.raw`A = \left(\prod_{\ell \in V} n_\ell\right) \cdot |[n]^W / G_W|`,
    };
  },
};
