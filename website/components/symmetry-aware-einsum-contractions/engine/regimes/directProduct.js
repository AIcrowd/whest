// website/components/symmetry-aware-einsum-contractions/engine/regimes/directProduct.js
import { sizeAwareBurnside } from '../sizeAware/burnside.js';
import { Permutation } from '../permutation.js';

function labelIndices(labels) {
  const idx = Object.create(null);
  labels.forEach((l, i) => { idx[l] = i; });
  return idx;
}

function movedPositions(perm) {
  const out = [];
  for (let i = 0; i < perm.arr.length; i += 1) {
    if (perm.arr[i] !== i) out.push(i);
  }
  return out;
}

function generatorIsVOnly(gen, vSet) {
  return movedPositions(gen).every((i) => vSet.has(i));
}

function generatorIsWOnly(gen, wSet) {
  return movedPositions(gen).every((i) => wSet.has(i));
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
      // Should never happen: generators accepted by recognize() never move W to non-W.
      throw new Error('directProduct.restrictToW: generator moves W out of W');
    }
    arr[i] = localImage;
  }
  return new Permutation(arr);
}

export const directProductRegime = {
  id: 'directProduct',
  recognize({ labels, va, wa, generators = [] }) {
    if (!generators.length) return { fired: false, reason: 'no generators provided' };
    const idx = labelIndices(labels);
    const vSet = new Set(va.map((l) => idx[l]));
    const wSet = new Set(wa.map((l) => idx[l]));
    for (const g of generators) {
      const vOnly = generatorIsVOnly(g, vSet);
      const wOnly = generatorIsWOnly(g, wSet);
      if (!vOnly && !wOnly) {
        return { fired: false, reason: `generator ${g.cycleNotation(labels)} crosses V/W` };
      }
    }
    return { fired: true, reason: 'every generator moves only V or only W' };
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
