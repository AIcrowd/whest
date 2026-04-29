// website/components/symmetry-aware-einsum-contractions/engine/regimes/young.js
//
// Young regime: fires when G = Sym(L_c) (the full symmetric group on the
// component's labels), there's at least one cross-V/W element, and |V_c| ≥ 2.
//
// Under output-orbit accumulation, both the visible side and the summed side
// quotient down to multisets when G is the full symmetric group:
//
//   alpha = C(n + |V| - 1, |V|) * C(n + |W| - 1, |W|)
//
// The visible factor is |Y/H| (output representatives = unordered visible
// multisets); the summed factor is the number of summed-side multisets that
// independently combine with each visible multiset to form a product orbit.

function factorial(n) {
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

/**
 * Number of multisets of size k from n values: C(n + k − 1, k).
 * Equivalent to |[n]^k / Sym(k)| — orbits of Sym(k) on [n]^k.
 */
function multisetCount(n, k) {
  if (k === 0) return 1;
  let num = 1;
  let den = 1;
  for (let i = 0; i < k; i += 1) {
    num *= (n + k - 1 - i);
    den *= (i + 1);
  }
  return num / den;
}

function labelIndices(labels) {
  const idx = Object.create(null);
  labels.forEach((l, i) => { idx[l] = i; });
  return idx;
}

export const youngRegime = {
  id: 'young',
  recognize({ labels, va, wa, elements = [], sizes }) {
    if (!elements.length || elements.length <= 1) {
      return { fired: false, reason: '|G|<=1' };
    }
    if (va.length < 2) {
      return { fired: false, reason: '|V_c|<2; singleton handles this' };
    }
    // Must be full Sym(L_c): |G| = |L_c|!.
    const expectedFullSym = factorial(labels.length);
    if (elements.length !== expectedFullSym) {
      return { fired: false, reason: `|G|=${elements.length} ≠ |L_c|!=${expectedFullSym}` };
    }
    // Must have at least one cross-V/W element.
    const idx = labelIndices(labels);
    const vSet = new Set(va.map((l) => idx[l]));
    const hasCross = elements.some((g) => va.some((l) => !vSet.has(g.arr[idx[l]])));
    if (!hasCross) {
      return { fired: false, reason: 'no cross-V/W element' };
    }
    // All labels must share the same dimension.
    if (!sizes || sizes.length === 0) {
      return { fired: false, reason: 'no sizes provided' };
    }
    const nL = sizes[0];
    if (!sizes.every((s) => s === nL)) {
      return { fired: false, reason: 'mixed label sizes' };
    }
    return { fired: true, reason: 'G = Sym(L_c); Young equation applies' };
  },
  compute({ va, wa, sizes }) {
    const nL = sizes[0];
    const visibleMultisets = multisetCount(nL, va.length);
    const summedMultisets = multisetCount(nL, wa.length);
    return {
      count: visibleMultisets * summedMultisets,
      latex: String.raw`A = \binom{n_L + |V| - 1}{|V|}\binom{n_L + |W| - 1}{|W|}`,
      latexSymbolic: String.raw`A = |\mathrm{Multiset}_n(V)|\,|\mathrm{Multiset}_n(W)|`,
      subTrace: [{
        step: 'full-symmetric-output-orbit-formula',
        n: nL,
        vCount: va.length,
        wCount: wa.length,
        visibleMultisets,
        summedMultisets,
        count: visibleMultisets * summedMultisets,
      }],
    };
  },
};
