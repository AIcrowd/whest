// website/components/symmetry-aware-einsum-contractions/engine/regimes/young.js
//
// Young regime: fires when G = Sym(L_c) (the full symmetric group on the
// component's labels), there's at least one cross-V/W element, and |V_c| ≥ 2.
//
// When G is full Sym(L) with cross elements, the V-pointwise-stabilizer is
// the Young subgroup Sym(W) — and α can be computed via a multinomial
// closed form instead of orbit enumeration.
//
// For |V_c| = 1, the singleton regime handles the case with its own closed
// form (equivalent; see Theorem 4 in the design doc). For no-cross cases,
// directProduct fires. The Young regime fills the |V_c| ≥ 2 cross-V/W
// full-Sym gap.
//
// Formula: α = n_L^|V| · C(n_L + |W| − 1, |W|)
//   where n_L is the common dimension of all component labels.

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
    const vPart = (nL) ** va.length;
    const wPart = multisetCount(nL, wa.length);
    return {
      count: vPart * wPart,
      latex: String.raw`A = n_L^{|V|} \cdot \binom{n_L + |W| - 1}{|W|}`,
      latexSymbolic: String.raw`A = |X / \mathrm{Stab}_G(V\text{ pointwise})|`,
    };
  },
};
