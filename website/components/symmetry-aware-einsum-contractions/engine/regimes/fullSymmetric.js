// website/components/symmetry-aware-einsum-contractions/engine/regimes/fullSymmetric.js

function factorial(n) {
  let p = 1;
  for (let i = 2; i <= n; i += 1) p *= i;
  return p;
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let p = 1;
  for (let i = 1; i <= k; i += 1) p = (p * (n - k + i)) / i;
  return Math.round(p);
}

export const fullSymmetricRegime = {
  id: 'fullSymmetric',
  recognize({ labels, elements, sizes }) {
    const L = labels.length;
    if (elements.length !== factorial(L)) {
      return { fired: false, reason: `|G|=${elements.length} ≠ |L|!=${factorial(L)}` };
    }
    const n0 = sizes[0];
    for (const s of sizes) {
      if (s !== n0) {
        return { fired: false, reason: `heterogeneous sizes; full-sym requires uniform support` };
      }
    }
    const hasTransposition = elements.some((g) => {
      const cycles = g.cyclicForm();
      return cycles.length === 1 && cycles[0].length === 2;
    });
    if (!hasTransposition) {
      return { fired: false, reason: 'no transposition found; group is not S_L' };
    }
    return { fired: true, reason: `G = S_L with uniform size n=${n0}` };
  },
  compute({ va, wa, sizes }) {
    const n = sizes[0];
    const m = va.length;
    const r = wa.length;
    const count = Math.pow(n, m) * binomial(r + n - 1, n - 1);
    return {
      count,
      latex: String.raw`A = n^m \binom{r + n - 1}{n - 1}`,
      latexSymbolic: String.raw`A = n^m \binom{r + n - 1}{n - 1}`,
    };
  },
};
