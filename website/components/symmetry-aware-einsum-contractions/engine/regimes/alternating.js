// website/components/symmetry-aware-einsum-contractions/engine/regimes/alternating.js

function factorial(n) {
  let p = 1;
  for (let i = 2; i <= n; i += 1) p *= i;
  return p;
}

function fallingFactorial(n, k) {
  let p = 1;
  for (let i = 0; i < k; i += 1) p *= (n - i);
  return p;
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let p = 1;
  for (let i = 1; i <= k; i += 1) p = (p * (n - k + i)) / i;
  return Math.round(p);
}

export const alternatingRegime = {
  id: 'alternating',
  recognize({ labels, elements, sizes }) {
    const L = labels.length;
    const expectedOrder = factorial(L) / 2;
    if (L < 3) {
      return { fired: false, reason: `L=${L} < 3; A_L requires L ≥ 3` };
    }
    if (elements.length !== expectedOrder) {
      return { fired: false, reason: `|G|=${elements.length} ≠ |L|!/2=${expectedOrder}` };
    }
    const n0 = sizes[0];
    for (const s of sizes) {
      if (s !== n0) return { fired: false, reason: 'heterogeneous sizes; alt requires uniform' };
    }
    const has3cycle = elements.some((g) => {
      const cycles = g.cyclicForm();
      return cycles.length === 1 && cycles[0].length === 3;
    });
    if (!has3cycle) return { fired: false, reason: 'no 3-cycle; group is not A_L' };
    const hasTransposition = elements.some((g) => {
      const cycles = g.cyclicForm();
      return cycles.length === 1 && cycles[0].length === 2;
    });
    if (hasTransposition) return { fired: false, reason: 'transposition present; group is S_L not A_L' };
    return { fired: true, reason: `G = A_L with uniform size n=${n0}` };
  },
  compute({ labels, va, wa, sizes }) {
    const n = sizes[0];
    const m = va.length;
    const r = wa.length;
    const N = labels.length;
    const base = Math.pow(n, m) * binomial(r + n - 1, n - 1);
    let correction = 0;
    if (n >= N && r >= 2) {
      correction = binomial(n - m, r) * fallingFactorial(n, m);
    }
    return {
      count: base + correction,
      latex: String.raw`A = n^m \binom{r + n - 1}{n - 1} + \mathbf{1}_{\{n \ge N,\, r \ge 2\}}\binom{n-m}{r}\, n^{\underline{m}}`,
      latexSymbolic: String.raw`A = n^m \binom{r + n - 1}{n - 1} + \mathbf{1}_{\{n \ge N,\, r \ge 2\}}\binom{n-m}{r}\, n^{\underline{m}}`,
    };
  },
};
