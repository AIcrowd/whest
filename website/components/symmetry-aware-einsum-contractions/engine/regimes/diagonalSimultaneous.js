// website/components/symmetry-aware-einsum-contractions/engine/regimes/diagonalSimultaneous.js

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let p = 1;
  for (let i = 1; i <= k; i += 1) p = (p * (n - k + i)) / i;
  return Math.round(p);
}

function* compositions(total, parts) {
  if (parts === 1) { yield [total]; return; }
  for (let first = 0; first <= total; first += 1) {
    for (const tail of compositions(total - first, parts - 1)) yield [first, ...tail];
  }
}

function multinomial(counts) {
  const total = counts.reduce((a, b) => a + b, 0);
  let p = 1;
  let remaining = total;
  for (const c of counts) {
    p *= binomial(remaining, c);
    remaining -= c;
  }
  return p;
}

function labelIndex(labels) {
  const m = Object.create(null);
  labels.forEach((l, i) => { m[l] = i; });
  return m;
}

function* permsOf(arr) {
  if (arr.length <= 1) { yield [...arr]; return; }
  for (let i = 0; i < arr.length; i += 1) {
    const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
    for (const p of permsOf(rest)) yield [arr[i], ...p];
  }
}

/**
 * Try to find a pairing W-positions such that for every g ∈ G, the induced V-perm
 * matches the induced W-perm (i.e., the action is diagonal S_m). Returns the paired
 * W-positions (in the same order as V) or null.
 */
function findDiagonalPairing(vPos, wPos, elements) {
  if (vPos.length !== wPos.length) return null;
  const m = vPos.length;
  if (m > 4) return null; // brute-force O(m!) capped.
  for (const pairedW of permsOf(wPos)) {
    let ok = true;
    for (const g of elements) {
      for (let i = 0; i < m; i += 1) {
        const vImage = g.arr[vPos[i]];
        const wImage = g.arr[pairedW[i]];
        const vImageIdx = vPos.indexOf(vImage);
        const wImageIdx = pairedW.indexOf(wImage);
        if (vImageIdx === -1 || wImageIdx === -1 || vImageIdx !== wImageIdx) {
          ok = false; break;
        }
      }
      if (!ok) break;
    }
    if (ok) return pairedW;
  }
  return null;
}

export const diagonalSimultaneousRegime = {
  id: 'diagonalSimultaneous',
  recognize({ labels, va, wa, elements, sizes }) {
    if (va.length !== wa.length) return { fired: false, reason: `|V|=${va.length} ≠ |W|=${wa.length}` };
    const idx = labelIndex(labels);
    const vPos = va.map((l) => idx[l]);
    const wPos = wa.map((l) => idx[l]);
    const nV = sizes[vPos[0]];
    for (const p of vPos) if (sizes[p] !== nV) return { fired: false, reason: 'V has heterogeneous sizes' };
    const nW = sizes[wPos[0]];
    for (const p of wPos) if (sizes[p] !== nW) return { fired: false, reason: 'W has heterogeneous sizes' };
    const pairing = findDiagonalPairing(vPos, wPos, elements);
    if (!pairing) return { fired: false, reason: 'no diagonal V↔W pairing works' };
    return { fired: true, reason: `diagonal S_m pairing found (m=${va.length})` };
  },
  compute({ labels, va, wa, sizes }) {
    const idx = labelIndex(labels);
    const m = va.length;
    const nV = sizes[idx[va[0]]];
    const nW = sizes[idx[wa[0]]];
    let total = 0;
    for (const counts of compositions(m, nV)) {
      let term = multinomial(counts);
      for (const c of counts) term *= binomial(c + nW - 1, nW - 1);
      total += term;
    }
    return {
      count: total,
      latex: String.raw`A = \sum_{c_1+\cdots+c_{n_V}=m} \frac{m!}{\prod c_a!} \prod_{a=1}^{n_V} \binom{c_a + n_W - 1}{n_W - 1}`,
      latexSymbolic: String.raw`A = \sum_{c_1+\cdots+c_{n_V}=m} \binom{m}{c_1,\ldots,c_{n_V}} \prod_{a=1}^{n_V} \binom{c_a + n_W - 1}{n_W - 1}`,
    };
  },
};
