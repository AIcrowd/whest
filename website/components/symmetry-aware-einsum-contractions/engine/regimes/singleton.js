// website/components/symmetry-aware-einsum-contractions/engine/regimes/singleton.js

/**
 * Compute the G-orbit of a single label index.
 */
function labelOrbit(elements, labelIdx) {
  const seen = new Set([labelIdx]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const g of elements) {
      for (const p of [...seen]) {
        const q = g.arr[p];
        if (!seen.has(q)) { seen.add(q); changed = true; }
      }
    }
  }
  return [...seen].sort((a, b) => a - b);
}

/**
 * Count cycles of perm restricted to the given subset of indices.
 * Requires subset to be perm-invariant (guaranteed when G is a valid symmetry
 * and subset is a union of G-orbits, e.g. Ω or R = L \ Ω).
 */
function cyclesOnSubset(perm, subset) {
  const subsetSet = new Set(subset);
  const seen = new Set();
  let cycles = 0;
  for (const start of subset) {
    if (seen.has(start)) continue;
    cycles += 1;
    let cur = start;
    while (!seen.has(cur)) {
      if (!subsetSet.has(cur)) throw new Error('subset not invariant under perm');
      seen.add(cur);
      cur = perm.arr[cur];
    }
  }
  return cycles;
}

/**
 * Size-aware product of per-cycle common sizes, over cycles of perm restricted
 * to `subset`. Each cycle's labels must all have the same size (enforced).
 */
function subsetCycleProduct(perm, subset, sizes) {
  const subsetSet = new Set(subset);
  const seen = new Set();
  let product = 1;
  for (const start of subset) {
    if (seen.has(start)) continue;
    const cycle = [];
    let cur = start;
    while (!seen.has(cur)) {
      if (!subsetSet.has(cur)) throw new Error('subset not invariant under perm');
      seen.add(cur);
      cycle.push(cur);
      cur = perm.arr[cur];
    }
    const n0 = sizes[cycle[0]];
    for (const i of cycle) {
      if (sizes[i] !== n0) {
        throw new Error('singleton: cycle in R has mixed sizes');
      }
    }
    product *= n0;
  }
  return product;
}

export const singletonRegime = {
  id: 'singleton',
  recognize({ va }) {
    if (va.length === 1) return { fired: true, reason: '|V| = 1' };
    return { fired: false, reason: `|V| = ${va.length}, not 1` };
  },
  compute({ labels, elements, sizes, visiblePositions }) {
    const vPos = visiblePositions[0];
    const omega = labelOrbit(elements, vPos);
    const nOmega = sizes[vPos];
    for (const idx of omega) {
      if (sizes[idx] !== nOmega) {
        throw new Error(`singleton: orbit of label has mixed sizes at ${labels[idx]}`);
      }
    }
    const omegaSet = new Set(omega);
    const rest = labels.map((_, i) => i).filter((i) => !omegaSet.has(i));

    let total = 0;
    for (const g of elements) {
      const restFactor = subsetCycleProduct(g, rest, sizes);
      const cOmega = cyclesOnSubset(g, omega);
      total += restFactor * (Math.pow(nOmega, cOmega) - Math.pow(nOmega - 1, cOmega));
    }
    const count = (nOmega * total) / elements.length;
    return {
      count,
      latex: String.raw`A = \frac{n_\Omega}{|G|} \sum_{g \in G} \left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega - 1)^{c_\Omega(g)}\right)`,
      latexSymbolic: String.raw`A = \frac{n_\Omega}{|G|} \sum_{g \in G} \left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega - 1)^{c_\Omega(g)}\right)`,
    };
  },
};
