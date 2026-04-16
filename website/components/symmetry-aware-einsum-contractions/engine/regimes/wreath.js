// website/components/symmetry-aware-einsum-contractions/engine/regimes/wreath.js
import { sizeAwareBurnside } from '../sizeAware/burnside.js';
import { Permutation, dimino } from '../permutation.js';

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let p = 1;
  for (let i = 1; i <= k; i += 1) p = (p * (n - k + i)) / i;
  return Math.round(p);
}

function divisors(n) {
  const out = [];
  for (let d = 1; d <= n; d += 1) if (n % d === 0) out.push(d);
  return out;
}

/**
 * Check if `gen` is a "pure block-swap" that swaps two specific blocks pointwise
 * (same position-offset within each block) and fixes all other blocks.
 * Returns { swapsBlocks: [b1, b2] } or null.
 */
function generatorIsBlockSwap(gen, s, b) {
  const cycles = gen.cyclicForm();
  if (cycles.length !== s) return null;
  const pairs = [];
  for (const c of cycles) {
    if (c.length !== 2) return null;
    const [p1, p2] = [...c].sort((a, b2) => a - b2);
    const block1 = Math.floor(p1 / s);
    const block2 = Math.floor(p2 / s);
    const off1 = p1 % s;
    const off2 = p2 % s;
    if (off1 !== off2) return null;
    pairs.push([block1, block2]);
  }
  const firstPair = pairs[0];
  for (const pr of pairs) {
    if (pr[0] !== firstPair[0] || pr[1] !== firstPair[1]) return null;
  }
  return { swapsBlocks: firstPair };
}

/**
 * Check if `gen` acts the same local permutation π within every block AND
 * doesn't move any block to another.
 * Returns { pi: Permutation } or null.
 */
function generatorIsBlockInternal(gen, s, b) {
  const n = s * b;
  if (gen.arr.length !== n) return null;
  const pi = new Array(s);
  // Derive π from block 0.
  for (let j = 0; j < s; j += 1) {
    const target = gen.arr[j];
    const block = Math.floor(target / s);
    const off = target % s;
    if (block !== 0) return null;
    pi[j] = off;
  }
  // Verify the same π on every other block.
  for (let bIdx = 1; bIdx < b; bIdx += 1) {
    for (let j = 0; j < s; j += 1) {
      const expected = bIdx * s + pi[j];
      if (gen.arr[bIdx * s + j] !== expected) return null;
    }
  }
  return { pi: new Permutation(pi) };
}

/**
 * Check if `gen` acts internally on exactly ONE block (fixes all others pointwise,
 * and doesn't move elements between blocks).
 * Returns { pi: Permutation } or null.
 */
function generatorIsLocalBlockInternal(gen, s, b) {
  const arr = gen.arr;
  let activeBlock = -1;
  for (let bIdx = 0; bIdx < b; bIdx += 1) {
    let identity = true;
    for (let j = 0; j < s; j += 1) {
      if (arr[bIdx * s + j] !== bIdx * s + j) { identity = false; break; }
    }
    if (!identity) {
      if (activeBlock !== -1) return null; // acts on more than one block
      activeBlock = bIdx;
    }
  }
  if (activeBlock === -1) return null; // identity permutation
  // Confirm active block maps internally.
  const pi = new Array(s);
  for (let j = 0; j < s; j += 1) {
    const target = arr[activeBlock * s + j];
    const block = Math.floor(target / s);
    const off = target % s;
    if (block !== activeBlock) return null;
    pi[j] = off;
  }
  return { pi: new Permutation(pi) };
}

export const wreathRegime = {
  id: 'wreath',
  recognize({ labels, va, generators = [], sizes }) {
    const L = labels.length;
    if (!generators.length) return { fired: false, reason: 'no generators' };

    // Uniform-size requirement.
    const n0 = sizes[0];
    for (const s of sizes) {
      if (s !== n0) return { fired: false, reason: 'heterogeneous sizes; wreath requires uniform support' };
    }

    // Try each divisor `s` of L.
    for (const s of divisors(L)) {
      const b = L / s;
      if (b < 2) continue;

      const baseGens = [];
      const blockSwaps = new Set();
      let ok = true;

      for (const g of generators) {
        const intr = generatorIsBlockInternal(g, s, b);
        if (intr) {
          baseGens.push(intr.pi);
          continue;
        }
        const local = generatorIsLocalBlockInternal(g, s, b);
        if (local) {
          baseGens.push(local.pi);
          continue;
        }
        const bs = generatorIsBlockSwap(g, s, b);
        if (bs) {
          const [x, y] = bs.swapsBlocks;
          blockSwaps.add(`${Math.min(x, y)},${Math.max(x, y)}`);
          continue;
        }
        ok = false;
        break;
      }
      if (!ok) continue;
      // Need at least b-1 distinct adjacent-ish block swaps to generate S_b.
      if (blockSwaps.size < b - 1) continue;

      // V must be a union of whole blocks.
      const labelIdx = Object.create(null);
      labels.forEach((l, i) => { labelIdx[l] = i; });
      const vPositions = new Set(va.map((l) => labelIdx[l]));
      let uBlocks = 0;
      let allWhole = true;
      for (let bIdx = 0; bIdx < b; bIdx += 1) {
        let inV = 0;
        for (let j = 0; j < s; j += 1) {
          if (vPositions.has(bIdx * s + j)) inV += 1;
        }
        if (inV === s) uBlocks += 1;
        else if (inV !== 0) { allWhole = false; break; }
      }
      if (!allWhole) continue;

      return {
        fired: true,
        reason: `H ≀ S_${b} with s=${s}, u=${uBlocks}`,
        s,
        b,
        u: uBlocks,
        baseGens,
      };
    }

    return { fired: false, reason: 'no (s, b) block structure recognized' };
  },
  compute(ctx) {
    const verdict = this.recognize(ctx);
    if (!verdict.fired) {
      throw new Error('wreathRegime.compute called without recognize firing');
    }
    const { s, b, u, baseGens } = verdict;
    const { sizes } = ctx;
    const n = sizes[0];
    const h = b - u;

    // Compute t_H(n) = |[n]^s / H| via size-aware Burnside on H, with blockSizes = [n, n, ..., n].
    const H = baseGens.length > 0 ? dimino(baseGens) : [Permutation.identity(s)];
    const blockSizes = Array.from({ length: s }, () => n);
    const tH = sizeAwareBurnside(H, blockSizes);

    const count = Math.pow(n, s * u) * binomial(h + tH - 1, tH - 1);

    return {
      count,
      latex: String.raw`A = n^{su} \binom{h + t_H(n) - 1}{t_H(n) - 1}`,
      latexSymbolic: String.raw`A = n^{su} \binom{h + t_H(n) - 1}{t_H(n) - 1}`,
    };
  },
};
