// website/components/symmetry-aware-einsum-contractions/engine/wreath.js
//
// Direct enumerator for the wreath product ∏_i (H_i ≀ S_{m_i}).
//
// `i` indexes identical-operand groups (operands sharing the same name).
// `H_i` is each operand's declared axis symmetry group on its own axes.
// `m_i` is the number of copies of operand i.
//
// Emission order (maintained contract): top-major, base-minor lex. For
// each identical-group, we iterate S_{m_i} in lex order over its
// permutation arrays; for each outer element we iterate the base tuple
// (h_0, …, h_{m-1}) in lex order over `H_i`'s element array. Across
// identical-groups we take the cartesian product.

import { Permutation, dimino } from './permutation.js';
import { parseCycleNotation } from './cycleParser.js';

// Enumerate every element of a rank-n permutation group described by
// `sym` ('symmetric' | 'cyclic' | 'dihedral' | { type: 'custom', generators, axes }
//  | null/'none'/undefined). Returns an array of Permutation on [0, rank).
//
// `sym.generators` for the custom branch may be either:
//   - a raw cycle-notation string ("(0 1), (2 3)"), or
//   - the pre-parsed array-of-cycle-arrays shape that
//     `pipeline.js::analyzeExample` produces via `parseCycleNotation`.
// Both shapes are accepted so callers don't need to re-normalise.
export function enumerateH(sym, rank) {
  if (!sym || sym === 'none') return [Permutation.identity(rank)];

  const identity = Permutation.identity(rank);

  if (sym === 'symmetric' || sym?.type === 'symmetric') {
    const axes = sym?.axes || Array.from({ length: rank }, (_, i) => i);
    // Generators: adjacent transpositions on the axes
    const gens = [];
    for (let k = 0; k < axes.length - 1; k += 1) {
      const arr = Array.from({ length: rank }, (_, i) => i);
      arr[axes[k]] = axes[k + 1];
      arr[axes[k + 1]] = axes[k];
      gens.push(new Permutation(arr));
    }
    return gens.length === 0 ? [identity] : dimino(gens);
  }

  if (sym?.type === 'cyclic' || sym === 'cyclic') {
    const axes = sym?.axes || Array.from({ length: rank }, (_, i) => i);
    if (axes.length <= 1) return [identity];
    const arr = Array.from({ length: rank }, (_, i) => i);
    for (let k = 0; k < axes.length; k += 1) arr[axes[k]] = axes[(k + 1) % axes.length];
    return dimino([new Permutation(arr)]);
  }

  if (sym?.type === 'dihedral' || sym === 'dihedral') {
    const axes = sym?.axes || Array.from({ length: rank }, (_, i) => i);
    if (axes.length <= 2) return [identity];
    // Rotation generator + reflection generator
    const rot = Array.from({ length: rank }, (_, i) => i);
    for (let k = 0; k < axes.length; k += 1) rot[axes[k]] = axes[(k + 1) % axes.length];
    const ref = Array.from({ length: rank }, (_, i) => i);
    for (let k = 0; k < Math.floor(axes.length / 2); k += 1) {
      const a = axes[k];
      const b = axes[axes.length - 1 - k];
      ref[a] = b;
      ref[b] = a;
    }
    return dimino([new Permutation(rot), new Permutation(ref)]);
  }

  if (sym?.type === 'custom') {
    const axes = Array.isArray(sym.axes)
      ? sym.axes
      : Array.from({ length: rank }, (_, i) => i);

    if (axes.length === 0) return [identity];

    let generatorCycles;
    if (typeof sym.generators === 'string') {
      const parsed = parseCycleNotation(sym.generators);
      generatorCycles = parsed.generators;
    } else if (Array.isArray(sym.generators)) {
      generatorCycles = sym.generators;
    } else {
      generatorCycles = null;
    }
    if (!generatorCycles || generatorCycles.length === 0) return [identity];
    const gens = generatorCycles.map((perm) => {
      const arr = Array.from({ length: rank }, (_, i) => i);
      for (const cycle of perm) {
        for (let k = 0; k < cycle.length; k += 1) {
          const fromLocal = cycle[k];
          const toLocal = cycle[(k + 1) % cycle.length];
          const fromAxis = axes[fromLocal];
          const toAxis = axes[toLocal];
          arr[fromAxis] = toAxis;
        }
      }
      return new Permutation(arr);
    });

    return dimino(gens);
  }

  return [identity];
}

// Yield every permutation of [0, m) in lex order.
function* enumerateSymmetric(m) {
  const arr = Array.from({ length: m }, (_, i) => i);
  // Lex-order permutation iteration (Heap's alternative)
  function* permute(k) {
    if (k === m - 1) {
      yield arr.slice();
      return;
    }
    for (let i = k; i < m; i += 1) {
      [arr[k], arr[i]] = [arr[i], arr[k]];
      yield* permute(k + 1);
      [arr[k], arr[i]] = [arr[i], arr[k]];
    }
  }
  if (m === 0) { yield []; return; }
  yield* permute(0);
}

// Given a list of arrays, yield the cartesian product as arrays.
function* cartesianProduct(lists) {
  if (lists.length === 0) { yield []; return; }
  const [head, ...rest] = lists;
  for (const item of head) {
    for (const tail of cartesianProduct(rest)) {
      yield [item, ...tail];
    }
  }
}

// Flatten a wreath element `(baseTuple, topPerm)` for a single
// identical-group into a row-perm on U-vertex indices. `uOffsets[p]`
// is the starting U-vertex index for operand position p. `axisRanks[p]`
// is the rank of operand p.
//
// topPerm[j] = new position of copy j (so if topPerm[0]=1, copy 0 moves
// to position 1). baseTuple[j] is the axis-permutation h_j applied to
// copy j's axes BEFORE the top permutation maps j to its new slot.
//
// Net: U-vertex (operand_position p, axis a) moves to
//   (operand_position topPerm[positionOf(p)], axis baseTuple[j].arr[a]).
function flattenFactorToRowPerm(group, baseTuple, topPerm, uOffsets, axisRanks, nU) {
  const arr = Array.from({ length: nU }, (_, i) => i);
  for (let j = 0; j < group.length; j += 1) {
    const p = group[j];
    const rank = axisRanks[p];
    const newJ = topPerm[j];
    const newP = group[newJ];
    const h = baseTuple[j];
    for (let a = 0; a < rank; a += 1) {
      const from = uOffsets[p] + a;
      const to = uOffsets[newP] + h.arr[a];
      arr[to] = from;
    }
  }
  return arr;
}

// Compose row-perms from per-group factors. Each factor has already
// been written into a full-length array; compose by arr[i] = accumulated
// where we apply factors left-to-right. Since different identical-groups
// touch disjoint U-vertex ranges, composition is a single merge.
function mergeFactors(factorArrs, nU) {
  const out = Array.from({ length: nU }, (_, i) => i);
  for (const arr of factorArrs) {
    for (let i = 0; i < nU; i += 1) {
      if (arr[i] !== i) out[i] = arr[i];
    }
  }
  return out;
}

/**
 * Main entry point.
 *
 * @param {object} args
 * @param {number[][]} args.identicalGroups - array of groups; each group is
 *   an array of operand positions (0-indexed) that share the same operand
 *   name.
 * @param {Array<string|object|null>} args.perOpSymmetry - per-operand
 *   declared axis symmetry (same shape as pipeline.js produces).
 * @param {number[]} args.axisRanks - rank of each operand position.
 * @param {number[]=} args.uOffsets - starting U-vertex index for each
 *   operand. If omitted, computed as the running sum of axisRanks.
 *
 * @yields {{ rowPerm: Permutation, factorization: Array<{baseTuple: Permutation[], topPerm: number[]}> }}
 *   One object per wreath element, in top-major / base-minor lex order.
 */
export function* enumerateWreath({ identicalGroups, perOpSymmetry, axisRanks, uOffsets }) {
  const nOperands = axisRanks.length;
  const offsets = uOffsets || (() => {
    const o = new Array(nOperands);
    let acc = 0;
    for (let p = 0; p < nOperands; p += 1) { o[p] = acc; acc += axisRanks[p]; }
    return o;
  })();
  const nU = offsets[nOperands - 1] + axisRanks[nOperands - 1];

  // For each identical-group, build a per-group factor enumerator.
  const factorIterables = identicalGroups.map((group) => {
    const m = group.length;
    const rank = axisRanks[group[0]];
    const sym = perOpSymmetry[group[0]];
    const H = enumerateH(sym, rank);

    // Per-group wreath order = |H|^m · m!
    return (function* () {
      for (const topPerm of enumerateSymmetric(m)) {
        const baseTupleLists = new Array(m).fill(H);
        for (const baseTuple of cartesianProduct(baseTupleLists)) {
          const arr = flattenFactorToRowPerm(group, baseTuple, topPerm, offsets, axisRanks, nU);
          yield { arr, baseTuple, topPerm: topPerm.slice() };
        }
      }
    })();
  });

  // Cartesian across identical-groups. For small counts this is fine.
  // (In practice nGroups is ≤ 3 on all 22 presets.)
  const factorArrays = factorIterables.map((iter) => [...iter]);
  for (const combo of cartesianProduct(factorArrays)) {
    const arrs = combo.map((f) => f.arr);
    const merged = mergeFactors(arrs, nU);
    yield {
      rowPerm: new Permutation(merged),
      factorization: combo.map((f) => ({ baseTuple: f.baseTuple, topPerm: f.topPerm })),
    };
  }
}
