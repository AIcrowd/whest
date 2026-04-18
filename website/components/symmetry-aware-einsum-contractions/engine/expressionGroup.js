// website/components/symmetry-aware-einsum-contractions/engine/expressionGroup.js
//
// Given the per-tuple group G_PT (materialized via Dimino), compute the
// expression-level group G_EXPR = V-sub × S(W).
//
// V-sub = G_PT projected onto V-labels (set of distinct V-restrictions,
//         always a subgroup of Sym(V)).
// S(W)  = full symmetric group on W-labels (|W|! permutations; every
//         permutation of the summed indices is a dummy-rename symmetry,
//         always an expression-level symmetry regardless of operand
//         structure).
// G_EXPR = Cartesian product, each pair lifted to a permutation of all labels.
//
// G_EXPR is strictly for pedagogy / display ("this einsum's counting
// symmetry"). It does NOT feed into Burnside compression — that uses
// G_PT. Conflating the two (which Source C effectively did) causes
// over-compression.

import { Permutation } from './permutation.js';

/**
 * Build G_EXPR = V-sub × S(W) from G_PT.
 *
 * @param {object} args
 * @param {Permutation[]} args.perTupleElements  G_PT elements over allLabels
 * @param {string[]}     args.vLabels             free labels
 * @param {string[]}     args.wLabels             summed labels
 * @param {string[]}     args.allLabels           full label list (indexing order)
 * @returns {{
 *   elements: Permutation[],  // full G_EXPR over allLabels
 *   vSub: Permutation[],      // distinct V-projections of G_PT (indexed over V only)
 *   sw: Permutation[],        // |W|! permutations indexed over W only
 *   order: number,
 * }}
 */
export function buildExpressionGroup({ perTupleElements = [], vLabels = [], wLabels = [], allLabels = [] }) {
  const N = allLabels.length;
  const vPos = vLabels.map((l) => allLabels.indexOf(l));
  const wPos = wLabels.map((l) => allLabels.indexOf(l));

  // V-sub: project each perTuple element onto V-positions; dedupe.
  const vSubKeys = new Set();
  const vSub = [];
  for (const g of perTupleElements) {
    const arr = vPos.map((p) => vPos.indexOf(g.arr[p]));
    // If any entry is -1, this element sends a V-label out of V (cross-V/W).
    // Skip it for the V-sub projection; the V-sub only captures the
    // V-preserving action.
    if (arr.some((x) => x < 0)) continue;
    const key = arr.join(',');
    if (vSubKeys.has(key)) continue;
    vSubKeys.add(key);
    vSub.push(new Permutation(arr));
  }

  // S(W): enumerate all permutations of wLabels.
  const sw = allPermutations(wLabels.length).map((p) => new Permutation(p));

  // G_EXPR: Cartesian product of vSub × sw, each pair lifted to a full-label
  // permutation that acts by vElem on V-positions and wElem on W-positions
  // (identity elsewhere — but our allLabels = V ∪ W so there is no elsewhere).
  const elements = [];
  for (const vElem of vSub) {
    for (const wElem of sw) {
      const arr = new Array(N);
      for (let i = 0; i < N; i += 1) arr[i] = i;
      for (let i = 0; i < vLabels.length; i += 1) {
        arr[vPos[i]] = vPos[vElem.arr[i]];
      }
      for (let j = 0; j < wLabels.length; j += 1) {
        arr[wPos[j]] = wPos[wElem.arr[j]];
      }
      elements.push(new Permutation(arr));
    }
  }

  return { elements, vSub, sw, order: elements.length };
}

/**
 * Enumerate all permutations of [0, n).
 * For n ≤ 6 this gives ≤ 720 results — adequate for |W| sizes seen in
 * the explorer's preset set (|W| ≤ 5 typically).
 */
function allPermutations(n) {
  if (n === 0) return [[]];
  const results = [];
  const cur = [];
  const used = new Array(n).fill(false);
  function rec() {
    if (cur.length === n) {
      results.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i += 1) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  }
  rec();
  return results;
}
