// website/components/symmetry-aware-einsum-contractions/engine/expressionGroup.js
//
// Given the pointwise group G_pt, compute the expression-level group
// G_expr = G_out × ∏_d S(W_d), where W_d ranges over summed-label blocks that
// are compatible in the engine's current label-domain model.
//
// Today that compatibility model is represented by label-size classes
// (`sizeByLabel` / cluster sizes), so only same-size summed labels share a
// dummy-renaming block.
//
// G_expr is strictly pedagogical/display-only. It does NOT feed into direct
// Burnside compression or accumulation counting. Direct compression uses G_pt.

import { Permutation } from './permutation.js';

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

function cartesianProduct(lists) {
  if (lists.length === 0) return [[]];
  const [head, ...rest] = lists;
  const tails = cartesianProduct(rest);
  const out = [];
  for (const item of head) {
    for (const tail of tails) out.push([item, ...tail]);
  }
  return out;
}

function readSize(sizeByLabel, label) {
  if (!sizeByLabel) return '__same-domain__';
  if (sizeByLabel instanceof Map) return sizeByLabel.get(label) ?? '__missing-size__';
  return Object.prototype.hasOwnProperty.call(sizeByLabel, label)
    ? sizeByLabel[label]
    : '__missing-size__';
}

function buildWBlocks(wLabels, sizeByLabel) {
  const blockByKey = new Map();
  for (let index = 0; index < wLabels.length; index += 1) {
    const label = wLabels[index];
    const key = String(readSize(sizeByLabel, label));
    if (!blockByKey.has(key)) {
      blockByKey.set(key, { key, labels: [], positions: [] });
    }
    const block = blockByKey.get(key);
    block.labels.push(label);
    block.positions.push(index);
  }
  return [...blockByKey.values()];
}

function enumerateBlockSymmetricW(wLabels, sizeByLabel) {
  const wBlocks = buildWBlocks(wLabels, sizeByLabel);
  const perBlockPerms = wBlocks.map((block) => allPermutations(block.positions.length));
  const combos = cartesianProduct(perBlockPerms);

  const sw = combos.map((combo) => {
    const arr = Array.from({ length: wLabels.length }, (_, i) => i);
    for (let blockIndex = 0; blockIndex < wBlocks.length; blockIndex += 1) {
      const block = wBlocks[blockIndex];
      const localPerm = combo[blockIndex];
      for (let localIndex = 0; localIndex < block.positions.length; localIndex += 1) {
        const from = block.positions[localIndex];
        const to = block.positions[localPerm[localIndex]];
        arr[from] = to;
      }
    }
    return new Permutation(arr);
  });

  return { sw, wBlocks };
}

/**
 * Build G_expr = G_out × ∏_d S(W_d) from G_pt.
 *
 * @param {object} args
 * @param {Permutation[]} args.perTupleElements G_pt elements over allLabels
 * @param {string[]} args.vLabels free labels
 * @param {string[]} args.wLabels summed labels
 * @param {string[]} args.allLabels full label list
 * @param {Map<string, number>|Record<string, number>=} args.sizeByLabel optional
 * label-domain lookup encoded by size classes in the current engine
 */
export function buildExpressionGroup({
  perTupleElements = [],
  vLabels = [],
  wLabels = [],
  allLabels = [],
  sizeByLabel = null,
}) {
  const N = allLabels.length;
  const vPos = vLabels.map((label) => allLabels.indexOf(label));
  const wPos = wLabels.map((label) => allLabels.indexOf(label));

  const vSubKeys = new Set();
  const vSub = [];
  for (const g of perTupleElements) {
    const arr = vPos.map((p) => vPos.indexOf(g.arr[p]));
    if (arr.some((x) => x < 0)) continue;
    const key = arr.join(',');
    if (vSubKeys.has(key)) continue;
    vSubKeys.add(key);
    vSub.push(new Permutation(arr));
  }

  const { sw, wBlocks } = enumerateBlockSymmetricW(wLabels, sizeByLabel);

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

  return { elements, vSub, sw, wBlocks, order: elements.length };
}
