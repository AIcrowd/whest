// Typed partition counting utilities.
//
// A typed equality pattern is a partition of label positions where blocks
// may only merge positions with the same domain class (here, the same
// numeric size n_s). The induced block action Ḡ_{x̃} is the IMAGE of the
// partition stabilizer Stab_G(x̃) acting on the blocks of x̃ — not the raw
// stabilizer order. Generators that permute positions within a block act
// trivially on blocks, so |Ḡ_{x̃}| ≤ |Stab_G(x̃)|. Using the wrong one
// breaks the integer division ∏_s (n_s)_{b_s(x̃)} / |Ḡ_{x̃}|.

import { applyPermutationToTupleArray } from '../outputOrbit.js';

export function fallingFactorial(n, m) {
  if (m < 0) throw new Error(`fallingFactorial received negative m=${m}`);
  if (m > n) return 0;
  let result = 1;
  for (let i = 0; i < m; i += 1) result *= (n - i);
  return result;
}

export function normalizePartition(partition) {
  const remap = new Map();
  let next = 0;
  return partition.map((block) => {
    if (!remap.has(block)) {
      remap.set(block, next);
      next += 1;
    }
    return remap.get(block);
  });
}

export function partitionKey(partition) {
  return normalizePartition(partition).join('|');
}

export function numBlocks(partition) {
  return new Set(partition).size;
}

export function blockDomains(partition, sizes) {
  const domains = new Map();
  partition.forEach((block, position) => {
    const domain = sizes[position];
    if (domains.has(block) && domains.get(block) !== domain) {
      throw new Error(`partition block ${block} mixes dimensions ${domains.get(block)} and ${domain}`);
    }
    domains.set(block, domain);
  });
  return domains;
}

export function typedLabelingCount(partition, sizes) {
  const domains = blockDomains(partition, sizes);
  const countsByDomain = new Map();
  for (const domain of domains.values()) {
    countsByDomain.set(domain, (countsByDomain.get(domain) ?? 0) + 1);
  }

  let result = 1;
  for (const [domainSize, blockCount] of countsByDomain.entries()) {
    result *= fallingFactorial(domainSize, blockCount);
  }
  return result;
}

export function generateTypedSetPartitions(sizes) {
  const results = [];
  const current = [];

  function visit(position, blockCount) {
    if (position === sizes.length) {
      results.push(normalizePartition(current));
      return;
    }

    for (let block = 0; block < blockCount; block += 1) {
      const firstPositionInBlock = current.findIndex((candidate) => candidate === block);
      if (firstPositionInBlock >= 0 && sizes[firstPositionInBlock] === sizes[position]) {
        current[position] = block;
        visit(position + 1, blockCount);
      }
    }

    current[position] = blockCount;
    visit(position + 1, blockCount + 1);
    current.length = position;
  }

  visit(0, 0);

  const byKey = new Map();
  for (const partition of results) byKey.set(partitionKey(partition), partition);
  return [...byKey.values()];
}

export function applyPermutationToPartition(partition, perm) {
  const moved = new Array(partition.length);
  for (let source = 0; source < partition.length; source += 1) {
    moved[perm.arr[source]] = partition[source];
  }
  return normalizePartition(moved);
}

export function partitionOrbitReps(partitions, elements) {
  const remaining = new Map(partitions.map((partition) => [partitionKey(partition), partition]));
  const reps = [];

  for (const [key, partition] of [...remaining.entries()]) {
    if (!remaining.has(key)) continue;
    reps.push(partition);
    for (const element of elements) {
      const moved = applyPermutationToPartition(partition, element);
      remaining.delete(partitionKey(moved));
    }
  }

  return reps;
}

export function inducedBlockPermutation(partition, perm) {
  const moved = applyPermutationToPartition(partition, perm);
  if (partitionKey(moved) !== partitionKey(partition)) return null;

  const representativeByBlock = new Map();
  partition.forEach((block, position) => {
    if (!representativeByBlock.has(block)) representativeByBlock.set(block, position);
  });

  const arr = [];
  const blocks = [...representativeByBlock.keys()].sort((a, b) => a - b);
  for (const block of blocks) {
    const sourcePosition = representativeByBlock.get(block);
    const targetPosition = perm.arr[sourcePosition];
    arr[block] = partition[targetPosition];
  }
  return arr.join('|');
}

export function inducedBlockActionSize(partition, elements) {
  const actions = new Set();
  for (const element of elements) {
    const actionKey = inducedBlockPermutation(partition, element);
    if (actionKey !== null) actions.add(actionKey);
  }
  return actions.size || 1;
}

export function inverseArray(perm) {
  const inv = new Array(perm.arr.length);
  perm.arr.forEach((target, source) => { inv[target] = source; });
  return inv;
}

export function mapKey(mapArray) {
  return mapArray.join('|');
}

export function inducedPrefixMap(partition, perm, visiblePositions) {
  const inv = inverseArray(perm);
  return visiblePositions.map((visiblePosition) => partition[inv[visiblePosition]]);
}

export function inducedPrefixMaps(partition, elements, visiblePositions) {
  const maps = new Set();
  for (const element of elements) {
    maps.add(mapKey(inducedPrefixMap(partition, element, visiblePositions)));
  }
  return maps;
}

export function mapArrayFromKey(key) {
  if (key === '') return [];
  return key.split('|').map((part) => Number(part));
}

export function actOnMapByOutputPermutation(mapArray, hElement) {
  return applyPermutationToTupleArray(mapArray, hElement);
}

export function canonicalMapUnderH(mapArray, hElements) {
  let best = null;
  for (const h of hElements) {
    const acted = actOnMapByOutputPermutation(mapArray, h);
    const key = mapKey(acted);
    if (best === null || key < best) best = key;
  }
  return best ?? mapKey(mapArray);
}

export function countMapOrbitsUnderH(mapKeys, hElements) {
  const remaining = new Set(mapKeys);
  let count = 0;
  for (const key of [...remaining]) {
    if (!remaining.has(key)) continue;
    count += 1;
    const mapArray = mapArrayFromKey(key);
    for (const h of hElements) {
      remaining.delete(mapKey(actOnMapByOutputPermutation(mapArray, h)));
    }
  }
  return count;
}
