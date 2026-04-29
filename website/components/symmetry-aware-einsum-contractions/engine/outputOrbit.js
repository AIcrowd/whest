// Output-orbit helpers.
//
// The detected pointwise group G acts on full tuples in X = ∏_p [n_p].
// The output representative action is H = Stab_G(V)|_V — the elements of G
// that preserve the visible/output position set V, restricted to V using
// local coordinates. Note that |H| ≤ |Stab_G(V)|: the kernel of restriction
// is the pointwise V-stabilizer, so distinct g, g' ∈ Stab_G(V) can yield the
// same g|_V. We deduplicate via a key map.
//
// Permutation convention matches engine/permutation.js: source -> target
// arrays, tuple action is out[perm.arr[source]] = tuple[source].

import { Permutation } from './permutation.js';

export function tupleArrayKey(tuple) {
  return tuple.join('|');
}

export function preservesPositionSet(perm, positions) {
  const positionSet = new Set(positions);
  for (const source of positions) {
    if (!positionSet.has(perm.arr[source])) return false;
  }
  return true;
}

export function restrictToPositions(perm, positions) {
  if (!preservesPositionSet(perm, positions)) return null;
  const localIndex = new Map(positions.map((globalPosition, localPosition) => [
    globalPosition,
    localPosition,
  ]));
  const arr = positions.map((globalSource) => localIndex.get(perm.arr[globalSource]));
  return new Permutation(arr);
}

export function restrictStabilizerToPositions(elements, positions) {
  const degree = positions.length;
  if (degree === 0) return [Permutation.identity(0)];

  const byKey = new Map();
  for (const element of elements) {
    const restricted = restrictToPositions(element, positions);
    if (restricted !== null) byKey.set(restricted.key(), restricted);
  }

  if (byKey.size === 0) byKey.set(Permutation.identity(degree).key(), Permutation.identity(degree));
  return [...byKey.values()];
}

export function applyPermutationToTupleArray(tuple, perm) {
  const next = new Array(tuple.length);
  for (let source = 0; source < tuple.length; source += 1) {
    next[perm.arr[source]] = tuple[source];
  }
  return next;
}

export function canonicalTupleUnderGroup(tuple, elements) {
  if (!elements || elements.length === 0) return tupleArrayKey(tuple);
  let best = null;
  for (const element of elements) {
    const moved = applyPermutationToTupleArray(tuple, element);
    const key = tupleArrayKey(moved);
    if (best === null || key < best) best = key;
  }
  return best;
}

export function visibleTupleFromFullTuple(fullTuple, visiblePositions) {
  return visiblePositions.map((position) => fullTuple[position]);
}

export function projectionIsFunctional(elements, visiblePositions) {
  return elements.every((element) => preservesPositionSet(element, visiblePositions));
}
