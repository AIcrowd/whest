// website/components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs

/**
 * Ground-truth oracle for A (accumulation count) and M (orbit count) with
 * heterogeneous label sizes. Port of followup_verification/generic_utils.py
 * used only from tests — not shipped in the app bundle.
 */

export function allAssignments(sizes) {
  const out = [];
  const current = new Array(sizes.length).fill(0);
  function visit(idx) {
    if (idx === sizes.length) {
      out.push([...current]);
      return;
    }
    for (let v = 0; v < sizes[idx]; v += 1) {
      current[idx] = v;
      visit(idx + 1);
    }
  }
  if (sizes.length === 0) out.push([]);
  else visit(0);
  return out;
}

export function applyPermToTuple(tuple, perm) {
  const n = tuple.length;
  const out = new Array(n);
  for (let i = 0; i < n; i += 1) {
    out[perm.arr[i]] = tuple[i];
  }
  return out;
}

function keyOf(arr) { return arr.join('|'); }

function orbitsOfAssignments(elements, sizes) {
  const remaining = new Map();
  for (const a of allAssignments(sizes)) remaining.set(keyOf(a), a);
  const orbits = [];
  while (remaining.size > 0) {
    const [, rep] = remaining.entries().next().value;
    const orbit = new Map();
    for (const g of elements) {
      const moved = applyPermToTuple(rep, g);
      const k = keyOf(moved);
      if (!orbit.has(k)) orbit.set(k, moved);
      remaining.delete(k);
    }
    orbits.push([...orbit.values()]);
  }
  return orbits;
}

export function computeMBruteforce(elements, sizes) {
  return orbitsOfAssignments(elements, sizes).length;
}

export function computeABruteforce(elements, sizes, visiblePositions) {
  let total = 0;
  for (const orbit of orbitsOfAssignments(elements, sizes)) {
    const projected = new Set();
    for (const a of orbit) {
      projected.add(visiblePositions.map((p) => a[p]).join('|'));
    }
    total += projected.size;
  }
  return total;
}
