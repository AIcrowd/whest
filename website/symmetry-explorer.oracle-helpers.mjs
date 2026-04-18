// website/symmetry-explorer.oracle-helpers.mjs
//
// Shared helpers for orbit enumeration and ground-truth α/μ computation.
// Used by the cross-V/W-aware regression tests.

import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';

/**
 * Enumerate all tuples in [n]^length.
 */
export function allTuples(n, length) {
  const out = [];
  const cur = new Array(length).fill(0);
  function rec(i) {
    if (i === length) {
      out.push([...cur]);
      return;
    }
    for (let v = 0; v < n; v++) {
      cur[i] = v;
      rec(i + 1);
    }
  }
  if (length === 0) out.push([]);
  else rec(0);
  return out;
}

/**
 * Apply a Permutation g (on index positions) to a tuple t.
 * Convention: (g · t)[g.arr[i]] = t[i]. Mirrors bruteForceOrbit.js:applyPerm.
 */
export function applyPermToTuple(g, t) {
  const out = new Array(t.length);
  for (let i = 0; i < t.length; i++) out[g.arr[i]] = t[i];
  return out;
}

/**
 * Enumerate orbits of [n]^L under the group `elements` (array of Permutations).
 * Returns an array of orbits; each orbit is an array of tuples.
 */
export function enumerateOrbits(elements, n, labelCount) {
  const tuples = allTuples(n, labelCount);
  const keyOf = (t) => t.join('|');
  const seen = new Set();
  const orbits = [];
  for (const t of tuples) {
    if (seen.has(keyOf(t))) continue;
    const orbit = [];
    for (const g of elements) {
      const moved = applyPermToTuple(g, t);
      const k = keyOf(moved);
      if (!seen.has(k)) {
        seen.add(k);
        orbit.push(moved);
      }
    }
    if (orbit.length > 0) orbits.push(orbit);
  }
  return orbits;
}

/**
 * Ground-truth μ and α for a group acting on [n]^L.
 *   μ = number of orbits
 *   α = Σ_{O ∈ X/G} |π_V(O)|  (each orbit contributes one accumulation per
 *                                 distinct V-projection it touches)
 *
 * Requires uniform dimension across labels (asserts).
 */
export function muAlphaGroundTruth(elements, sizes, vPositions) {
  const n = sizes[0];
  for (const s of sizes) {
    if (s !== n) throw new Error('muAlphaGroundTruth: mixed sizes not supported');
  }
  const orbits = enumerateOrbits(elements, n, sizes.length);
  const mu = orbits.length;
  let alpha = 0;
  for (const O of orbits) {
    const vImages = new Set();
    for (const t of O) vImages.add(vPositions.map((p) => t[p]).join(','));
    alpha += vImages.size;
  }
  return { mu, alpha };
}

/**
 * Expected |G_pt| per preset from the Python prototype's oracle.
 * Source: /Users/mohanty/.claude/plans/sigma-prototype/alpha_mu_audit_v2_output.txt
 *
 * These are the group orders AFTER Source C removal. Presets not listed
 * here are either new (post-cleanup) or outside the original 15-preset set.
 */
export const EXPECTED_GROUP_ORDERS = {
  'matrix-chain': 1,
  'mixed-chain': 1,
  'triple-outer': 6,
  'outer': 2,
  'triangle': 3,
  'four-cycle': 8,
  'trace-product': 2,
  'frobenius': 1,
  'bilinear-trace': 2,
  'cross-s2': 2,
  'cross-s3': 6,
  'cyclic-cross': 3,
  'declared-c3': 3,
  'declared-d4': 8,
  'sym-tensor-fullV': 6,
};
