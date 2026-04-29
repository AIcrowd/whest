// website/components/symmetry-aware-einsum-contractions/engine/regimes/index.js
//
// Mixed-shape regime ladder. The outer accumulationCount.js routes through
// functionalProjection FIRST (alpha = M when every g preserves V as a set),
// then falls through this ladder when projection can branch:
//
//   1. singleton         — closed form for |V| = 1.
//   2. young             — closed form for G = Sym(L) with uniform sizes.
//   3. partitionCount    — typed partition counting (heterogeneous-safe).
//   4. bruteForceOrbit   — corrected explicit enumeration; budget-gated.
//
// Earlier entries are cheaper. Overlapping regimes must agree numerically
// (verified by symmetry-explorer.partition-count.test.mjs and the sympy
// ground-truth oracle in .aicrowd/sympy-validation-scripts.zip).

import { bruteForceOrbitRegime } from './bruteForceOrbit.js';
import { partitionCountRegime } from './partitionCount.js';
import { singletonRegime } from './singleton.js';
import { youngRegime } from './young.js';

export const MIXED_REGIMES = [
  singletonRegime,
  youngRegime,
  partitionCountRegime,
  bruteForceOrbitRegime,
];

export function getMixedRegimes() {
  return [...MIXED_REGIMES];
}
