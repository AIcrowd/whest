// website/components/symmetry-aware-einsum-contractions/engine/regimeRegistry.js
import { MIXED_REGIMES } from './regimes/index.js';

/**
 * Ordered list of mixed-shape regimes. Priority:
 *   1. singleton
 *   2. fullSymmetric
 *   3. alternating
 *   4. directProduct
 *   5. wreath
 *   6. diagonalSimultaneous
 *   7. vSetwiseStable
 *   8. bruteForceOrbit
 *
 * Earlier entries are tried first. Later entries must agree numerically with
 * earlier ones on overlapping preconditions (tested in the property suite).
 */
export function getMixedRegimes() {
  return [...MIXED_REGIMES];
}
