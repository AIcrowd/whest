import { MIXED_REGIMES } from './regimes/index.js';

/**
 * Mixed accumulation regimes all compute the same output-orbit alpha
 *   alpha = #{(O, Q) ∈ X/G × Y/H : pi_V(O) ∩ Q ≠ ∅},  H = Stab_G(V)|_V.
 *
 * Closed forms run first. If none apply, the engine prefers typed partition
 * counting when its budget passes; otherwise it falls back to corrected
 * brute-force enumeration when the tuple-enumeration budget passes. Both
 * general counters are exact and must agree whenever both are feasible.
 *
 * Order in MIXED_REGIMES:
 *   1. singleton         — closed form for |V| = 1 (point-stabilizer Burnside).
 *   2. young             — closed form for full Sym(L) with one shared dimension.
 *   3. partitionCount    — typed equality-pattern enumeration (compressed).
 *   4. bruteForceOrbit   — explicit fallback when partition counting refuses.
 */
export function getMixedRegimes() {
  return [...MIXED_REGIMES];
}
