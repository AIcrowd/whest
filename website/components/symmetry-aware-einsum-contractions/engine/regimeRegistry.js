import { MIXED_REGIMES } from './regimes/index.js';

/**
 * Ordered list of mixed-shape regimes. Priority rationale:
 *
 *   1. singleton        — O(|G|) recognizer; closed form via weighted Burnside
 *                         (inclusion–exclusion on the free label's orbit).
 *   2. directProduct    — one pass over generators (checks V-only vs W-only
 *                         split); formula (Π_V n_ℓ) · |[n]^W / G_W|.
 *   3. bruteForceOrbit  — always correct; budget-capped at Π n_ℓ · |G| ≤ 1.5e6.
 *
 * Earlier entries are tried first. Overlapping recognizers must agree
 * numerically on their common domains (enforced by the property test).
 *
 * Earlier iterations exposed five extra analytic regimes between singleton
 * and bruteForceOrbit. They were removed to keep the classification surface
 * learnable; inputs that used to land there now fall through to
 * bruteForceOrbit, which is always correct within the budget cap.
 */
export function getMixedRegimes() {
  return [...MIXED_REGIMES];
}
