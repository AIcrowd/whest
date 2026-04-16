import { MIXED_REGIMES } from './regimes/index.js';

/**
 * Ordered list of mixed-shape regimes. Priority rationale:
 *
 *   1. singleton            — O(|G|) recognizer; closed form via weighted Burnside
 *                             (inclusion–exclusion on the free label's orbit).
 *   2. fullSymmetric        — |G| vs |L|! + has-transposition check; closed form
 *                             n^m · C(r+n-1, n-1) (Young shortcut).
 *   3. alternating          — |G| vs |L|!/2 + has-3-cycle + no-transposition;
 *                             base formula + injective-coloring correction.
 *   4. directProduct        — one pass over generators (checks V-only vs W-only
 *                             split); formula (Π_V n_ℓ) · |[n]^W / G_W|.
 *   5. wreath               — divisor search O(σ(|L|) · |G|); recognizes block
 *                             structure H ≀ S_b; formula n^(su) · C(h+t_H-1, t_H-1).
 *   6. diagonalSimultaneous — brute-force O(m!) pairing search, capped at m ≤ 4.
 *                             Below cap, composition-sum closed form.
 *   7. vSetwiseStable       — most general analytic reduction; iterates visible
 *                             orbits and sums |H·u|·|[n]^W/G_u| via Burnside
 *                             on stabilizers.
 *   8. bruteForceOrbit      — always correct; budget-capped at Π n_ℓ · |G| ≤ 1.5e6.
 *
 * Earlier entries are tried first. Overlapping recognizers must agree numerically
 * on their common domains (enforced by the property test).
 *
 * Changing this order changes only WHICH formula the user sees, not the final
 * count — all closed forms are mathematically equivalent where they apply.
 */
export function getMixedRegimes() {
  return [...MIXED_REGIMES];
}
