// website/components/symmetry-aware-einsum-contractions/engine/regimes/index.js
import { bruteForceOrbitRegime } from './bruteForceOrbit.js';
import { directProductRegime } from './directProduct.js';
import { singletonRegime } from './singleton.js';
import { youngRegime } from './young.js';

// Order rationale:
//  - singleton first — closed form for |V_c|=1 (covers both no-cross and
//    cross-V/W cases via the point-stabilizer formula).
//  - directProduct next — fires only if no cross element AND F-check
//    passes with non-trivial projections.
//  - young third — closed form for cross-V/W + |V_c|≥2 + G = Sym(L_c).
//    Multinomial Burnside on the V-pointwise-stabilizer (which is Sym(W)).
//  - bruteForceOrbit last — always-correct fallback, budget-gated.
export const MIXED_REGIMES = [
  singletonRegime,
  directProductRegime,
  youngRegime,
  bruteForceOrbitRegime,
];
