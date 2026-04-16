// website/components/symmetry-aware-einsum-contractions/engine/regimes/index.js
import { alternatingRegime } from './alternating.js';
import { bruteForceOrbitRegime } from './bruteForceOrbit.js';
import { directProductRegime } from './directProduct.js';
import { fullSymmetricRegime } from './fullSymmetric.js';
import { singletonRegime } from './singleton.js';
import { wreathRegime } from './wreath.js';

export const MIXED_REGIMES = [
  singletonRegime,
  fullSymmetricRegime,
  alternatingRegime,
  directProductRegime,
  wreathRegime,
  bruteForceOrbitRegime,
];
