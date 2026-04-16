// website/components/symmetry-aware-einsum-contractions/engine/regimes/index.js
import { bruteForceOrbitRegime } from './bruteForceOrbit.js';
import { directProductRegime } from './directProduct.js';

export const MIXED_REGIMES = [
  directProductRegime,
  bruteForceOrbitRegime,
];
