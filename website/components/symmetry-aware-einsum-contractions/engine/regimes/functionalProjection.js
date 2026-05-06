import { projectionIsFunctional } from '../outputOrbit.js';
import { sizeAwareBurnside } from '../sizeAware/burnside.js';

export const functionalProjectionRegime = {
  id: 'functionalProjection',

  recognize({ elements, visiblePositions }) {
    if (!projectionIsFunctional(elements, visiblePositions)) {
      return {
        fired: false,
        reason: 'some pointwise symmetry moves an output label into a summed label',
      };
    }
    return {
      fired: true,
      reason: 'each product orbit reaches exactly one stored output representative',
    };
  },

  compute({ elements, sizes }) {
    const count = sizeAwareBurnside(elements, sizes);
    return {
      count,
      latex: String.raw`A = M = |X/G|`,
      latexSymbolic: String.raw`A = M`,
      subTrace: [{
        step: 'projection-functional',
        reason: 'G preserves the visible-label set, so projection descends to output representatives',
        count,
      }],
    };
  },
};
