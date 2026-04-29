// website/components/symmetry-aware-einsum-contractions/engine/accumulationCount.js
import { functionalProjectionRegime } from './regimes/functionalProjection.js';
import { getMixedRegimes } from './regimeRegistry.js';

function productOverSizes(sizes) {
  return sizes.reduce((product, size) => product * size, 1);
}

export function computeAccumulation({
  labels, va, wa, elements, sizes, visiblePositions, generators,
}) {
  const trace = [];
  const ctx = { labels, va, wa, elements, sizes, visiblePositions, generators };

  if (!elements || elements.length === 0 || elements.length === 1) {
    const count = productOverSizes(sizes);
    return {
      regimeId: 'trivial',
      count,
      latex: String.raw`A = M = |X| = \prod_{\ell \in L} n_\ell`,
      latexSymbolic: String.raw`A = M`,
      trace: [{ regimeId: 'trivial', decision: 'fired', reason: '|G| = 1' }],
    };
  }

  const functionalVerdict = functionalProjectionRegime.recognize(ctx);
  if (functionalVerdict.fired) {
    const { count, latex, latexSymbolic, subTrace } = functionalProjectionRegime.compute(ctx);
    return {
      regimeId: functionalProjectionRegime.id,
      count,
      latex,
      latexSymbolic,
      trace: [{
        regimeId: functionalProjectionRegime.id,
        decision: 'fired',
        reason: functionalVerdict.reason,
        subSteps: subTrace,
      }],
    };
  }

  trace.push({
    regimeId: functionalProjectionRegime.id,
    decision: 'refused',
    reason: functionalVerdict.reason,
  });

  for (const regime of getMixedRegimes()) {
    const verdict = regime.recognize(ctx);
    if (!verdict.fired) {
      trace.push({ regimeId: regime.id, decision: 'refused', reason: verdict.reason });
      continue;
    }
    const { count, latex, latexSymbolic, subTrace } = regime.compute(ctx);
    trace.push({
      regimeId: regime.id,
      decision: 'fired',
      reason: verdict.reason,
      subSteps: subTrace,
    });
    return { regimeId: regime.id, count, latex, latexSymbolic, trace };
  }

  return {
    regimeId: null,
    count: null,
    latex: '',
    latexSymbolic: '',
    trace: [...trace, {
      regimeId: 'fallthrough',
      decision: 'refused',
      reason: 'no exact output-orbit accumulation regime fired inside the interactive budget',
    }],
  };
}
