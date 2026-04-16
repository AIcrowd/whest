// website/components/symmetry-aware-einsum-contractions/engine/accumulationCount.js
import { detectShape } from './shapeLayer.js';
import { sizeAwareBurnside } from './sizeAware/burnside.js';
import { getMixedRegimes } from './regimeRegistry.js';

function productOver(positions, sizes) {
  let p = 1;
  for (const i of positions) p *= sizes[i];
  return p;
}

function vPositions(labels, va) {
  const idx = Object.create(null);
  labels.forEach((l, i) => { idx[l] = i; });
  return va.map((l) => idx[l]);
}

function piSizesProduct(va, sizes, labels) {
  return productOver(vPositions(labels, va), sizes);
}

export function computeAccumulation({
  labels, va, wa, elements, sizes, visiblePositions, generators,
}) {
  const shape = detectShape({ va, wa, elements }).kind;

  if (shape === 'trivial') {
    // No symmetry: each full assignment is its own orbit, |π_V(O)| = 1, so
    // A = |X| = Π n_ℓ. This matches Σ_O |π_V(O)| when every orbit is a singleton.
    const count = sizes.reduce((p, n) => p * n, 1);
    return {
      regimeId: 'trivial',
      count,
      latex: String.raw`A = \prod_{\ell \in L} n_\ell`,
      latexSymbolic: String.raw`A = |X|`,
      trace: [{ regimeId: 'trivial', decision: 'fired', reason: '|G| = 1' }],
    };
  }

  if (shape === 'allVisible') {
    const count = piSizesProduct(va, sizes, labels);
    return {
      regimeId: 'allVisible',
      count,
      latex: String.raw`A = \prod_{\ell \in V} n_\ell`,
      latexSymbolic: String.raw`A = \prod_{\ell \in V} n_\ell`,
      trace: [{ regimeId: 'allVisible', decision: 'fired', reason: 'W = ∅' }],
    };
  }

  if (shape === 'allSummed') {
    const count = sizeAwareBurnside(elements, sizes);
    return {
      regimeId: 'allSummed',
      count,
      latex: String.raw`A = \frac{1}{|G|} \sum_{g \in G} \prod_{c \in \mathrm{cycles}(g)} n_c`,
      latexSymbolic: String.raw`A = |X/G|`,
      trace: [{ regimeId: 'allSummed', decision: 'fired', reason: 'V = ∅' }],
    };
  }

  // Mixed: run the ladder.
  const trace = [];
  const ctx = { labels, va, wa, elements, sizes, visiblePositions, generators };
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
  // Should never reach here once bruteForceOrbit is installed as fallback.
  return {
    regimeId: null,
    count: null,
    latex: '',
    latexSymbolic: '',
    trace: [...trace, {
      regimeId: 'fallthrough',
      decision: 'refused',
      reason: 'no regime fired and no brute-force fallback installed',
    }],
  };
}
