import { analyzeExample } from './pipeline.js';

function allAssignments(sizes) {
  const out = [];
  const cur = new Array(sizes.length).fill(0);
  function rec(idx) {
    if (idx === sizes.length) {
      out.push([...cur]);
      return;
    }
    for (let value = 0; value < sizes[idx]; value += 1) {
      cur[idx] = value;
      rec(idx + 1);
    }
  }
  if (sizes.length === 0) out.push([]);
  else rec(0);
  return out;
}

function tupleKey(tuple) {
  return tuple.join('|');
}

function applyPerm(tuple, perm) {
  const out = new Array(tuple.length);
  for (let i = 0; i < tuple.length; i += 1) out[perm.arr[i]] = tuple[i];
  return out;
}

function canonicalOutputOrbitKey(outputTuple, vSubElements) {
  const elements = vSubElements?.length ? vSubElements : [{ arr: outputTuple.map((_, i) => i) }];
  const keys = elements.map((elem) => tupleKey(applyPerm(outputTuple, elem)));
  keys.sort();
  return keys[0] ?? '';
}

function buildProductOrbits(assignments, elements) {
  const group = elements?.length ? elements : [{ arr: assignments[0]?.map((_, i) => i) ?? [] }];
  const seen = new Set();
  const orbits = [];

  for (const tuple of assignments) {
    const startKey = tupleKey(tuple);
    if (seen.has(startKey)) continue;

    const membersByKey = new Map();
    for (const elem of group) {
      const moved = applyPerm(tuple, elem);
      membersByKey.set(tupleKey(moved), moved);
    }

    for (const key of membersByKey.keys()) seen.add(key);
    orbits.push([...membersByKey.values()]);
  }

  return orbits;
}

function sizeArrayFromAnalysis(analysis) {
  const sizeByLabel = new Map();
  for (const cluster of analysis.clusters ?? []) {
    for (const label of cluster.labels) sizeByLabel.set(label, cluster.size);
  }
  return (analysis.symmetry?.allLabels ?? []).map((label) => sizeByLabel.get(label) ?? 1);
}

function latexSet(labels) {
  return labels.length ? labels.join(',') : String.raw`\varnothing`;
}

export function computeStorageSavingsRow(example, dimensionN = 3) {
  const analysis = analyzeExample(example, dimensionN);
  const labels = analysis.symmetry?.allLabels ?? [];
  const vLabels = analysis.symmetry?.vLabels ?? [];
  const vPositions = vLabels.map((label) => labels.indexOf(label));
  const sizes = sizeArrayFromAnalysis(analysis);
  const assignments = allAssignments(sizes);
  const productOrbits = buildProductOrbits(assignments, analysis.symmetry?.fullElements ?? []);
  const vSubElements = analysis.expressionGroup?.vSub ?? [];

  let alphaStorage = 0;
  for (const orbit of productOrbits) {
    const outputOrbitKeys = new Set();
    for (const tuple of orbit) {
      const outputTuple = vPositions.map((pos) => tuple[pos]);
      outputOrbitKeys.add(canonicalOutputOrbitKey(outputTuple, vSubElements));
    }
    alphaStorage += outputOrbitKeys.size;
  }

  const alphaEngine = analysis.componentCosts?.alpha ?? 0;
  const saving = Math.max(alphaEngine - alphaStorage, 0);
  const savingPctNumber = alphaEngine > 0 ? (saving / alphaEngine) * 100 : 0;

  return {
    id: example.id,
    vLatex: latexSet(vLabels),
    vSubLatex: vSubElements.length <= 1
      ? String.raw`\{e\}`
      : `\\text{order-}${vSubElements.length}`,
    alphaEngine,
    alphaStorage,
    saving,
    savingPct: savingPctNumber.toFixed(1).replace(/\.0$/, ''),
    savingPctNumber,
  };
}

export function buildStorageSavingsRows(examples, dimensionN = 3) {
  return examples
    .map((example) => computeStorageSavingsRow(example, dimensionN))
    .sort((a, b) => {
      if (b.savingPctNumber !== a.savingPctNumber) return b.savingPctNumber - a.savingPctNumber;
      return a.id.localeCompare(b.id);
    });
}
