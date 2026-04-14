function tupleKey(labels, tuple) {
  return labels.map((label) => tuple[label]).join('|');
}

function enumerateTuples(labels, dimensionN) {
  const tuples = [];
  const current = {};

  function visit(idx) {
    if (idx === labels.length) {
      tuples.push({ ...current });
      return;
    }

    const label = labels[idx];
    for (let value = 0; value < dimensionN; value++) {
      current[label] = value;
      visit(idx + 1);
    }
  }

  visit(0);
  return tuples;
}

function applyPiToTuple(tuple, labels, perm) {
  const next = {};
  for (let i = 0; i < labels.length; i++) {
    next[labels[perm.arr[i]]] = tuple[labels[i]];
  }
  return next;
}

function projectTuple(tuple, vLabels) {
  const projected = {};
  for (const label of vLabels) projected[label] = tuple[label];
  return projected;
}

export function computeExactCostModel({ labels, vLabels, groupElements, dimensionN, numTerms }) {
  const tuples = enumerateTuples(labels, dimensionN);
  const effectiveElements = groupElements.length > 0
    ? groupElements
    : [{ arr: Array.from({ length: labels.length }, (_, idx) => idx) }];
  const seen = new Set();
  const orbitRows = [];
  let reductionCost = 0;

  for (const tuple of tuples) {
    const repKey = tupleKey(labels, tuple);
    if (seen.has(repKey)) continue;

    const orbitMap = new Map();
    for (const element of effectiveElements) {
      const moved = applyPiToTuple(tuple, labels, element);
      orbitMap.set(tupleKey(labels, moved), moved);
    }

    for (const key of orbitMap.keys()) seen.add(key);

    const outputs = new Map();
    for (const moved of orbitMap.values()) {
      const outTuple = projectTuple(moved, vLabels);
      const outKey = tupleKey(vLabels, outTuple);
      const current = outputs.get(outKey) ?? { outTuple, coeff: 0 };
      current.coeff += 1;
      outputs.set(outKey, current);
    }

    const row = {
      repTuple: tuple,
      orbitSize: orbitMap.size,
      orbitTuples: [...orbitMap.values()],
      outputs: [...outputs.values()],
      outputCount: outputs.size,
    };
    orbitRows.push(row);
    reductionCost += row.outputCount;
  }

  const orbitCount = orbitRows.length;
  const evaluationCost = Math.max(numTerms - 1, 0) * orbitCount;

  return {
    orbitCount,
    evaluationCost,
    reductionCost,
    orbitRows,
  };
}
