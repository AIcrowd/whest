// website/components/symmetry-aware-einsum-contractions/engine/comparisonAlpha.js
//
// Pedagogical helpers for the appendix chapter that contrasts the pointwise
// compression group G_pt with the larger formal group G_f = V-sub × S(W).
//
// `computeExpressionAlphaTotal` preserves the existing scalar helper:
//   what α would be if one naively counted orbits under G_f instead of G_pt.
//
// `computeExpressionAlphaComparison` adds the chapter-ready comparison object:
//   - exprAlpha:   naive α under G_f
//   - correctAlpha: true α under G_pt
//   - state:       none | coincident | mismatch
//   - witness:     one concrete bad formal orbit witness when a real mismatch
//                  exists for the current preset

function applyPerm(tuple, perm) {
  const out = new Array(tuple.length);
  for (let i = 0; i < tuple.length; i += 1) out[perm.arr[i]] = tuple[i];
  return out;
}

function allAssignments(sizes) {
  const out = [];
  const cur = new Array(sizes.length).fill(0);
  function rec(idx) {
    if (idx === sizes.length) {
      out.push([...cur]);
      return;
    }
    for (let v = 0; v < sizes[idx]; v += 1) {
      cur[idx] = v;
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

function getCorrectAlpha(analysis) {
  if (typeof analysis?.componentCosts?.alpha === 'number') return analysis.componentCosts.alpha;
  const components = analysis?.componentData?.components ?? [];
  return components.reduce((sum, component) => sum + (component.accumulation?.count ?? 0), 0);
}

function getLabelContext(analysis) {
  const perTupleGroup = analysis?.symmetry ?? null;
  const globalLabels = perTupleGroup?.allLabels ?? [];
  const vLabels = perTupleGroup?.vLabels ?? [];
  const clusters = analysis?.clusters ?? [];
  const sizeByLabel = new Map();
  for (const cluster of clusters) {
    for (const label of cluster.labels) sizeByLabel.set(label, cluster.size);
  }
  const sizes = globalLabels.map((label) => sizeByLabel.get(label) ?? 1);
  const vPos = vLabels.map((label) => globalLabels.indexOf(label));
  return {
    globalLabels,
    vLabels,
    sizes,
    vPos,
  };
}

function buildOrbitPartition(assignments, elements) {
  const keyToTuple = new Map(assignments.map((tuple) => [tupleKey(tuple), tuple]));
  const orbitIdByKey = new Map();
  const orbits = [];

  for (const tuple of assignments) {
    const key = tupleKey(tuple);
    if (orbitIdByKey.has(key)) continue;

    const local = new Map();
    for (const element of elements) {
      const moved = applyPerm(tuple, element);
      local.set(tupleKey(moved), moved);
    }

    const memberKeys = [...local.keys()].sort();
    for (const memberKey of memberKeys) {
      if (!keyToTuple.has(memberKey)) {
        throw new Error(
          `group action mapped tuple outside the assignment domain: ${memberKey}`,
        );
      }
    }
    const orbitId = memberKeys[0];
    const members = memberKeys.map((memberKey) => keyToTuple.get(memberKey));
    for (const memberKey of memberKeys) orbitIdByKey.set(memberKey, orbitId);
    orbits.push({ id: orbitId, members });
  }

  orbits.sort((a, b) => a.id.localeCompare(b.id));
  return { orbitIdByKey, orbits };
}

function normalizeExampleExpression(example) {
  if (!example) return { operandNames: [], operandSubscripts: [] };

  if (Array.isArray(example.subscripts) && Array.isArray(example.operandNames)) {
    return {
      operandNames: example.operandNames,
      operandSubscripts: example.subscripts,
    };
  }

  const expression = example.expression ?? {};
  return {
    operandNames: typeof expression.operandNames === 'string'
      ? expression.operandNames.split(',').map((part) => part.trim()).filter(Boolean)
      : [],
    operandSubscripts: typeof expression.subscripts === 'string'
      ? expression.subscripts.split(',').map((part) => part.trim()).filter(Boolean)
      : [],
  };
}

function buildSummandFactors(example, globalLabels, tuple) {
  const { operandNames, operandSubscripts } = normalizeExampleExpression(example);
  const labelPos = new Map(globalLabels.map((label, idx) => [label, idx]));

  return operandSubscripts.map((subscript, idx) => {
    const operandName = operandNames[idx] ?? `X${idx}`;
    const coords = [...subscript].map((label) => tuple[labelPos.get(label)] ?? '?');
    return `${operandName}[${coords.join(',')}]`;
  });
}

function buildSummandString(example, globalLabels, tuple) {
  return buildSummandFactors(example, globalLabels, tuple).join(' · ');
}

function productSignature(example, globalLabels, tuple) {
  return buildSummandFactors(example, globalLabels, tuple)
    .slice()
    .sort()
    .join(' · ');
}

function compareTupleLex(a, b) {
  return tupleKey(a).localeCompare(tupleKey(b));
}

function toOutputTuple(tuple, vPos) {
  return vPos.map((pos) => tuple[pos]);
}

function buildWitness({ analysis, example, exprPartition }) {
  const expressionGroup = analysis?.expressionGroup ?? null;
  const perTupleGroup = analysis?.symmetry ?? null;
  if (!expressionGroup?.elements?.length || !perTupleGroup?.fullElements?.length || !example) return null;

  const { globalLabels, sizes, vPos } = getLabelContext(analysis);
  if (!globalLabels.length) return null;

  const assignments = allAssignments(sizes);
  const ptPartition = buildOrbitPartition(assignments, perTupleGroup.fullElements);

  for (const exprOrbit of exprPartition.orbits) {
    const buckets = new Map();
    for (const tuple of exprOrbit.members.slice().sort(compareTupleLex)) {
      const outputKey = tupleKey(toOutputTuple(tuple, vPos));
      const group = buckets.get(outputKey) ?? [];
      group.push(tuple);
      buckets.set(outputKey, group);
    }

    for (const outputKey of [...buckets.keys()].sort()) {
      const tuples = buckets.get(outputKey) ?? [];
      const variants = tuples.map((tuple) => ({
        tuple,
        ptOrbitId: ptPartition.orbitIdByKey.get(tupleKey(tuple)),
        summand: buildSummandString(example, globalLabels, tuple),
        signature: productSignature(example, globalLabels, tuple),
      }));

      for (let i = 0; i < variants.length; i += 1) {
        for (let j = i + 1; j < variants.length; j += 1) {
          const left = variants[i];
          const right = variants[j];
          if (left.ptOrbitId === right.ptOrbitId) continue;
          if (left.signature === right.signature) continue;
          return {
            tupleA: left.tuple,
            tupleB: right.tuple,
            summandA: left.summand,
            summandB: right.summand,
            outputA: toOutputTuple(left.tuple, vPos),
            outputB: toOutputTuple(right.tuple, vPos),
          };
        }
      }
    }
  }

  return null;
}

function computeExpressionAlphaData({ analysis }) {
  const expressionGroup = analysis?.expressionGroup ?? null;
  const perTupleGroup = analysis?.symmetry ?? null;
  if (!expressionGroup?.elements?.length) return { exprAlpha: null, exprPartition: null };

  const gExprOrder = expressionGroup.elements.length;
  const gPtOrder = perTupleGroup?.fullElements?.length ?? 1;
  if (gExprOrder === gPtOrder) return { exprAlpha: null, exprPartition: null };

  const { globalLabels, sizes, vPos } = getLabelContext(analysis);
  if (!globalLabels.length) return { exprAlpha: null, exprPartition: null };

  const assignments = allAssignments(sizes);
  const exprPartition = buildOrbitPartition(assignments, expressionGroup.elements);

  let exprAlpha = 0;
  for (const orbit of exprPartition.orbits) {
    const projected = new Set();
    for (const tuple of orbit.members) {
      projected.add(tupleKey(toOutputTuple(tuple, vPos)));
    }
    exprAlpha += projected.size;
  }

  return { exprAlpha, exprPartition };
}

/**
 * Compute α that would result from applying orbit-based compression under
 * G_expr instead of G_pt. Purely didactic — never used for the real cost
 * display.
 */
export function computeExpressionAlphaTotal({ analysis }) {
  return computeExpressionAlphaData({ analysis }).exprAlpha;
}

/**
 * Rich comparison object for the appendix Chapter 3 explanation block.
 */
export function computeExpressionAlphaComparison({ analysis, example = null }) {
  const correctAlpha = getCorrectAlpha(analysis);
  const { exprAlpha, exprPartition } = computeExpressionAlphaData({ analysis });

  if (exprAlpha === null) {
    return {
      exprAlpha: null,
      correctAlpha,
      state: 'none',
      witness: null,
    };
  }

  if (exprAlpha === correctAlpha) {
    return {
      exprAlpha,
      correctAlpha,
      state: 'coincident',
      witness: null,
    };
  }

  return {
    exprAlpha,
    correctAlpha,
    state: 'mismatch',
    witness: buildWitness({ analysis, example, exprPartition }),
  };
}
