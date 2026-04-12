/**
 * Core algorithm engine — JS port of the subgraph symmetry detection.
 * Ports logic from mechestim/_opt_einsum/_subgraph_symmetry.py
 */

import { Permutation, dimino, burnsideCount } from './permutation.js';

// ─── Bipartite Graph Construction ─────────────────────────────────

/**
 * Build the bipartite graph for an einsum expression.
 * Port of _build_bipartite (subgraph_symmetry.py:174-277)
 */
export function buildBipartite(example) {
  const { subscripts, output, operandNames, perOpSymmetry } = example;
  const numOps = subscripts.length;

  const uVertices = [];  // { opIdx, classId, labels: Set<string> }
  const incidence = [];  // { [label]: multiplicity }
  const uOperand = [];
  const operandLabels = [];

  for (let opIdx = 0; opIdx < numOps; opIdx++) {
    const sub = subscripts[opIdx];
    operandLabels.push(new Set(sub));

    // Build equivalence classes on axes.
    // Each axis starts in its own class. Per-op symmetry merges them.
    const classOf = {};
    for (let k = 0; k < sub.length; k++) classOf[k] = k;

    if (perOpSymmetry === 'symmetric') {
      // All axes in one class (full S_k on all axes of this operand)
      for (let k = 1; k < sub.length; k++) classOf[k] = 0;
    }

    // Normalize class ids
    const classOrder = {};
    let nextClass = 0;
    for (let k = 0; k < sub.length; k++) {
      const c = classOf[k];
      if (!(c in classOrder)) classOrder[c] = nextClass++;
      classOf[k] = classOrder[c];
    }

    // Build one U-vertex per class
    const numClasses = nextClass;
    const classInc = Array.from({ length: numClasses }, () => ({}));
    const classLabels = Array.from({ length: numClasses }, () => new Set());

    for (let k = 0; k < sub.length; k++) {
      const cls = classOf[k];
      const ch = sub[k];
      classInc[cls][ch] = (classInc[cls][ch] || 0) + 1;
      classLabels[cls].add(ch);
    }

    for (let cls = 0; cls < numClasses; cls++) {
      uVertices.push({ opIdx, classId: cls, labels: classLabels[cls] });
      incidence.push(classInc[cls]);
      uOperand.push(opIdx);
    }
  }

  // All labels
  const allLabels = new Set();
  operandLabels.forEach(s => s.forEach(l => allLabels.add(l)));
  const outputSet = new Set(output);
  const freeLabels = new Set([...allLabels].filter(l => outputSet.has(l)));
  const summedLabels = new Set([...allLabels].filter(l => !outputSet.has(l)));

  // Identical-operand groups (by name — simulates Python `is`)
  const nameToPositions = {};
  for (let i = 0; i < numOps; i++) {
    const name = operandNames[i];
    (nameToPositions[name] ??= []).push(i);
  }
  const identicalGroups = Object.values(nameToPositions).filter(g => g.length >= 2);

  return {
    uVertices, incidence, uOperand, operandLabels,
    freeLabels, summedLabels, identicalGroups,
    allLabels: [...allLabels].sort(),
    numOperands: numOps,
  };
}

// ─── Incidence Matrix ─────────────────────────────────────────────

export function buildIncidenceMatrix(graph) {
  const { uVertices, incidence, allLabels, freeLabels, summedLabels } = graph;
  const labels = allLabels;
  const matrix = uVertices.map((_, rowIdx) =>
    labels.map(lbl => incidence[rowIdx][lbl] || 0)
  );

  // Column fingerprints
  const colFingerprints = {};
  const fpToLabels = {};
  for (let c = 0; c < labels.length; c++) {
    const fp = matrix.map(row => row[c]).join(',');
    colFingerprints[labels[c]] = fp;
    (fpToLabels[fp] ??= new Set()).add(labels[c]);
  }

  // Row labels for display
  const rowLabels = uVertices.map(u => {
    const opName = `${graph.operandLabels[u.opIdx] ? '' : ''}Op${u.opIdx}`;
    return { opIdx: u.opIdx, classId: u.classId, labels: u.labels };
  });

  return { matrix, labels, rowLabels, colFingerprints, fpToLabels };
}

// ─── σ-Loop: enumerate operand permutations ───────────────────────

function enumeratePermutations(groups) {
  // Enumerate all permutations within each identical-operand group.
  // Returns list of {opIdx -> permutedOpIdx} mappings.
  function permsOfGroup(group) {
    const result = [];
    const arr = [...group];
    function permute(start) {
      if (start === arr.length) {
        const mapping = {};
        arr.forEach((val, i) => mapping[group[i]] = val);
        result.push(mapping);
        return;
      }
      for (let i = start; i < arr.length; i++) {
        [arr[start], arr[i]] = [arr[i], arr[start]];
        permute(start + 1);
        [arr[start], arr[i]] = [arr[i], arr[start]];
      }
    }
    permute(0);
    return result;
  }

  const perGroupPerms = groups.map(permsOfGroup);
  if (perGroupPerms.length === 0) return [{}];

  // Cartesian product across groups
  function cartesian(arrays, idx = 0) {
    if (idx === arrays.length) return [{}];
    const rest = cartesian(arrays, idx + 1);
    const result = [];
    for (const p of arrays[idx]) {
      for (const r of rest) {
        result.push({ ...p, ...r });
      }
    }
    return result;
  }
  return cartesian(perGroupPerms);
}

/**
 * Lift operand permutation σ to U-vertex row permutation.
 * Port of _lift_operand_perm_to_u (subgraph_symmetry.py:458-495)
 */
function liftSigmaToU(sigma, rowOrder, graph) {
  const opToUVertices = {};
  for (let uIdx = 0; uIdx < graph.uOperand.length; uIdx++) {
    const opIdx = graph.uOperand[uIdx];
    (opToUVertices[opIdx] ??= []).push(uIdx);
  }

  const result = [...rowOrder];
  for (let k = 0; k < rowOrder.length; k++) {
    const uIdx = rowOrder[k];
    const opIdx = graph.uOperand[uIdx];
    if (!(opIdx in sigma) || sigma[opIdx] === opIdx) continue;
    const jOp = sigma[opIdx];
    const opClasses = opToUVertices[opIdx];
    const pos = opClasses.indexOf(uIdx);
    const jClasses = opToUVertices[jOp] || [];
    if (pos >= jClasses.length) return null;
    result[k] = jClasses[pos];
  }
  return result;
}

/**
 * Derive π from σ(M) via column fingerprint matching.
 * Port of _derive_pi_canonical (subgraph_symmetry.py:32-70)
 */
function derivePi(sigmaColOf, fpToLabels, vLabels, wLabels) {
  const pi = {};
  const used = new Set();
  const allLabels = [...vLabels, ...wLabels].sort();

  for (const label of allLabels) {
    const fp = sigmaColOf[label];
    const candidates = fpToLabels[fp];
    if (!candidates) return null;
    let pick = null;
    for (const c of [...candidates].sort()) {
      if (!used.has(c)) { pick = c; break; }
    }
    if (pick === null) return null;
    pi[label] = pick;
    used.add(pick);
  }

  // Validate V→V and W→W
  for (const [lbl, target] of Object.entries(pi)) {
    if (vLabels.has(lbl) && !vLabels.has(target)) return null;
    if (wLabels.has(lbl) && !wLabels.has(target)) return null;
  }

  return pi;
}

/**
 * Run the full σ-loop. Returns results for each σ.
 */
export function runSigmaLoop(graph, matrixData) {
  const { identicalGroups, freeLabels, summedLabels, allLabels, incidence } = graph;
  const { colFingerprints, fpToLabels } = matrixData;

  const vLabels = freeLabels;
  const wLabels = summedLabels;
  const rowOrder = Array.from({ length: graph.uVertices.length }, (_, i) => i);

  const allSigmas = enumeratePermutations(identicalGroups);
  const results = [];

  for (const sigma of allSigmas) {
    // Check if identity
    const isIdentity = Object.entries(sigma).every(([k, v]) => Number(k) === v);
    if (isIdentity || Object.keys(sigma).length === 0) {
      results.push({ sigma, isIdentity: true, skipped: true });
      continue;
    }

    // Lift to U
    const sigmaRowPerm = liftSigmaToU(sigma, rowOrder, graph);
    if (!sigmaRowPerm) {
      results.push({ sigma, isValid: false, reason: 'Lift failed' });
      continue;
    }

    // Compute σ(M) column fingerprints
    const sigmaColOf = {};
    for (const label of allLabels) {
      sigmaColOf[label] = rowOrder.map(
        (_, k) => incidence[sigmaRowPerm[k]][label] || 0
      ).join(',');
    }

    // Build σ(M) matrix for display
    const sigmaMatrix = sigmaRowPerm.map(uIdx =>
      allLabels.map(lbl => incidence[uIdx][lbl] || 0)
    );

    // Derive π
    const pi = derivePi(sigmaColOf, fpToLabels, vLabels, wLabels);
    if (!pi) {
      results.push({
        sigma, sigmaRowPerm, sigmaMatrix, sigmaColOf,
        isValid: false, reason: 'No matching π (fingerprint mismatch)',
      });
      continue;
    }

    // Check for identity π
    const piIsIdentity = Object.entries(pi).every(([k, v]) => k === v);

    results.push({
      sigma, sigmaRowPerm, sigmaMatrix, sigmaColOf, pi,
      isValid: true, piIsIdentity,
    });
  }

  return results;
}

// ─── Group Construction ───────────────────────────────────────────

export function buildGroup(sigmaResults, graph) {
  const vLabels = [...graph.freeLabels].sort();
  const wLabels = [...graph.summedLabels].sort();

  const vIdx = {};
  vLabels.forEach((l, i) => vIdx[l] = i);
  const wIdx = {};
  wLabels.forEach((l, i) => wIdx[l] = i);

  const vGenerators = [];
  const wGenerators = [];

  for (const r of sigmaResults) {
    if (!r.isValid || r.skipped || r.piIsIdentity) continue;
    const { pi } = r;

    // Restrict π to V labels
    if (vLabels.length >= 2) {
      const arr = vLabels.map(l => vIdx[pi[l] ?? l]);
      const perm = new Permutation(arr);
      if (!perm.isIdentity) vGenerators.push(perm);
    }

    // Restrict π to W labels
    if (wLabels.length >= 2) {
      const arr = wLabels.map(l => wIdx[pi[l] ?? l]);
      const perm = new Permutation(arr);
      if (!perm.isIdentity) wGenerators.push(perm);
    }
  }

  // Deduplicate generators
  const dedup = (gens) => {
    const seen = new Set();
    return gens.filter(g => {
      const k = g.key();
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  };

  const vGens = dedup(vGenerators);
  const wGens = dedup(wGenerators);

  const vElements = vGens.length > 0 ? dimino(vGens) : (vLabels.length > 0 ? [Permutation.identity(vLabels.length)] : []);
  const wElements = wGens.length > 0 ? dimino(wGens) : (wLabels.length > 0 ? [Permutation.identity(wLabels.length)] : []);

  // Classify group
  const vOrder = vElements.length;
  const vDegree = vLabels.length;
  let vGroupName = 'trivial';
  if (vDegree >= 2 && vOrder > 1) {
    const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
    // Check for well-known groups
    if (vOrder === factorial(vDegree)) {
      vGroupName = `S${subscriptDigit(vDegree)}`;
    } else if (vOrder === vDegree && vDegree >= 3) {
      vGroupName = `C${subscriptDigit(vDegree)}`;
    } else if (vOrder === 2 * vDegree && vDegree >= 3) {
      vGroupName = `D${subscriptDigit(vDegree)}`;
    } else if (vOrder === 2 && vDegree > 2) {
      // Block S₂ or Z₂: order 2 on more than 2 labels
      // Check if the generator has only 2-cycles (block swap)
      const gen = vGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      vGroupName = allTwoCycles && cycles.length > 1 ? 'Block S\u2082' : 'Z\u2082';
    } else if (vOrder === 2 && vDegree === 2) {
      vGroupName = 'S\u2082';
    } else {
      vGroupName = `|G|=${vOrder}`;
    }
  }

  return {
    vLabels, wLabels,
    vGenerators: vGens, wGenerators: wGens,
    vElements, wElements,
    vOrder, vGroupName,
    vDegree,
  };
}

function subscriptDigit(n) {
  const subs = '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089';
  return String(n).split('').map(d => subs[parseInt(d)]).join('');
}

// ─── Burnside Counting ───────────────────────────────────────────

export function computeBurnside(group, dimensionN) {
  const sizes = Array(group.vDegree).fill(dimensionN);
  const result = burnsideCount(group.vElements, sizes);
  const totalCount = Math.pow(dimensionN, group.vDegree);
  return {
    ...result,
    totalCount,
    ratio: result.uniqueCount / totalCount,
    dimensionN,
  };
}

// ─── Cost Reduction ──────────────────────────────────────────────

export function computeCostReduction(burnside, group) {
  const { uniqueCount, totalCount, ratio } = burnside;
  // Simplified: assume a contraction with op_factor = 2
  const denseCost = 2 * totalCount;
  const reducedCost = Math.max(1, Math.round(denseCost * ratio));
  return {
    denseCost,
    reducedCost,
    ratio,
    speedup: denseCost / reducedCost,
    uniqueCount,
    totalCount,
    groupName: group.vGroupName,
    groupOrder: group.vOrder,
  };
}
