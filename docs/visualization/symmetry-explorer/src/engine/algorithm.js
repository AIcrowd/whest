/**
 * Core algorithm engine — JS port of the subgraph symmetry detection.
 * Ports logic from whest/_opt_einsum/_subgraph_symmetry.py
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

    // Each axis gets its own class (one U-vertex per axis).
    // No axis merging — per-operand symmetry handled by σ-loop generators.
    const classOf = {};
    for (let k = 0; k < sub.length; k++) classOf[k] = k;

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
 * Convert cycle-array notation [[0,1],[2,3]] to array-form permutation of length k.
 */
function cyclesToArrayForm(cycles, k) {
  const arr = Array.from({ length: k }, (_, i) => i);
  for (const cycle of cycles) {
    for (let i = 0; i < cycle.length; i++) {
      arr[cycle[i]] = cycle[(i + 1) % cycle.length];
    }
  }
  return arr;
}

/**
 * Run the full σ-loop. Returns results for each σ generator.
 *
 * Sources of σ generators:
 *   A) Per-operand declared symmetry → row permutations within one operand
 *   B) Identical-operand groups → adjacent swap generators between operands
 */
export function runSigmaLoop(graph, matrixData, example) {
  const { freeLabels, summedLabels, allLabels, incidence, uOperand } = graph;
  const { fpToLabels } = matrixData;

  const vLabels = freeLabels;
  const wLabels = summedLabels;
  const rowOrder = Array.from({ length: graph.uVertices.length }, (_, i) => i);

  // Build opToUIndices: operand index → list of row positions in rowOrder
  const opToUIndices = {};
  for (let k = 0; k < rowOrder.length; k++) {
    const opIdx = uOperand[rowOrder[k]];
    (opToUIndices[opIdx] ??= []).push(k);
  }

  const results = [];

  /**
   * Process a single row permutation (sigma on rows of M).
   * Computes σ(M), derives π, and appends to results.
   */
  function processSigmaRow(sigmaRowPerm, sourceLabel) {
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
        sigma: { _source: sourceLabel }, sigmaRowPerm, sigmaMatrix, sigmaColOf,
        isValid: false, reason: 'No matching π (fingerprint mismatch)',
      });
      return;
    }

    // Check for identity π
    const piIsIdentity = Object.entries(pi).every(([k, v]) => k === v);

    results.push({
      sigma: { _source: sourceLabel }, sigmaRowPerm, sigmaMatrix, sigmaColOf, pi,
      isValid: true, piIsIdentity,
    });
  }

  // ── Source A: per-operand declared symmetry generators ──
  const perOpSymmetry = example?.perOpSymmetry;
  if (perOpSymmetry) {
    const subscripts = example.subscripts;
    for (let opIdx = 0; opIdx < subscripts.length; opIdx++) {
      const opSym = Array.isArray(perOpSymmetry) ? perOpSymmetry[opIdx] : perOpSymmetry;
      if (!opSym) continue;

      const sub = subscripts[opIdx];
      let symType, symAxes;
      if (typeof opSym === 'string') {
        symType = opSym;
        symAxes = Array.from({ length: sub.length }, (_, i) => i);
      } else if (opSym && typeof opSym === 'object') {
        symType = opSym.type || 'symmetric';
        symAxes = opSym.axes || Array.from({ length: sub.length }, (_, i) => i);
      } else {
        continue;
      }

      // Get generators for this symmetry type
      let gens;
      if (symType === 'custom' && opSym.generators) {
        gens = opSym.generators.map(cycles => cyclesToArrayForm(cycles, symAxes.length))
          .filter(arr => !arr.every((v, i) => v === i));
      } else {
        gens = declaredSymGenerators(symType, symAxes.length).map(p => p.arr);
      }

      // Map each generator (acting on symAxes positions) to a row permutation
      const uIndices = opToUIndices[opIdx];
      if (!uIndices || uIndices.length === 0) continue;

      for (const genArr of gens) {
        // genArr permutes positions within symAxes of this operand.
        // Map to row permutation: symAxes[i] -> row position uIndices[symAxes[i]]
        const sigmaRowPerm = [...rowOrder];
        let isId = true;
        for (let i = 0; i < symAxes.length; i++) {
          const fromRow = uIndices[symAxes[i]];
          const toRow = uIndices[symAxes[genArr[i]]];
          if (fromRow !== undefined && toRow !== undefined) {
            sigmaRowPerm[fromRow] = rowOrder[toRow];
            if (fromRow !== toRow) isId = false;
          }
        }
        if (isId) continue;
        processSigmaRow(sigmaRowPerm, `Op${opIdx} sym`);
      }
    }
  }

  // ── Source B: identical-operand adjacent swap generators ──
  const { identicalGroups } = graph;
  for (const group of identicalGroups) {
    for (let g = 0; g < group.length - 1; g++) {
      const iOp = group[g];
      const jOp = group[g + 1];
      // Build sigma that swaps operand iOp <-> jOp
      const sigma = { [iOp]: jOp, [jOp]: iOp };
      const sigmaRowPerm = liftSigmaToU(sigma, rowOrder, graph);
      if (!sigmaRowPerm) {
        results.push({
          sigma, isValid: false, reason: 'Lift failed',
        });
        continue;
      }
      processSigmaRow(sigmaRowPerm, `Swap Op${iOp}↔Op${jOp}`);
    }
  }

  return results;
}

// ─── Group Construction ───────────────────────────────────────────

/**
 * Reduce a set of generators to a minimal generating set.
 * Greedily adds generators that strictly grow the group.
 */
function minimalGenerators(gens) {
  if (gens.length <= 1) return gens;
  const minimal = [];
  let currentSize = 1; // just identity
  for (const g of gens) {
    const trial = [...minimal, g];
    const elems = dimino(trial);
    if (elems.length > currentSize) {
      minimal.push(g);
      currentSize = elems.length;
    }
  }
  return minimal;
}

/**
 * Build generators for a declared symmetry type on a given number of axes.
 * Returns Permutation[] or empty array.
 */
function declaredSymGenerators(symType, k) {
  if (symType === 'symmetric') {
    // S_k: adjacent transpositions
    const gens = [];
    for (let i = 0; i < k - 1; i++) {
      const arr = Array.from({ length: k }, (_, j) => j);
      arr[i] = i + 1;
      arr[i + 1] = i;
      gens.push(new Permutation(arr));
    }
    return gens;
  }
  if (symType === 'cyclic') {
    return [new Permutation(Array.from({ length: k }, (_, j) => (j + 1) % k))];
  }
  if (symType === 'dihedral') {
    const gens = [new Permutation(Array.from({ length: k }, (_, j) => (j + 1) % k))];
    if (k >= 3) {
      gens.push(new Permutation([0, ...Array.from({ length: k - 1 }, (_, j) => k - 1 - j)]));
    }
    return gens;
  }
  return [];
}

export function buildGroup(sigmaResults, graph, example) {
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

  let vGensAll = dedup(vGenerators);
  let wGens = dedup(wGenerators);

  // Reduce to a minimal generating set: only keep generators that grow the group
  const vGens = minimalGenerators(vGensAll);

  const vElements = vGens.length > 0 ? dimino(vGens) : (vLabels.length > 0 ? [Permutation.identity(vLabels.length)] : []);
  const wElements = wGens.length > 0 ? dimino(wGens) : (wLabels.length > 0 ? [Permutation.identity(wLabels.length)] : []);

  // Classify group — use Python PathInfo convention: S2{a,b}, C3{i,j,k}, etc.
  // When the group is embedded (acts on a subset of labels, fixing the rest),
  // use the effective degree (number of moved labels) for classification, and
  // show only the moved labels in the label set.
  const vOrder = vElements.length;
  const vDegree = vLabels.length;
  let vGroupName = 'trivial';
  if (vDegree >= 2 && vOrder > 1) {
    // Find which labels are actually moved by any group element
    const movedSet = new Set();
    for (const el of vElements) {
      for (let i = 0; i < el.arr.length; i++) {
        if (el.arr[i] !== i) movedSet.add(i);
      }
    }
    const movedLabels = [...movedSet].sort((a, b) => a - b).map(i => vLabels[i]);
    const effectiveDegree = movedLabels.length || vDegree;
    const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
    const labelSet = `{${movedLabels.length > 0 ? movedLabels.join(',') : vLabels.join(',')}}`;
    if (vOrder === factorial(effectiveDegree)) {
      vGroupName = `S${effectiveDegree}${labelSet}`;
    } else if (vOrder === effectiveDegree && effectiveDegree >= 3) {
      vGroupName = `C${effectiveDegree}${labelSet}`;
    } else if (vOrder === 2 * effectiveDegree && effectiveDegree >= 3) {
      vGroupName = `D${effectiveDegree}${labelSet}`;
    } else if (vOrder === 2 && effectiveDegree > 2) {
      const gen = vGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      if (allTwoCycles && cycles.length > 1) {
        // Format as product of S2 on each orbit
        const orbitParts = cycles.map(c => `S2{${c.map(i => vLabels[i]).join(',')}}`);
        vGroupName = orbitParts.join('\u00d7');  // × multiplication sign
      } else {
        vGroupName = `Z2${labelSet}`;
      }
    } else if (vOrder === 2 && vDegree === 2) {
      vGroupName = `S2${labelSet}`;
    } else {
      // Fallback: show generators in cycle notation (matches Python PermGroup⟨...⟩)
      const genStr = vGens.map(g => g.cycleNotation(vLabels)).join(', ');
      vGroupName = `PermGroup\u27e8${genStr}\u27e9`;
    }
  }

  // Classify W-side group the same way
  const wOrder = wElements.length;
  const wDegree = wLabels.length;
  let wGroupName = 'trivial';
  if (wDegree >= 2 && wOrder > 1) {
    const wMovedSet = new Set();
    for (const el of wElements) {
      for (let i = 0; i < el.arr.length; i++) {
        if (el.arr[i] !== i) wMovedSet.add(i);
      }
    }
    const wMovedLabels = [...wMovedSet].sort((a, b) => a - b).map(i => wLabels[i]);
    const wEffectiveDegree = wMovedLabels.length || wDegree;
    const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
    const wLabelSet = `{${wMovedLabels.length > 0 ? wMovedLabels.join(',') : wLabels.join(',')}}`;
    if (wOrder === factorial(wEffectiveDegree)) {
      wGroupName = `S${wEffectiveDegree}${wLabelSet}`;
    } else if (wOrder === wEffectiveDegree && wEffectiveDegree >= 3) {
      wGroupName = `C${wEffectiveDegree}${wLabelSet}`;
    } else if (wOrder === 2 * wEffectiveDegree && wEffectiveDegree >= 3) {
      wGroupName = `D${wEffectiveDegree}${wLabelSet}`;
    } else if (wOrder === 2 && wEffectiveDegree > 2) {
      const gen = wGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      if (allTwoCycles && cycles.length > 1) {
        const orbitParts = cycles.map(c => `S2{${c.map(i => wLabels[i]).join(',')}}`);
        wGroupName = orbitParts.join('\u00d7');
      } else {
        wGroupName = `Z2${wLabelSet}`;
      }
    } else if (wOrder === 2 && wDegree === 2) {
      wGroupName = `S2${wLabelSet}`;
    } else {
      const genStr = wGens.map(g => g.cycleNotation(wLabels)).join(', ');
      wGroupName = `PermGroup\u27e8${genStr}\u27e9`;
    }
  }

  return {
    vLabels, wLabels,
    vGenerators: vGens, vGeneratorsAll: vGensAll,
    wGenerators: wGens,
    vElements, wElements,
    vOrder, vGroupName,
    wOrder, wGroupName,
    vDegree, wDegree,
  };
}

// ─── Declared Symmetry (user-provided, not detected) ────────────

/**
 * Build a group directly from a declared symmetry type.
 * Used for examples where the user declares symmetry on a single tensor
 * (not auto-detected from identical operands in an einsum).
 */
export function buildDeclaredGroup(example) {
  const labels = [...example.output].sort();
  const k = labels.length;
  const sym = Array.isArray(example.perOpSymmetry)
    ? example.perOpSymmetry[0]
    : example.perOpSymmetry;

  let vGens = [];

  if (sym === 'symmetric') {
    // S_k: adjacent transpositions
    for (let i = 0; i < k - 1; i++) {
      const arr = Array.from({ length: k }, (_, j) => j);
      arr[i] = i + 1;
      arr[i + 1] = i;
      vGens.push(new Permutation(arr));
    }
  } else if (sym === 'cyclic') {
    // C_k: single k-cycle (0 → 1 → 2 → ... → k-1 → 0)
    const arr = Array.from({ length: k }, (_, j) => (j + 1) % k);
    vGens.push(new Permutation(arr));
  } else if (sym === 'dihedral') {
    // D_k: rotation + reflection
    const rotation = Array.from({ length: k }, (_, j) => (j + 1) % k);
    vGens.push(new Permutation(rotation));
    if (k >= 3) {
      const reflection = [0, ...Array.from({ length: k - 1 }, (_, j) => k - 1 - j)];
      vGens.push(new Permutation(reflection));
    }
  } else if (sym === 'block-swap') {
    // Block swap: (0,1) ↔ (2,3) via (0 2)(1 3)
    if (k === 4) {
      vGens.push(new Permutation([2, 3, 0, 1]));
    }
  }

  if (vGens.length === 0) {
    vGens = [Permutation.identity(k)];
  }

  const vElements = dimino(vGens);
  const vOrder = vElements.length;
  const vDegree = k;

  // Classify the group name
  const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
  const labelSet = `{${labels.join(',')}}`;
  let vGroupName = 'trivial';
  if (vDegree >= 2 && vOrder > 1) {
    if (vOrder === factorial(vDegree)) {
      vGroupName = `S${vDegree}${labelSet}`;
    } else if (vOrder === vDegree && vDegree >= 3) {
      vGroupName = `C${vDegree}${labelSet}`;
    } else if (vOrder === 2 * vDegree && vDegree >= 3) {
      vGroupName = `D${vDegree}${labelSet}`;
    } else if (vOrder === 2) {
      const gen = vGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      if (allTwoCycles && cycles.length > 1) {
        const orbitParts = cycles.map(c => `S2{${c.map(i => labels[i]).join(',')}}`);
        vGroupName = orbitParts.join('\u00d7');
      } else {
        vGroupName = `S2${labelSet}`;
      }
    } else {
      const genStr = vGens.map(g => g.cycleNotation(labels)).join(', ');
      vGroupName = `PermGroup\u27e8${genStr}\u27e9`;
    }
  }

  return {
    vLabels: labels,
    wLabels: [],
    vGenerators: vGens,
    vGeneratorsAll: vGens,
    wGenerators: [],
    vElements,
    wElements: [],
    vOrder,
    vGroupName,
    vDegree,
    declared: true,
  };
}

// ─── Burnside Counting ───────────────────────────────────────────

export function computeBurnside(group, dimensionN) {
  // V-side Burnside
  const vSizes = Array(group.vDegree).fill(dimensionN);
  const vResult = group.vDegree > 0
    ? burnsideCount(group.vElements, vSizes)
    : { perElement: [], totalFixed: 0, uniqueCount: 1 };
  const vTotalCount = group.vDegree > 0 ? Math.pow(dimensionN, group.vDegree) : 1;

  // W-side Burnside
  const wDegree = group.wDegree || (group.wLabels ? group.wLabels.length : 0);
  const wOrder = group.wOrder || 1;
  const wSizes = Array(wDegree).fill(dimensionN);
  const wResult = (wDegree > 0 && wOrder > 1)
    ? burnsideCount(group.wElements, wSizes)
    : { perElement: [], totalFixed: 0, uniqueCount: wDegree > 0 ? Math.pow(dimensionN, wDegree) : 1 };
  const wTotalCount = wDegree > 0 ? Math.pow(dimensionN, wDegree) : 1;

  return {
    // V-side
    perElement: vResult.perElement,
    totalFixed: vResult.totalFixed,
    uniqueCount: vResult.uniqueCount,
    totalCount: vTotalCount,
    ratio: vTotalCount > 0 ? vResult.uniqueCount / vTotalCount : 1,
    // W-side
    wPerElement: wResult.perElement,
    wUniqueCount: wResult.uniqueCount,
    wTotalCount: wTotalCount,
    wRatio: wTotalCount > 0 ? wResult.uniqueCount / wTotalCount : 1,
    wHasSymmetry: wOrder > 1,
    // Common
    dimensionN,
  };
}

// ─── Cost Reduction ──────────────────────────────────────────────

export function computeCostReduction(burnside, group, numTerms = 2) {
  const { uniqueCount, totalCount, ratio } = burnside;

  // Total contraction size = V * W (output elements × summed elements)
  const allCount = totalCount * burnside.wTotalCount;
  // Dense cost: FMA counts as 1 op (not 2), so op_factor = max(1, num_terms - 1)
  const opFactor = Math.max(1, numTerms - 1);
  const denseCost = Math.max(1, opFactor * allCount);

  // V-only reduction: reduce output side by V symmetry
  const vReducedCost = Math.max(1, Math.round(denseCost * ratio));

  // V+W combined reduction: also reduce inner loop by W symmetry
  const combinedRatio = ratio * burnside.wRatio;
  const combinedReducedCost = Math.max(1, Math.round(denseCost * combinedRatio));

  return {
    denseCost,
    // V-side only
    reducedCost: vReducedCost,
    ratio,
    speedup: denseCost / vReducedCost,
    uniqueCount,
    totalCount,
    groupName: group.vGroupName,
    groupOrder: group.vOrder,
    // W-side savings (additional, shown separately)
    wReducedCost: combinedReducedCost,
    wRatio: burnside.wRatio,
    wSpeedup: denseCost / combinedReducedCost,
    wUniqueCount: burnside.wUniqueCount,
    wTotalCount: burnside.wTotalCount,
    wHasSymmetry: burnside.wHasSymmetry,
    wGroupName: group.wGroupName || 'trivial',
    // Combined
    combinedRatio,
    combinedReducedCost,
  };
}
