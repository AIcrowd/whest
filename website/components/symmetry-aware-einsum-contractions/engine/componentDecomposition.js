/**
 * Component Decomposition Engine
 *
 * Decomposes the full symmetry group G into independent label components
 * (connected components of the label-interaction graph) and classifies
 * each component using the shared spec in ./classificationSpec.js.
 *
 * Cases:
 *   trivial  — no nontrivial symmetry (|Gₐ| = 1)
 *   A        — only V-labels (Wa = empty)
 *   B        — only W-labels (Va = empty)
 *   C        — both V and W labels, but no generator crosses the V/W boundary
 *   D        — cross-V/W generators AND the restricted group is the full symmetric group
 *   E        — cross-V/W generators but NOT the full symmetric group
 *
 * The decision order and predicates live in classificationSpec.js; changes
 * to the tree structure happen in one place there, and both this module
 * and the tree visualization pick them up automatically.
 */

import { Permutation, dimino } from './permutation.js';
import { classifyComponent } from './classificationSpec.js';

export const CASE_META = {
  trivial: {
    label: 'Direct count (trivial)',
    description: 'Trivial group — no symmetry, count every assignment directly',
    color: '#CBD5E1',
    method: 'ρ = |Iₐ| (direct)',
  },
  A: {
    label: 'Case A: V-only',
    description: 'V-only component (free labels only, no summed labels)',
    color: '#4A7CFF',
    method: 'ρ = ∏nₗ (no accumulation savings)',
  },
  B: {
    label: 'Case B: W-only',
    description: 'W-only component (summed labels only, no free labels)',
    color: '#94A3B8',
    method: 'ρ = Burnside on Gₐ',
  },
  C: {
    label: 'Case C: Correlated',
    description: 'Mixed V+W component — no cross-boundary generators',
    color: '#FA9E33',
    method: 'ρ = orbit enumeration',
  },
  D: {
    label: 'Case D: Cross (Young)',
    description: 'Mixed V+W — cross generators, full symmetric group',
    color: '#23B761',
    method: 'ρ = Burnside on Hₐ',
  },
  E: {
    label: 'Case E: Cross (general)',
    description: 'Mixed V+W — cross generators, partial group',
    color: '#F0524D',
    method: 'ρ = orbit enumeration',
  },
};

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x) {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }

  union(x, y) {
    const rx = this.find(x);
    const ry = this.find(y);
    if (rx === ry) return;
    if (this.rank[rx] < this.rank[ry]) {
      this.parent[rx] = ry;
    } else if (this.rank[rx] > this.rank[ry]) {
      this.parent[ry] = rx;
    } else {
      this.parent[ry] = rx;
      this.rank[rx] += 1;
    }
  }
}

export function buildLabelInteractionGraph(allLabels, generators) {
  const n = allLabels.length;
  const uf = new UnionFind(n);
  const edges = [];

  for (const gen of generators) {
    const movedByThisGen = [];
    for (let i = 0; i < n; i += 1) {
      if (gen.arr[i] !== i) {
        movedByThisGen.push(i);
        edges.push([i, gen.arr[i]]);
        uf.union(i, gen.arr[i]);
      }
    }

    for (let j = 1; j < movedByThisGen.length; j += 1) {
      uf.union(movedByThisGen[0], movedByThisGen[j]);
    }
  }

  const componentMap = new Map();
  for (let i = 0; i < n; i += 1) {
    const root = uf.find(i);
    if (!componentMap.has(root)) componentMap.set(root, []);
    componentMap.get(root).push(i);
  }

  const components = [...componentMap.values()].map((indices) => indices.sort((a, b) => a - b));
  return { edges, components };
}

function restrictPermutation(perm, indices) {
  const localIdx = new Map(indices.map((globalIdx, localPos) => [globalIdx, localPos]));
  const arr = indices.map((globalIdx) => {
    const target = perm.arr[globalIdx];
    return localIdx.get(target);
  });
  if (arr.some((value) => value === undefined)) return null;
  return new Permutation(arr);
}

function factorial(n) {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

function classifyGroupName(labels, generators, elements) {
  const order = elements.length;
  const degree = labels.length;
  if (degree < 2 || order <= 1) return 'trivial';

  const movedSet = new Set();
  for (const el of elements) {
    for (let i = 0; i < el.arr.length; i += 1) {
      if (el.arr[i] !== i) movedSet.add(i);
    }
  }

  const movedIndices = [...movedSet].sort((a, b) => a - b);
  const movedLabels = movedIndices.map((i) => labels[i]);
  const effectiveDegree = movedLabels.length || degree;
  const labelSet = `{${movedLabels.length > 0 ? movedLabels.join(',') : labels.join(',')}}`;
  const supportIndices = movedIndices.length > 0 ? movedIndices : labels.map((_, i) => i);

  const supportElements = elements
    .map((el) => restrictPermutation(el, supportIndices))
    .filter((el) => el !== null);

  if (order === factorial(effectiveDegree)) {
    const hasTransposition = supportElements.some((el) => {
      const cycles = el.cyclicForm();
      return cycles.length === 1 && cycles[0].length === 2;
    });
    if (hasTransposition || effectiveDegree <= 2) return `S${effectiveDegree}${labelSet}`;
  }

  if (order === effectiveDegree && effectiveDegree >= 3) {
    const hasFullCycle = supportElements.some((el) => {
      const cycles = el.cyclicForm();
      return cycles.length === 1 && cycles[0].length === effectiveDegree;
    });
    if (hasFullCycle) return `C${effectiveDegree}${labelSet}`;
  }

  if (order === 2 * effectiveDegree && effectiveDegree >= 3) {
    const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
    const lcm = (a, b) => (a * b) / gcd(a, b);
    const permOrder = (el) => {
      const cycles = el.cyclicForm();
      if (cycles.length === 0) return 1;
      return cycles.reduce((acc, cycle) => lcm(acc, cycle.length), 1);
    };
    const rotations = supportElements.filter((el) => permOrder(el) === effectiveDegree);
    const reflections = supportElements.filter((el) => permOrder(el) === 2 && !el.isIdentity);
    let isDihedral = false;
    outer: for (const rot of rotations) {
      for (const ref of reflections) {
        const conj = ref.compose(rot).compose(ref);
        if (conj.equals(rot.inverse())) {
          isDihedral = true;
          break outer;
        }
      }
    }
    if (isDihedral) return `D${effectiveDegree}${labelSet}`;
  }

  if (order === 2) {
    if (effectiveDegree > 2) {
      const gen = generators[0];
      const cycles = gen?.cyclicForm() || [];
      if (cycles.length > 1 && cycles.every((cycle) => cycle.length === 2)) {
        return cycles
          .map((cycle) => `S2{${cycle.map((i) => labels[i]).join(',')}}`)
          .join('\u00d7');
      }
      return `Z2${labelSet}`;
    }
    if (effectiveDegree === 2) return `S2${labelSet}`;
  }

  const genStr = generators.map((g) => g.cycleNotation(labels)).join(', ');
  return `PermGroup\u27e8${genStr}\u27e9`;
}

export function decomposeAndClassify(allLabels, vLabels, wLabels, fullGenerators, fullElements) {
  const vSet = new Set(vLabels);
  const wSet = new Set(wLabels);
  const effectiveGenerators = fullGenerators && fullGenerators.length > 0 ? fullGenerators : [];

  const interactionGraph = buildLabelInteractionGraph(allLabels, effectiveGenerators);
  const { components: rawComponents } = interactionGraph;

  const components = rawComponents.map((indices) => {
    const labels = indices.map((i) => allLabels[i]);
    const va = labels.filter((label) => vSet.has(label));
    const wa = labels.filter((label) => wSet.has(label));

    const restrictedGens = [];
    for (const gen of effectiveGenerators) {
      const restricted = restrictPermutation(gen, indices);
      if (restricted && !restricted.isIdentity) restrictedGens.push(restricted);
    }

    const seenKeys = new Set();
    const dedupGens = restrictedGens.filter((g) => {
      const key = g.key();
      if (seenKeys.has(key)) return false;
      seenKeys.add(key);
      return true;
    });

    const elements = dedupGens.length > 0 ? dimino(dedupGens) : [Permutation.identity(indices.length)];
    const order = elements.length;
    const groupName = classifyGroupName(labels, dedupGens, elements);

    const hasCrossGen = dedupGens.some((gen) => {
      for (let localPos = 0; localPos < indices.length; localPos += 1) {
        const globalFrom = indices[localPos];
        const globalTo = indices[gen.arr[localPos]];
        const fromIsV = vSet.has(allLabels[globalFrom]);
        const toIsV = vSet.has(allLabels[globalTo]);
        if (fromIsV !== toIsV) return true;
      }
      return false;
    });

    const isFullSym = order === factorial(indices.length);
    const classification = classifyComponent({
      order,
      vCount: va.length,
      wCount: wa.length,
      hasCrossGen,
      isFullSym,
      labelCount: indices.length,
    });
    const { caseType, path } = classification;

    let ha = null;
    let haElements = null;
    if (caseType === 'D') {
      const vaLocalPos = labels
        .map((label, localPos) => ({ label, localPos }))
        .filter(({ label }) => vSet.has(label))
        .map(({ localPos }) => localPos);

      haElements = elements.filter((el) => {
        for (const pos of vaLocalPos) {
          if (el.arr[pos] !== pos) return false;
        }
        return true;
      });

      const haGens = haElements.filter((el) => !el.isIdentity);
      const haGenKeys = new Set();
      const haGensDedup = haGens.filter((el) => {
        const key = el.key();
        if (haGenKeys.has(key)) return false;
        haGenKeys.add(key);
        return true;
      });

      ha = {
        generators: haGensDedup,
        elements: haElements,
        order: haElements.length,
      };
    }

    return {
      indices,
      labels,
      va,
      wa,
      generators: dedupGens,
      elements,
      order,
      groupName,
      caseType,
      path,
      ha,
      haElements,
    };
  });

  return { interactionGraph, components };
}
