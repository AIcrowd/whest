import { Permutation, dimino } from './permutation.js';

function dedupGenerators(generators) {
  const seen = new Set();
  return generators.filter((generator) => {
    const key = generator.key();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function minimalGenerators(generators) {
  if (generators.length <= 1) return generators;
  const minimal = [];
  let currentSize = 1;
  for (const generator of generators) {
    const trial = [...minimal, generator];
    const elements = dimino(trial);
    if (elements.length > currentSize) {
      minimal.push(generator);
      currentSize = elements.length;
    }
  }
  return minimal;
}

function buildGeneratorSelectionProvenance(allLabels, validPiResults) {
  const allLabelIndex = new Map(allLabels.map((label, index) => [label, index]));

  const validPiCards = validPiResults
    .filter((result) => !result.skipped && !result.piIsIdentity)
    .map((result, index) => {
      const arr = allLabels.map((label) => allLabelIndex.get(result.pi[label]));
      const permutation = new Permutation(arr);
      return {
        id: `valid-pi-${index}`,
        piKind: result.piKind,
        pi: result.pi,
        permutation,
        permutationKey: permutation.key(),
        arrayForm: permutation.arr,
        cycleNotation: permutation.cycleNotation(allLabels),
        isIdentity: permutation.isIdentity,
      };
    })
    .filter((card) => !card.isIdentity);

  const piToCandidateKey = new Map();
  const candidateByKey = new Map();
  const candidatePermutations = [];
  for (const card of validPiCards) {
    piToCandidateKey.set(card.id, card.permutationKey);
    const existing = candidateByKey.get(card.permutationKey);
    if (existing) {
      existing.sourcePiIds.push(card.id);
      existing.sourcePiKinds.push(card.piKind);
      continue;
    }
    const candidate = {
      id: `candidate-${candidatePermutations.length}`,
      permutation: card.permutation,
      permutationKey: card.permutationKey,
      arrayForm: card.permutation.arr,
      cycleNotation: card.permutation.cycleNotation(allLabels),
      sourcePiIds: [card.id],
      sourcePiKinds: [card.piKind],
      kept: false,
      beforeElements: [],
      afterElements: [],
    };
    candidatePermutations.push(candidate);
    candidateByKey.set(card.permutationKey, candidate);
  }

  const selectedGenerators = [];
  let currentSize = 1;
  let currentElements = allLabels.length > 0 ? [Permutation.identity(allLabels.length)] : [];
  for (const candidate of candidatePermutations) {
    candidate.beforeElements = [...currentElements];
    const trial = [...selectedGenerators.map((entry) => entry.permutation), candidate.permutation];
    const elements = dimino(trial);
    candidate.afterElements = [...elements];
    if (elements.length > currentSize) {
      candidate.kept = true;
      candidate.growthFrom = currentSize;
      candidate.growthTo = elements.length;
      selectedGenerators.push({
        id: `selected-generator-${selectedGenerators.length}`,
        sourcePiIds: [...candidate.sourcePiIds],
        permutation: candidate.permutation,
        permutationKey: candidate.permutationKey,
        arrayForm: candidate.arrayForm,
        cycleNotation: candidate.cycleNotation,
      });
      currentSize = elements.length;
      currentElements = elements;
    } else {
      candidate.growthFrom = currentSize;
      candidate.growthTo = currentSize;
    }
  }

  return {
    validPiCards,
    candidatePermutations,
    selectedGenerators,
    piToCandidateKey: Object.fromEntries(piToCandidateKey),
  };
}

function permutationOrder(permutation) {
  const cycles = permutation.cyclicForm();
  if (cycles.length === 0) return 1;
  const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
  const lcm = (a, b) => (a * b) / gcd(a, b);
  return cycles.reduce((order, cycle) => lcm(order, cycle.length), 1);
}

function restrictPermutation(permutation, indices) {
  const indexToLocal = new Map(indices.map((index, localIndex) => [index, localIndex]));
  const arr = indices.map((index) => indexToLocal.get(permutation.arr[index]));
  if (arr.some((value) => value === undefined)) return null;
  return new Permutation(arr);
}

function isFullSymmetricGroup(elements, degree) {
  const factorial = (n) => (n <= 1 ? 1 : n * factorial(n - 1));
  if (elements.length !== factorial(degree)) return false;
  return elements.some((element) => {
    const cycles = element.cyclicForm();
    return cycles.length === 1 && cycles[0].length === 2;
  });
}

function isCyclicGroup(elements, degree) {
  if (elements.length !== degree || degree < 3) return false;
  return elements.some((element) => permutationOrder(element) === degree);
}

function isDihedralGroup(elements, degree) {
  if (elements.length !== 2 * degree || degree < 3) return false;
  const rotations = elements.filter((element) => permutationOrder(element) === degree);
  const reflections = elements.filter((element) => permutationOrder(element) === 2 && !element.isIdentity);
  for (const rotation of rotations) {
    for (const reflection of reflections) {
      const conjugated = reflection.compose(rotation).compose(reflection);
      if (conjugated.equals(rotation.inverse())) return true;
    }
  }
  return false;
}

function classifyGroupName(labels, generators, elements) {
  const order = elements.length;
  const degree = labels.length;
  if (degree < 2 || order <= 1) return 'trivial';

  const movedSet = new Set();
  for (const element of elements) {
    for (let i = 0; i < element.arr.length; i++) {
      if (element.arr[i] !== i) movedSet.add(i);
    }
  }

  const movedLabels = [...movedSet].sort((a, b) => a - b).map((i) => labels[i]);
  const movedIndices = [...movedSet].sort((a, b) => a - b);
  const effectiveDegree = movedLabels.length || degree;
  const labelSet = `{${movedLabels.length > 0 ? movedLabels.join(',') : labels.join(',')}}`;
  const supportIndices = movedIndices.length > 0 ? movedIndices : labels.map((_, index) => index);
  const supportElements = elements
    .map((element) => restrictPermutation(element, supportIndices))
    .filter((element) => element !== null);

  if (isFullSymmetricGroup(supportElements, effectiveDegree)) return `S${effectiveDegree}${labelSet}`;
  if (isCyclicGroup(supportElements, effectiveDegree)) return `C${effectiveDegree}${labelSet}`;
  if (isDihedralGroup(supportElements, effectiveDegree)) return `D${effectiveDegree}${labelSet}`;
  if (order === 2 && effectiveDegree > 2) {
    const generator = generators[0];
    const cycles = generator?.cyclicForm() || [];
    if (cycles.length > 0 && cycles.every((cycle) => cycle.length === 2) && cycles.length > 1) {
      return cycles
        .map((cycle) => `S2{${cycle.map((i) => labels[i]).join(',')}}`)
        .join('\u00d7');
    }
    return `Z2${labelSet}`;
  }
  if (order === 2 && effectiveDegree === 2) return `S2${labelSet}`;

  const generatorText = generators.map((generator) => generator.cycleNotation(labels)).join(', ');
  return `PermGroup\u27e8${generatorText}\u27e9`;
}

export function classifyPi(pi, vLabelsInput, wLabelsInput) {
  const vLabels = vLabelsInput instanceof Set ? vLabelsInput : new Set(vLabelsInput);
  const wLabels = wLabelsInput instanceof Set ? wLabelsInput : new Set(wLabelsInput);

  let movesV = false;
  let movesW = false;
  let crosses = false;

  for (const [label, target] of Object.entries(pi)) {
    if (label !== target) {
      if (vLabels.has(label)) movesV = true;
      if (wLabels.has(label)) movesW = true;
    }
    if ((vLabels.has(label) && wLabels.has(target)) || (wLabels.has(label) && vLabels.has(target))) {
      crosses = true;
    }
  }

  const piIsIdentity = !movesV && !movesW;

  let piKind = 'identity';
  if (!piIsIdentity) {
    if (crosses) {
      piKind = 'cross';
    } else if (movesV && movesW) {
      piKind = 'correlated';
    } else if (movesV) {
      piKind = 'v-only';
    } else if (movesW) {
      piKind = 'w-only';
    }
  }

  return {
    piIsIdentity,
    piKind,
    crosses,
    movesV,
    movesW,
  };
}

function classifyPermutationAction(labels, permutation, vLabelsInput, wLabelsInput) {
  const pi = Object.fromEntries(labels.map((label, index) => [label, labels[permutation.arr[index]]]));
  return classifyPi(pi, vLabelsInput, wLabelsInput);
}

function validatePiMappings(allLabels, validPiResults) {
  const labelSet = new Set(allLabels);

  for (const [index, result] of validPiResults.entries()) {
    if (!result.pi || typeof result.pi !== 'object') {
      throw new Error(`Invalid pi mapping at result ${index}: missing pi object`);
    }

    const keys = Object.keys(result.pi).sort();
    const expectedKeys = [...allLabels].sort();
    if (keys.length !== expectedKeys.length || keys.some((key, keyIndex) => key !== expectedKeys[keyIndex])) {
      throw new Error(`Invalid pi mapping at result ${index}: keys must exactly match allLabels`);
    }

    const image = new Set();
    for (const label of allLabels) {
      const target = result.pi[label];
      if (!labelSet.has(target)) {
        throw new Error(`Invalid pi mapping at result ${index}: target "${target}" is not an active label`);
      }
      image.add(target);
    }

    if (image.size !== allLabels.length) {
      throw new Error(`Invalid pi mapping at result ${index}: mapping must be bijective`);
    }
  }
}

export function buildFullGroup(allLabelsInput, validPiResultsInput, vLabelsInput = [], wLabelsInput = []) {
  const allLabels = [...allLabelsInput];
  const validPiResults = validPiResultsInput.map((result) => ({ ...result }));
  const vLabels = vLabelsInput instanceof Set ? vLabelsInput : new Set(vLabelsInput);
  const wLabels = wLabelsInput instanceof Set ? wLabelsInput : new Set(wLabelsInput);
  validatePiMappings(allLabels, validPiResults);
  const allLabelIndex = new Map(allLabels.map((label, index) => [label, index]));
  const generatorSelection = buildGeneratorSelectionProvenance(allLabels, validPiResults);

  const generatorsAll = [];
  for (const result of validPiResults) {
    const arr = allLabels.map((label) => allLabelIndex.get(result.pi[label]));
    const permutation = new Permutation(arr);
    if (!permutation.isIdentity) generatorsAll.push(permutation);
  }

  const fullGeneratorsAll = dedupGenerators(generatorsAll);
  const fullGenerators = minimalGenerators(fullGeneratorsAll);
  const fullElements = fullGenerators.length > 0
    ? dimino(fullGenerators)
    : (allLabels.length > 0 ? [Permutation.identity(allLabels.length)] : []);
  const fullOrder = fullElements.length;
  const fullDegree = allLabels.length;
  const fullGroupName = classifyGroupName(allLabels, fullGenerators, fullElements);
  const actionKinds = fullElements.map((element) => classifyPermutationAction(
    allLabels,
    element,
    vLabels,
    wLabels
  ));
  const actionSummary = {
    hasIdentity: actionKinds.some((result) => result.piKind === 'identity'),
    hasVOnly: actionKinds.some((result) => result.piKind === 'v-only'),
    hasWOnly: actionKinds.some((result) => result.piKind === 'w-only'),
    hasCross: actionKinds.some((result) => result.piKind === 'cross'),
    hasCorrelated: actionKinds.some((result) => result.piKind === 'correlated'),
  };

  return {
    allLabels,
    fullGeneratorsAll,
    fullGenerators,
    fullElements,
    fullOrder,
    fullDegree,
    fullGroupName,
    actionSummary,
    validPiResults,
    generatorSelection,
  };
}
