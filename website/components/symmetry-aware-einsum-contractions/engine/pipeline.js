import {
  buildBipartite,
  buildIncidenceMatrix,
  runSigmaLoop,
  buildGroup,
  computeBurnside,
} from './algorithm.js';
import { parseCycleNotation } from './cycleParser.js';
import { computeExactCostModel } from './costModel.js';
import { decomposeClassifyAndCount } from './componentDecomposition.js';

function normalizeExample(example) {
  if (!example) return null;
  if (Array.isArray(example.subscripts)) return example;

  const { expression, variables = [] } = example;
  if (!expression?.subscripts || !expression?.operandNames) return example;

  const subscripts = expression.subscripts.split(',').map((item) => item.trim());
  const operandNames = expression.operandNames.split(',').map((item) => item.trim());
  const perOpSymmetry = operandNames.map((operandName) => {
    const variable = variables.find((entry) => entry.name === operandName);
    if (!variable || variable.symmetry === 'none') return null;

    const axes = variable.symAxes || Array.from({ length: variable.rank }, (_, idx) => idx);
    if (variable.symmetry === 'symmetric' && axes.length === variable.rank) {
      return 'symmetric';
    }

    if (variable.symmetry === 'custom' && variable.generators) {
      const { generators } = parseCycleNotation(variable.generators);
      return { type: 'custom', axes, generators };
    }

    return { type: variable.symmetry, axes };
  });

  return {
    ...example,
    subscripts,
    output: expression.output ?? '',
    operandNames,
    perOpSymmetry: perOpSymmetry.some((entry) => entry !== null) ? perOpSymmetry : null,
  };
}

export function analyzeExample(example, dimensionN) {
  const normalizedExample = normalizeExample(example);
  const graph = buildBipartite(normalizedExample);
  const matrixData = buildIncidenceMatrix(graph);
  const sigmaResults = runSigmaLoop(graph, matrixData, normalizedExample);
  const symmetry = buildGroup(sigmaResults, graph, normalizedExample);
  const sizes = symmetry.allLabels.map(() => dimensionN);
  const componentData = decomposeClassifyAndCount(
    symmetry.allLabels,
    symmetry.vLabels,
    symmetry.wLabels,
    symmetry.fullGenerators,
    symmetry.fullElements,
    sizes,
  );
  const burnside = computeBurnside(symmetry, dimensionN);
  const costModel = computeExactCostModel({
    labels: symmetry.allLabels,
    vLabels: symmetry.vLabels,
    groupElements: symmetry.fullElements,
    dimensionN,
    numTerms: normalizedExample.subscripts.length,
  });

  return { graph, matrixData, sigmaResults, symmetry, componentData, burnside, costModel };
}
