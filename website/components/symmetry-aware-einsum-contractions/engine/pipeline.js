import {
  buildBipartite,
  buildIncidenceMatrix,
  runSigmaLoop,
  buildGroup,
  computeBurnside,
} from './algorithm.js';
import { parseCycleNotation } from './cycleParser.js';
import { computeExactCostModel, aggregateComponentCosts } from './costModel.js';
import { decomposeClassifyAndCount } from './componentDecomposition.js';
import { computeLabelClusters } from './sizeAware/labelClusters.js';
import { buildExpressionGroup } from './expressionGroup.js';

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
  const { results: sigmaResults, wreathElements } = runSigmaLoop(graph, matrixData, normalizedExample);
  const symmetry = buildGroup(sigmaResults, graph, normalizedExample);
  const rawClusters = computeLabelClusters(symmetry.allLabels, symmetry.fullGenerators, dimensionN);
  const labelSizesOverride = example.labelSizes || {};
  const clusters = rawClusters.map((c) => ({
    ...c,
    size: Object.prototype.hasOwnProperty.call(labelSizesOverride, c.id)
      ? labelSizesOverride[c.id]
      : dimensionN,
  }));
  const sizeByLabel = new Map();
  for (const c of clusters) {
    for (const label of c.labels) sizeByLabel.set(label, c.size);
  }
  const sizes = symmetry.allLabels.map((l) => sizeByLabel.get(l) ?? dimensionN);
  const componentData = decomposeClassifyAndCount(
    symmetry.allLabels,
    symmetry.vLabels,
    symmetry.wLabels,
    symmetry.fullGenerators,
    symmetry.fullElements,
    sizes,
  );
  const burnside = computeBurnside(symmetry, dimensionN);
  const numTerms = normalizedExample.subscripts.length;
  const costModel = computeExactCostModel({
    labels: symmetry.allLabels,
    vLabels: symmetry.vLabels,
    groupElements: symmetry.fullElements,
    dimensionN,
    sizes,
    numTerms,
  });
  // Per-component aggregation drives the displayed μ and α.
  // costModel above is kept as a brute-force ground truth.
  const componentCosts = aggregateComponentCosts(componentData.components, numTerms);

  // Expression-level group: V-sub × S(W). Pedagogical only — not used for
  // compression. See engine/expressionGroup.js for the derivation.
  const expressionGroup = buildExpressionGroup({
    perTupleElements: symmetry.fullElements ?? [],
    vLabels: symmetry.vLabels ?? [],
    wLabels: symmetry.wLabels ?? [],
    allLabels: symmetry.allLabels ?? [],
  });

  return {
    graph,
    matrixData,
    sigmaResults,
    symmetry: {
      ...symmetry,
      wreathElements,
      identicalGroups: graph.identicalGroups,
    },
    componentData,
    burnside,
    costModel,
    componentCosts,
    clusters,
    expressionGroup,
  };
}
