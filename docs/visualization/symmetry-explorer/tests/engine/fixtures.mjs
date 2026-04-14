export function makeExample(subscripts, output, operandNames, perOpSymmetry = null) {
  return {
    subscripts: subscripts.split(',').map(s => s.trim()),
    output,
    operandNames: operandNames.split(',').map(s => s.trim()),
    perOpSymmetry,
  };
}

export const crossVwRelabelingExample = makeExample(
  'ab,ab',
  'b',
  'S, S',
  ['symmetric', 'symmetric']
);

export const costModelExample = makeExample(
  'ab,ab',
  'b',
  'S, S',
  ['symmetric', 'symmetric']
);

export const CASES = {
  pureWS2: {
    name: 'pure W S2',
    dimensionN: 3,
    example: makeExample('ijk', 'i', 'T', [{ type: 'symmetric', axes: [1, 2] }]),
    expected: {
      piKinds: ['w-only'],
      fullGroupName: 'S2{j,k}',
      orbitCount: 18,
      evaluationCost: 0,
      reductionCost: 18,
    },
  },
  crossS2: {
    name: 'cross S2',
    dimensionN: 3,
    example: makeExample('ij,k', 'ik', 'A, B', ['symmetric', null]),
    expected: {
      piKinds: ['cross'],
      fullGroupName: 'S2{i,j}',
      orbitCount: 18,
      evaluationCost: 18,
      reductionCost: 27,
    },
  },
  mixedS3: {
    name: 'mixed S3',
    dimensionN: 3,
    example: makeExample('ijk', 'i', 'T', ['symmetric']),
    expected: {
      piKinds: ['cross', 'w-only'],
      fullGroupName: 'S3{i,j,k}',
      orbitCount: 10,
      evaluationCost: 0,
      reductionCost: 18,
    },
  },
  mixedC3: {
    name: 'mixed C3',
    dimensionN: 3,
    example: makeExample('ijk', 'i', 'T', [{ type: 'cyclic', axes: [0, 1, 2] }]),
    expected: {
      piKinds: ['cross'],
      fullGroupName: 'C3{i,j,k}',
      orbitCount: 11,
      evaluationCost: 0,
      reductionCost: 21,
    },
  },
};
