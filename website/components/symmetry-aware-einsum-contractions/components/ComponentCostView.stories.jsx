import ComponentCostView from './ComponentCostView.jsx';

export default {
  title: 'Section4/ComponentCostView',
  component: ComponentCostView,
  parameters: { layout: 'padded' },
};

const noop = () => {};

// ---------------------------------------------------------------------------
// Story 1: SingleComponent
// Cross S2 — einsum "ij,jk->ik" with one label-interaction component {i,j,k}.
// Symmetry: S2 swaps i↔k (free labels); j is summed.
// Regime: singleton (|V|=1 free-axis rule doesn't apply; S2 acts on {i,k}
//   which are both visible, so functionalProjection fires: α = M = 3).
// With N=3 the orbit count M = 6 (Burnside on S2 over [3]^3 / swap(i,k)).
// ---------------------------------------------------------------------------

const crossS2OrbitRows = [
  { repTuple: { i: 0, j: 0, k: 0 }, outputs: [{ outTuple: { i: 0, k: 0 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 0, k: 1 }, outputs: [{ outTuple: { i: 0, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 1, k: 0 }, outputs: [{ outTuple: { i: 0, k: 0 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 1, k: 1 }, outputs: [{ outTuple: { i: 0, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 0, k: 1 }, outputs: [{ outTuple: { i: 1, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 1, k: 1 }, outputs: [{ outTuple: { i: 1, k: 1 }, coeff: 1 }] },
];

// componentData.components: one component spanning all labels {i, j, k}.
// shape='allVisible' because all labels appear in the output ("ij,jk->ik"
// includes i, j, k). groupName 'S2{i,k}' means S2 swaps i and k.
// accumulation.regimeId='functionalProjection' because S2 preserves V={i,k} as a set.
const crossS2ComponentData = {
  components: [
    {
      labels: ['i', 'j', 'k'],
      va: ['i', 'k'],       // free/output labels
      wa: ['j'],            // summed labels
      shape: 'mixed',
      groupName: 'S2{i,k}',
      elements: [
        { arr: [0, 1, 2] }, // identity: i→i, j→j, k→k
        { arr: [2, 1, 0] }, // swap: i→k, j→j, k→i
      ],
      generators: [{ arr: [2, 1, 0] }],
      order: 2,
      sizes: [3, 3, 3],
      multiplication: { count: 6 },
      accumulation: {
        count: 6,
        regimeId: 'functionalProjection',
        latex: String.raw`\alpha = M = |X / G|`,
        trace: [
          { regimeId: 'functionalProjection', decision: 'accepted', reason: 'G preserves V as a set' },
        ],
      },
    },
  ],
};

const crossS2CostModel = {
  orbitRows: crossS2OrbitRows,
};

export const SingleComponent = {
  args: {
    componentData: crossS2ComponentData,
    costModel: crossS2CostModel,
    dimensionN: 3,
    numTerms: 2,
    allLabels: ['i', 'j', 'k'],
    vLabels: ['i', 'k'],
    fullGenerators: [{ arr: [2, 1, 0] }],
    selectedOrbitIdx: -1,
    onSelectOrbit: noop,
    onGraphHover: noop,
    spotlightLeafIds: null,
    expressionInfo: {
      subscripts: ['ij', 'jk'],
      output: 'ik',
      operandNames: ['A', 'B'],
    },
  },
};

// ---------------------------------------------------------------------------
// Story 2: MultiComponent
// Triple outer S3 — einsum "ijk,jkl->il" with two independent components:
//   component 0: {j, k}  (summed-only pair sharing S2 symmetry)
//   component 1: {i, l}  (free-only pair, trivial group)
// This gives multiple component rows in the summary table.
// ---------------------------------------------------------------------------

const multiOrbitRows = [
  { repTuple: { i: 0, j: 0, k: 0, l: 0 }, outputs: [{ outTuple: { i: 0, l: 0 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 0, k: 1, l: 0 }, outputs: [{ outTuple: { i: 0, l: 0 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 0, k: 0, l: 1 }, outputs: [{ outTuple: { i: 0, l: 1 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 0, k: 0, l: 0 }, outputs: [{ outTuple: { i: 1, l: 0 }, coeff: 1 }] },
];

const multiComponentData = {
  components: [
    // Component 0: {j, k} — summed labels only, S2 symmetry swapping j↔k
    {
      labels: ['j', 'k'],
      va: [],               // no free labels in this component
      wa: ['j', 'k'],       // both summed
      shape: 'allSummed',
      groupName: 'S2{j,k}',
      elements: [
        { arr: [0, 1] },    // identity
        { arr: [1, 0] },    // swap j↔k
      ],
      generators: [{ arr: [1, 0] }],
      order: 2,
      sizes: [4, 4],
      multiplication: { count: 10 },
      accumulation: {
        count: 10,
        regimeId: 'functionalProjection',
        latex: String.raw`\alpha = M`,
        trace: [
          { regimeId: 'functionalProjection', decision: 'accepted', reason: 'G fixes V trivially (V is empty)' },
        ],
      },
    },
    // Component 1: {i, l} — free labels only, trivial group (no symmetry)
    {
      labels: ['i', 'l'],
      va: ['i', 'l'],       // both free/output
      wa: [],               // no summed labels
      shape: 'allVisible',
      groupName: 'trivial',
      elements: [
        { arr: [0, 1] },    // identity only
      ],
      generators: [],
      order: 1,
      sizes: [3, 3],
      multiplication: { count: 9 },
      accumulation: {
        count: 9,
        regimeId: 'functionalProjection',
        latex: String.raw`\alpha = M`,
        trace: [
          { regimeId: 'functionalProjection', decision: 'accepted', reason: 'trivial group — every tuple is its own orbit' },
        ],
      },
    },
  ],
};

const multiCostModel = {
  orbitRows: multiOrbitRows,
};

export const MultiComponent = {
  args: {
    componentData: multiComponentData,
    costModel: multiCostModel,
    dimensionN: 4,
    numTerms: 2,
    allLabels: ['i', 'j', 'k', 'l'],
    vLabels: ['i', 'l'],
    fullGenerators: [{ arr: [0, 2, 1, 3] }], // swap j(pos 1)↔k(pos 2) in full label array
    selectedOrbitIdx: -1,
    onSelectOrbit: noop,
    onGraphHover: noop,
    spotlightLeafIds: null,
    expressionInfo: {
      subscripts: ['ijk', 'jkl'],
      output: 'il',
      operandNames: ['A', 'B'],
    },
  },
};

// ---------------------------------------------------------------------------
// Story 3: UnavailableState
// A component where α is unavailable per V3.1 §B.9 exactness contract:
// the brute-force pair-touch budget was exceeded, so no regime fires and
// the accumulation count is null. The "Unavailable" amber badge appears.
//
// Example: large C3 component (C3 on {i,j,k} summed — 3^5 × 6 pair-touches
// exceed the 1,500,000 budget, so bruteForceOrbit refuses and no closed form
// matches).
// ---------------------------------------------------------------------------

const unavailableComponentData = {
  components: [
    {
      labels: ['i', 'j', 'k', 'l', 'm'],
      va: ['i'],            // one free label
      wa: ['j', 'k', 'l', 'm'], // four summed labels
      shape: 'mixed',
      groupName: 'C3{i,j,k}',
      elements: [
        { arr: [0, 1, 2, 3, 4] },  // identity
        { arr: [1, 2, 0, 3, 4] },  // rotate i→j→k→i
        { arr: [2, 0, 1, 3, 4] },  // rotate²
      ],
      generators: [{ arr: [1, 2, 0, 3, 4] }],
      order: 3,
      sizes: [8, 8, 8, 8, 8],
      multiplication: { count: null }, // also unavailable — orbit count not computed
      accumulation: {
        count: null,       // null = α unavailable per exactness contract
        regimeId: null,
        latex: null,
        trace: [
          { regimeId: 'functionalProjection', decision: 'refused', reason: 'G does not preserve V as a set' },
          { regimeId: 'singleton', decision: 'refused', reason: '|V| ≠ 1' },
          { regimeId: 'young', decision: 'refused', reason: 'group is not the full symmetric group' },
          { regimeId: 'partitionCount', decision: 'refused', reason: 'typed partition budget exceeded' },
          {
            regimeId: 'bruteForceOrbit',
            decision: 'refused',
            reason: '|X|·|G| = 98304 > 1,500,000 budget; exact count withheld',
          },
          { regimeId: 'fallthrough', decision: 'refused', reason: 'no regime accepted' },
        ],
      },
    },
  ],
};

const unavailableCostModel = {
  orbitRows: [], // no orbit rows — brute-force was not run
};

export const UnavailableState = {
  args: {
    componentData: unavailableComponentData,
    costModel: unavailableCostModel,
    dimensionN: 8,
    numTerms: 2,
    allLabels: ['i', 'j', 'k', 'l', 'm'],
    vLabels: ['i'],
    fullGenerators: [{ arr: [1, 2, 0, 3, 4] }],
    selectedOrbitIdx: -1,
    onSelectOrbit: noop,
    onGraphHover: noop,
    spotlightLeafIds: null,
    expressionInfo: {
      subscripts: ['ijklm', 'jklm'],
      output: 'i',
      operandNames: ['A', 'B'],
    },
  },
};
