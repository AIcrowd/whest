import OrbitRepMatrix from './OrbitRepMatrix.jsx';

export default {
  title: 'Section4/OrbitRepMatrix',
  component: OrbitRepMatrix,
  parameters: { layout: 'padded' },
};

const noop = () => {};

// ─── Helper: build an orbitRow ────────────────────────────────────────────────
// orbitRows[i] = { repTuple: { i: number, j: number, ... }, outputs: [...] }
// outputs[k]   = { outTuple: { i: number, k: number, ... }, coeff: number }

// ─── Story 1: Empty ───────────────────────────────────────────────────────────
// Verifies the empty-state JSX ("no orbit data for this preset") renders without crash.

export const Empty = {
  args: {
    orbitRows: [],
    selectedOrbitIdx: -1,
    onSelectOrbit: noop,
    onHoverChange: noop,
    onHover: null,
    hover: null,
    onExpand: null,
    componentInfo: null,
    expressionInfo: null,
  },
};

// ─── Story 2: Populated ───────────────────────────────────────────────────────
// Small Cross-S2-style matrix: 9 orbits × 4 unique reps, with branching.
// Mirrors the i,j,k subscript pattern ("ij,jk->ik").
// vLabels = ['i','k'] → visible labels; 'j' is summed.

const s2OrbitRows = [
  { repTuple: { i: 0, j: 0, k: 0 }, outputs: [{ outTuple: { i: 0, k: 0 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 0, k: 1 }, outputs: [{ outTuple: { i: 0, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 1, k: 0 }, outputs: [{ outTuple: { i: 0, k: 0 }, coeff: 1 }] },
  { repTuple: { i: 0, j: 1, k: 1 }, outputs: [{ outTuple: { i: 0, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 0, k: 0 }, outputs: [{ outTuple: { i: 1, k: 0 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 0, k: 1 }, outputs: [{ outTuple: { i: 1, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 1, k: 0 }, outputs: [{ outTuple: { i: 1, k: 0 }, coeff: 1 }] },
  { repTuple: { i: 1, j: 1, k: 1 }, outputs: [{ outTuple: { i: 1, k: 1 }, coeff: 1 }] },
  { repTuple: { i: 2, j: 0, k: 0 }, outputs: [{ outTuple: { i: 2, k: 0 }, coeff: 2 }, { outTuple: { i: 2, k: 1 }, coeff: 1 }] },
];

export const Populated = {
  args: {
    orbitRows: s2OrbitRows,
    selectedOrbitIdx: -1,
    onSelectOrbit: noop,
    onHoverChange: noop,
    onHover: null,
    hover: null,
    onExpand: noop,
    componentInfo: { vLabels: ['i', 'k'], dimensionN: 2 },
    expressionInfo: { subscripts: ['ij', 'jk'], output: 'ik', operandNames: ['A', 'B'] },
  },
};

// ─── Story 3: LargePreset ─────────────────────────────────────────────────────
// ~60 rows × 60 cols — programmatically generated to confirm tick gutters and
// canvas scaling work correctly on dense matrices.
// Each orbit (i, j, k) maps to output tuple (i, k) with coefficient 1,
// giving a full 60-row × 36-col spread (6×6×6 orbits, 6×6 rep tuples).

const N = 6; // dimension N → indices 0..N-1 for i, j, k
const largeOrbitRows = [];
for (let i = 0; i < N; i++) {
  for (let j = 0; j < N; j++) {
    for (let k = 0; k < N; k++) {
      largeOrbitRows.push({
        repTuple: { i, j, k },
        outputs: [{ outTuple: { i, k }, coeff: 1 }],
      });
    }
  }
}

export const LargePreset = {
  args: {
    orbitRows: largeOrbitRows,
    selectedOrbitIdx: -1,
    onSelectOrbit: noop,
    onHoverChange: noop,
    onHover: null,
    hover: null,
    onExpand: noop,
    componentInfo: { vLabels: ['i', 'k'], dimensionN: N },
    expressionInfo: { subscripts: ['ij', 'jk'], output: 'ik', operandNames: ['A', 'B'] },
  },
};
