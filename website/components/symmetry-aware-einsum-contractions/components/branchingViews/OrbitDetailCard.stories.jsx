import OrbitDetailCard from './OrbitDetailCard.jsx';

export default {
  title: 'Section4/OrbitDetailCard',
  component: OrbitDetailCard,
  parameters: { layout: 'fullscreen' }, // floating mode renders position:fixed; need fullscreen
};

const noop = () => {};

// ---------------------------------------------------------------------------
// Shared mock data — Cross-S2 style: einsum "ij,jk->ik" with 2 orbit rows
// and 2 representative columns. Row index 0 is a simple 1-branch orbit;
// row index 1 (hover.row=1) branches to 2 output reps.
// ---------------------------------------------------------------------------

const expressionInfo = {
  subscripts: ['ij', 'jk'],
  output: 'ik',
  operandNames: ['A', 'B'],
};

// componentInfo.vLabels are the "V" labels that π_V projects onto (output indices).
const componentInfo = {
  vLabels: ['i', 'k'],
};

// orbitRows: each has repTuple (the canonical member), orbitSize, orbitTuples, outputs.
// outputs[*].outTuple must only contain keys from vLabels.
const orbitRows = [
  {
    repTuple: { i: 0, j: 0, k: 0 },
    orbitSize: 2,
    orbitTuples: [
      { i: 0, j: 0, k: 0 },
      { i: 0, j: 1, k: 0 },
    ],
    outputs: [
      { outTuple: { i: 0, k: 0 }, coeff: 1 },
    ],
  },
  {
    repTuple: { i: 1, j: 0, k: 1 },
    orbitSize: 3,
    orbitTuples: [
      { i: 1, j: 0, k: 1 },
      { i: 1, j: 1, k: 2 },
      { i: 2, j: 0, k: 1 },
    ],
    outputs: [
      { outTuple: { i: 1, k: 1 }, coeff: 1 },
      { outTuple: { i: 1, k: 2 }, coeff: 1 },
    ],
  },
];

// reps: derived from the union of all outTuples across orbitRows.
// Each rep has a `k` (JSON key) and a `tuple` matching componentInfo.vLabels.
const reps = [
  { k: JSON.stringify({ i: 0, k: 0 }), tuple: { i: 0, k: 0 } },
  { k: JSON.stringify({ i: 1, k: 1 }), tuple: { i: 1, k: 1 } },
  { k: JSON.stringify({ i: 1, k: 2 }), tuple: { i: 1, k: 2 } },
];

// cells[row][col]: coeff if the orbit reaches that rep, else null.
// Row 0 reaches rep 0 only; row 1 reaches reps 1 and 2.
const cells = [
  [1, null, null],
  [null, 1, 1],
];

// A fake ref — the IntersectionObserver in floating mode null-checks .current,
// so { current: null } is safe and avoids the observer running at all.
const fakeMatrixRef = { current: null };

// ---------------------------------------------------------------------------
// Story 1: Empty — hover=null, component renders nothing (returns null).
// ---------------------------------------------------------------------------
export const Empty = {
  args: {
    hover: null,
    orbitRows,
    reps,
    cells,
    expressionInfo,
    componentInfo,
    onDismiss: noop,
    matrixRef: fakeMatrixRef,
    mode: 'floating',
  },
};

// ---------------------------------------------------------------------------
// Story 2: FloatingHover — hover at row=1 col=1, position:fixed near click.
// Row 1 branches to 2 reps so the "branches to N cells" caption fires.
// ---------------------------------------------------------------------------
export const FloatingHover = {
  args: {
    hover: { row: 1, col: 1, clickX: 400, clickY: 300 },
    mode: 'floating',
    orbitRows,
    reps,
    cells,
    expressionInfo,
    componentInfo,
    onDismiss: noop,
    matrixRef: fakeMatrixRef,
  },
};

// ---------------------------------------------------------------------------
// Story 3: InlineMode — same orbit data, mode="inline" (no position:fixed,
// no Esc handler, no IntersectionObserver). Used inside OrbitRepMatrixModal.
// ---------------------------------------------------------------------------
export const InlineMode = {
  args: {
    hover: { row: 1, col: 1, clickX: 0, clickY: 0 },
    mode: 'inline',
    orbitRows,
    reps,
    cells,
    expressionInfo,
    componentInfo,
    onDismiss: noop,
    matrixRef: fakeMatrixRef,
  },
};
