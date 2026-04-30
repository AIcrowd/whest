import StickyBar from './StickyBar.jsx';

export default {
  title: 'Section4/StickyBar',
  component: StickyBar,
  parameters: { layout: 'fullscreen' }, // sticky bar usually spans full width
};

// ---------------------------------------------------------------------------
// CrossS2Default
// V3.1 default microscope state: einsum('ij,k->ik', A, B)
// V = {i,k} (output labels), W = {j} (summation label)
// A is symmetric (S2), n=3, |G_pt|=2, α method = singleton
// ---------------------------------------------------------------------------
export const CrossS2Default = {
  args: {
    example: {
      formula: "einsum('ij,k->ik', A, B)",
      expression: {
        subscripts: 'ij,k',
        output: 'ik',
        operandNames: 'A, B',
      },
      operandNames: ['A', 'B'],
      variables: [
        {
          name: 'A',
          rank: 2,
          symmetry: 'symmetric',
          symAxes: [0, 1],
        },
        {
          name: 'B',
          rank: 1,
          symmetry: 'none',
        },
      ],
      perOpSymmetry: [],
    },
    group: {
      fullGroupName: 'S₂',
      vLabels: ['i', 'k'],
      wLabels: ['j'],
    },
    activeActId: 'setup',
    hoveredLabels: null,
    dimensionN: 3,
  },
};

// ---------------------------------------------------------------------------
// LargePreset
// Outer product 136×136: einsum('ij,kl->ijkl', A, B)
// Both operands symmetric, n=136, large group G = S₂ × S₂
// Represents the many-fields-populated production state
// ---------------------------------------------------------------------------
export const LargePreset = {
  args: {
    example: {
      formula: "einsum('ij,kl->ijkl', A, B)",
      expression: {
        subscripts: 'ij,kl',
        output: 'ijkl',
        operandNames: 'A, B',
      },
      operandNames: ['A', 'B'],
      variables: [
        {
          name: 'A',
          rank: 2,
          symmetry: 'symmetric',
          symAxes: [0, 1],
        },
        {
          name: 'B',
          rank: 2,
          symmetry: 'symmetric',
          symAxes: [0, 1],
        },
      ],
      perOpSymmetry: [],
    },
    group: {
      fullGroupName: 'S₂ × S₂',
      vLabels: ['i', 'j', 'k', 'l'],
      wLabels: [],
    },
    activeActId: 'decompose',
    hoveredLabels: null,
    dimensionN: 136,
  },
};

// ---------------------------------------------------------------------------
// LiveContractionStripCandidate
// Richer mock previewing C01 Live Contraction Strip layout (V3.1 §C01).
// Contraction: einsum('ijk,jl->il', T, M) — tensor × matrix
// T has S3 symmetry on first two axes, M is dense; summation over {j,k}
// α method = orbit-counted; active act = 'cost-savings' for the final act.
// hoveredLabels highlights 'i','l' (the free output labels).
// ---------------------------------------------------------------------------
export const LiveContractionStripCandidate = {
  args: {
    example: {
      formula: "einsum('ijk,jl->il', T, M)",
      expression: {
        subscripts: 'ijk,jl',
        output: 'il',
        operandNames: 'T, M',
      },
      operandNames: ['T', 'M'],
      variables: [
        {
          name: 'T',
          rank: 3,
          symmetry: 'symmetric',
          symAxes: [0, 1],
        },
        {
          name: 'M',
          rank: 2,
          symmetry: 'none',
        },
      ],
      perOpSymmetry: [],
    },
    group: {
      fullGroupName: 'S₂ ≀ S₁',
      vLabels: ['i', 'l'],
      wLabels: ['j', 'k'],
    },
    activeActId: 'cost-savings',
    hoveredLabels: new Set(['i', 'l']),
    dimensionN: 8,
  },
};
