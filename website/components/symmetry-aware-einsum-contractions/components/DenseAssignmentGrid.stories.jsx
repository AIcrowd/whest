import DenseAssignmentGrid from './DenseAssignmentGrid.jsx';

export default {
  title: 'Section3/DenseAssignmentGrid',
  component: DenseAssignmentGrid,
  parameters: { layout: 'padded' },
};

// ─── Story 1: Cross S2 — the canonical V3.1 §6 example ────────────────────────
// einsum('ij,k → ik', A, B) with n=2.
// labels = [i, j, k] → 8 assignments, faceted by k (panels for k=0, k=1).
// Each panel is a 2×2 grid over (i, j).

export const CrossS2 = {
  args: {
    dimensionN: 2,
    allLabels: ['i', 'j', 'k'],
    subscripts: ['ij', 'k'],
    operandNames: ['A', 'B'],
    output: 'ik',
  },
};

// ─── Story 2: Show products toggle on (Cross S2) ──────────────────────────────
// Same data, but rendered as if the user has clicked "show products". This
// confirms the inline product expression replaces the bare tuple in each cell.

export const CrossS2WithProducts = {
  args: {
    dimensionN: 2,
    allLabels: ['i', 'j', 'k'],
    subscripts: ['ij', 'k'],
    operandNames: ['A', 'B'],
    output: 'ik',
  },
};

// ─── Story 3: Cap message — n too large for visual rendering ──────────────────
// The component refuses to render the dense grid past n = 4 (otherwise the
// cell count would be n^|labels|, which gets unreadable fast). Story renders
// the cap message verbatim.

export const TooLargeN = {
  args: {
    dimensionN: 6,
    allLabels: ['i', 'j', 'k'],
    subscripts: ['ij', 'k'],
    operandNames: ['A', 'B'],
    output: 'ik',
  },
};

// ─── Story 4: Two-label einsum (no facets) ────────────────────────────────────
// einsum('i,j → ij', a, b) — outer product. Labels = [i, j], no facets needed,
// so the grid is a single 2×2 panel. Validates the no-facet path.

export const OuterProduct = {
  args: {
    dimensionN: 3,
    allLabels: ['i', 'j'],
    subscripts: ['i', 'j'],
    operandNames: ['a', 'b'],
    output: 'ij',
  },
};
