import TuplePatternMeter from './TuplePatternMeter.jsx';

export default {
  title: 'Section8/TuplePatternMeter',
  component: TuplePatternMeter,
  parameters: { layout: 'padded' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Mock componentData fixtures.
// The shape mirrors what App passes in: components[0].sizes drives typed-
// partition enumeration; components[0].elements is the pointwise group used
// to quotient the partitions into orbits. Identity-only elements means
// pattern orbits collapse to typed patterns (the "no symmetry" reading).
// ─────────────────────────────────────────────────────────────────────────────

// Cross S2 small case: 3 labels, identity-only group → orbits == patterns.
// Bell(3) = 5 typed patterns.
const SMALL_COMPONENT = {
  components: [
    {
      sizes: [3, 3, 3],
      elements: [{ arr: [0, 1, 2] }],
      labels: ['i', 'j', 'k'],
      va: ['i'],
    },
  ],
};

// Cross S2 with a transposition: 3 labels, identity + (i,j) swap.
// 5 typed patterns; orbits should be 4 (the i↔j-swap merges two patterns).
const CROSS_S2_COMPONENT = {
  components: [
    {
      sizes: [3, 3, 3],
      elements: [{ arr: [0, 1, 2] }, { arr: [1, 0, 2] }],
      labels: ['i', 'j', 'k'],
      va: ['i'],
    },
  ],
};

// Larger feasible case: 5 labels, mix of domain sizes 4 and 4 and 4 and 4 and 4.
// Bell(5) = 52 typed patterns. Orbits depend on group; we use a small cyclic
// group on the first three labels to keep counts interesting.
const LARGE_FEASIBLE_COMPONENT = {
  components: [
    {
      sizes: [4, 4, 4, 4, 4],
      elements: [
        { arr: [0, 1, 2, 3, 4] },
        { arr: [1, 2, 0, 3, 4] },
        { arr: [2, 0, 1, 3, 4] },
      ],
      labels: ['a', 'b', 'c', 'd', 'e'],
      va: ['a', 'b'],
    },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// Default — Cross S2 small n. All four bars present; the savings between
// dense and pattern orbits should be visually obvious.
// ─────────────────────────────────────────────────────────────────────────────
export const Default = {
  args: {
    dimensionN: 3,
    allLabels: ['i', 'j', 'k'],
    groupSize: 2,
    componentData: CROSS_S2_COMPONENT,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Small — n=2, identity group, 3 labels. Bars are tiny but the relative
// magnitudes are still visible thanks to log scaling.
// ─────────────────────────────────────────────────────────────────────────────
export const Small = {
  args: {
    dimensionN: 2,
    allLabels: ['i', 'j', 'k'],
    groupSize: 1,
    componentData: SMALL_COMPONENT,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// LargeDimension — n=10, 5 labels, cyclic group of order 3.
// dense = 10^5 = 100 000; tupleGroup = 300 000; typed patterns = 52;
// pattern orbits ≪ 52. Bars span 4 orders of magnitude — log scaling
// keeps the smaller pattern bars visible.
// ─────────────────────────────────────────────────────────────────────────────
export const LargeDimension = {
  args: {
    dimensionN: 10,
    allLabels: ['a', 'b', 'c', 'd', 'e'],
    groupSize: 3,
    componentData: LARGE_FEASIBLE_COMPONENT,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// OverBudget — typed-pattern enumeration unavailable. The last two bars
// render "—" and the caption appends "(when pattern budget passes)".
// We trigger this by passing componentData = null, simulating a preset
// where the partition enumerate is too large for interactive feedback.
// ─────────────────────────────────────────────────────────────────────────────
export const OverBudget = {
  args: {
    dimensionN: 8,
    allLabels: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    groupSize: 24,
    componentData: null,
  },
};
