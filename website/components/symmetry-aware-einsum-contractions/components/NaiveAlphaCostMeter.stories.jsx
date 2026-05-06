import NaiveAlphaCostMeter from './NaiveAlphaCostMeter.jsx';

export default {
  title: 'Section7/NaiveAlphaCostMeter',
  component: NaiveAlphaCostMeter,
  parameters: { layout: 'padded' },
};

// ---------------------------------------------------------------------------
// Small — n=2, 2 labels, trivial group → "small" tier
// Tuple space = 2^2 = 4, groupSize = 1, touches = 4 < 1000.
// ---------------------------------------------------------------------------
export const Small = {
  args: {
    dimensionN: 2,
    allLabels: ['i', 'j'],
    groupSize: 1,
    hSize: 1,
  },
};

// ---------------------------------------------------------------------------
// Feasible — Cross S2 default n=3, 3 labels, |G|=2, |H|=1.
// Tuple space = 3^3 = 27, touches = 27 * 2 = 54 < 100 000.
// ---------------------------------------------------------------------------
export const Feasible = {
  args: {
    dimensionN: 3,
    allLabels: ['i', 'j', 'k'],
    groupSize: 2,
    hSize: 1,
  },
};

// ---------------------------------------------------------------------------
// Expensive — n=5, 4 labels, |G|=6, touches = 5^4 * 6 = 3750 < 1M.
// (Falls into "expensive" tier at ~1e5–1e6 depending on exact numbers.)
// ---------------------------------------------------------------------------
export const Expensive = {
  args: {
    dimensionN: 5,
    allLabels: ['a', 'b', 'c', 'd'],
    groupSize: 6,
    hSize: 6,
  },
};

// ---------------------------------------------------------------------------
// Unavailable — n=8, 5 labels, |G|=24 → huge tuple space.
// Tuple space = 8^5 = 32768, touches = 32768 * 24 = 786 432 → approaching 1M.
// With n=10: 10^5 * 24 = 2 400 000 → "unavailable".
// ---------------------------------------------------------------------------
export const Unavailable = {
  args: {
    dimensionN: 10,
    allLabels: ['a', 'b', 'c', 'd', 'e'],
    groupSize: 24,
    hSize: 24,
  },
};

// ---------------------------------------------------------------------------
// NoLabels — edge case: empty label list (guard against division by zero)
// ---------------------------------------------------------------------------
export const NoLabels = {
  args: {
    dimensionN: 3,
    allLabels: [],
    groupSize: 1,
    hSize: 1,
  },
};
