import TypedPartitionDemo from './TypedPartitionDemo.jsx';
import { Permutation } from '../engine/permutation.js';

export default {
  title: 'Section8/TypedPartitionDemo',
  component: TypedPartitionDemo,
  parameters: { layout: 'padded' },
};

// ---------------------------------------------------------------------------
// Empty — no component data, component returns null (smoke test for the guard)
// ---------------------------------------------------------------------------
export const Empty = {
  args: {
    componentData: null,
    costModel: null,
  },
};

// ---------------------------------------------------------------------------
// CrossC3Pattern — Cross C3, einsum('abc->ab', T) with T cyclic on (a,b,c)
//
// Labels: a=0, b=1, c=2
// Group: C3 = { e, (a b c), (a c b) } = { [0,1,2], [1,2,0], [2,0,1] }
// Visible (output) positions: va = ['a','b']  → indices [0,1]
// Summed position: 'c' → index 2
// Domain sizes: n=3 for all positions (homogeneous)
//
// Orbit representatives under C3:
//   abc   (all same block)         → partition [0,0,0]  key "0|0|0"
//   ab|c  (a=b, c separate)        → partition [0,0,1]  key "0|0|1"
//   ac|b  (a=c, b separate)        → partition [0,1,0]  key "0|1|0"
//   bc|a  (b=c, a separate)        → partition [0,1,1]  key "0|1|1"
//   a|b|c (all different)          → partition [0,1,2]  key "0|1|2"
//
// The component calls generateTypedSetPartitions then partitionOrbitReps to
// derive these from the raw group elements — we supply the group and sizes.
// ---------------------------------------------------------------------------
const c3Elements = [
  new Permutation([0, 1, 2]), // identity
  new Permutation([1, 2, 0]), // a→b, b→c, c→a
  new Permutation([2, 0, 1]), // a→c, b→a, c→b
];

const crossC3ComponentData = {
  components: [
    {
      labels: ['a', 'b', 'c'],
      // va = output/visible labels
      va: ['a', 'b'],
      // sizes[i] = domain size of labels[i]; all 3 for n=3
      sizes: [3, 3, 3],
      elements: c3Elements,
      accumulation: {
        regimeId: 'partitionCount',
        trace: [],
      },
    },
  ],
};

export const CrossC3Pattern = {
  args: {
    componentData: crossC3ComponentData,
    // costModel is only checked for truthiness; pass a minimal sentinel
    costModel: {},
  },
};

// ---------------------------------------------------------------------------
// LargePattern — heterogeneous domain sizes + a larger symmetry group
//
// Simulates einsum('abcd->ab', T) where:
//   labels: a=0, b=1, c=2, d=3
//   domain sizes: a,b → n=4; c,d → n=5  (heterogeneous)
//   Group: C4 × C2 on positions — use a realistic 8-element group
//   Visible positions: va = ['a','b']
//
// This exercises the multi-domain falling-factorial path in typedLabelingCount
// and the table / chip-strip pagination (VISIBLE_LIMIT = 8) since typed
// partitions for 4 positions can exceed 8 orbit reps.
// ---------------------------------------------------------------------------
const c4c2Elements = [
  // C4 part acting on positions 0,1,2,3: rotate (0→1→2→3→0) combined with
  // C2 swapping positions 2 and 3 to form an 8-element group.
  new Permutation([0, 1, 2, 3]), // identity
  new Permutation([1, 2, 3, 0]), // r1
  new Permutation([2, 3, 0, 1]), // r2
  new Permutation([3, 0, 1, 2]), // r3
  new Permutation([0, 1, 3, 2]), // s  (swap last two)
  new Permutation([1, 2, 0, 3]), // rs
  new Permutation([2, 3, 1, 0]), // r2s
  new Permutation([3, 0, 2, 1]), // r3s
];

const largePatternComponentData = {
  components: [
    {
      labels: ['a', 'b', 'c', 'd'],
      va: ['a', 'b'],
      // a,b share domain size 4; c,d share domain size 5
      sizes: [4, 4, 5, 5],
      elements: c4c2Elements,
      accumulation: {
        regimeId: 'bruteForceOrbit',
        trace: [],
      },
    },
  ],
};

export const LargePattern = {
  args: {
    componentData: largePatternComponentData,
    costModel: {},
  },
};
