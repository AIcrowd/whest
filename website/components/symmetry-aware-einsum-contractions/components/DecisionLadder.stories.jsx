import DecisionLadder from './DecisionLadder.jsx';

export default {
  title: 'Section7/DecisionLadder',
  component: DecisionLadder,
  parameters: { layout: 'padded' },
};

// ---------------------------------------------------------------------------
// Default — no preset active, tree idle.
// Shows the two-stage classification tree with no highlighted path.
// ---------------------------------------------------------------------------
export const Default = {
  args: {},
};

// ---------------------------------------------------------------------------
// ActiveLeaf — partition count leaf highlighted (cross-c3 style preset).
// The partitionCount leaf in Stage 2 gets the active same-color ring.
// ---------------------------------------------------------------------------
export const ActiveLeaf = {
  args: {
    activeLeafIds: ['partitionCount'],
  },
};

// ---------------------------------------------------------------------------
// SpotlightLeaf — functionalProjection spotlit via label-interaction graph
// hover. Shows the stronger halo (v_free color) applied by LabelInteractionGraph
// cross-highlight. Combined with an active leaf so both states are visible.
// ---------------------------------------------------------------------------
export const SpotlightLeaf = {
  args: {
    activeLeafIds: ['functionalProjection'],
    spotlightLeafIds: ['functionalProjection'],
  },
};

// ---------------------------------------------------------------------------
// AlphaMethodHighlight — activeAlphaMethod prop set to 'singleton'.
// The singleton leaf gets a coral outline (the α-method bus highlight).
// This simulates StickyBar's α-method badge hover emitting 'singleton'.
// ---------------------------------------------------------------------------
export const AlphaMethodHighlight = {
  args: {
    activeAlphaMethod: 'singleton',
  },
};
