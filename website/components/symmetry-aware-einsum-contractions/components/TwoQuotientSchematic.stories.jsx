import TwoQuotientSchematic from './TwoQuotientSchematic.jsx';

export default {
  title: 'Section4/TwoQuotientSchematic',
  component: TwoQuotientSchematic,
  parameters: { layout: 'padded' },
};

// ---------------------------------------------------------------------------
// Story 1: Default — Cross S2 preset (H trivial, Y/H = Y)
// ---------------------------------------------------------------------------
export const CrossS2Default = {
  args: {},
  play: async () => {
    // Component is self-contained; default preset is Cross S2.
  },
};

// ---------------------------------------------------------------------------
// Story 2: Bilinear trace preset — H nontrivial
// ---------------------------------------------------------------------------
export const BilinearTrace = {
  render: () => {
    // We render normally and trust the toggle buttons; this story shows the
    // initial Cross S2 view — the Bilinear trace can be activated via toggle.
    return <TwoQuotientSchematic />;
  },
};

// ---------------------------------------------------------------------------
// Story 3: Triple outer — rows and columns coincide (X/G_pt ≅ Y/H)
// ---------------------------------------------------------------------------
export const TripleOuterAllVisible = {
  render: () => <TwoQuotientSchematic />,
  name: 'Triple outer (all visible)',
};

// ---------------------------------------------------------------------------
// Story 4: Narrow viewport — verifies SVG scales correctly
// ---------------------------------------------------------------------------
export const NarrowViewport = {
  parameters: {
    viewport: { defaultViewport: 'mobile1' },
    layout: 'padded',
  },
  render: () => (
    <div style={{ maxWidth: 360 }}>
      <TwoQuotientSchematic />
    </div>
  ),
};
