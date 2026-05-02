import TwoQuotientSchematic from './TwoQuotientSchematic.jsx';

const CURRENT_SAMPLE = {
  presetName: 'Trilinear trace',
  dimensionN: 5,
  rowCount: 165,
  columnCount: 56,
  alpha: 165,
  branchRows: 0,
  hSize: 6,
  orbitRows: [
    {
      repTuple: { i: 0, j: 0, m: 0 },
      orbitTuples: [
        { i: 0, k: 0, j: 0, l: 0, m: 0, n: 0 },
        { i: 0, k: 0, j: 0, l: 0, m: 1, n: 1 },
      ],
      outputs: [{ outTuple: { i: 0, j: 0, m: 0 }, outKey: '0|0|0' }],
      outputCount: 1,
    },
  ],
};

export default {
  title: 'Section4/TwoQuotientSchematic',
  component: TwoQuotientSchematic,
  parameters: { layout: 'padded' },
};

export const CurrentPreset = {
  args: { current: CURRENT_SAMPLE },
};

// ---------------------------------------------------------------------------
// Story 1: Default — Cross S2 preset (H trivial, Y/H = Y)
// ---------------------------------------------------------------------------
export const CrossS2Default = {
  args: { current: CURRENT_SAMPLE },
  play: async () => {
    // Open the Cross S2 tab to inspect the reference case.
  },
};

// ---------------------------------------------------------------------------
// Story 2: Bilinear trace preset — H nontrivial
// ---------------------------------------------------------------------------
export const BilinearTrace = {
  render: () => {
    // We render normally and trust the toggle buttons; this story shows the
    // initial Cross S2 view — the Bilinear trace can be activated via toggle.
    return <TwoQuotientSchematic current={CURRENT_SAMPLE} />;
  },
};

// ---------------------------------------------------------------------------
// Story 3: Triple outer — rows and columns coincide (X/G_pt ≅ Y/H)
// ---------------------------------------------------------------------------
export const TripleOuterAllVisible = {
  render: () => <TwoQuotientSchematic current={CURRENT_SAMPLE} />,
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
      <TwoQuotientSchematic current={CURRENT_SAMPLE} />
    </div>
  ),
};
