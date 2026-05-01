import LoopMorphStepper from './LoopMorphStepper.jsx';

export default {
  title: 'Section3/LoopMorphStepper',
  component: LoopMorphStepper,
  parameters: { layout: 'padded' },
};

// ─── Story 1: Default — fresh stepper at step 1 (dense assignments) ───────────
// The default boot state is step 1 of 5: the dense pseudocode is fully
// opaque while the representative pseudocode shows in muted form. The reader
// hits Next or arrow keys to walk the morph.

export const Default = {
  args: {},
};

// ─── Story 2: Centered narrow column — narrative-flow placement ───────────────
// The stepper is mounted inline in §3 between DenseAssignmentGrid and
// ComponentCostView, so it lives inside a narrow text column. This story
// stacks it inside a constrained max-width container to confirm the
// side-by-side columns and stepper controls remain legible at narrative width.

export const NarrowColumn = {
  render: (args) => (
    <div style={{ maxWidth: '720px', margin: '0 auto' }}>
      <LoopMorphStepper {...args} />
    </div>
  ),
  args: {},
};

// ─── Story 3: Two-up — visual diff between two instances ──────────────────────
// Renders two independent steppers side-by-side. The reader can advance one
// while leaving the other parked, which mirrors how the morph is meant to
// be compared frame-to-frame in editorial review.

export const TwoUp = {
  render: (args) => (
    <div
      style={{
        display: 'grid',
        gap: '24px',
        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
      }}
    >
      <LoopMorphStepper {...args} />
      <LoopMorphStepper {...args} />
    </div>
  ),
  args: {},
};
