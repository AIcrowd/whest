import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// L0.5 of the V3.1 narrative migration ships Storybook stories for the 6
// priority "visual sibling" components that L5 component plans will cite as
// design references. The remaining 4 components from the spec's L0.5 list
// (SigmaLoop, ExpressionLevelModal, CaseBadge, WreathStructureView) are
// deferred to follow-up — added on-demand when an L5 plan cites them.
//
// See: .aicrowd/superpowers/specs/2026-05-01-v3-narrative-migration-design.md
//      .aicrowd/superpowers/plans/2026-05-01-l05-storybook-bootstrap.md
const REQUIRED_STORIES = [
  'components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.stories.jsx',
  'components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.stories.jsx',
  'components/symmetry-aware-einsum-contractions/components/BipartiteGraph.stories.jsx',
  'components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.stories.jsx',
  'components/symmetry-aware-einsum-contractions/components/ComponentCostView.stories.jsx',
  'components/symmetry-aware-einsum-contractions/components/StickyBar.stories.jsx',
];

test('L0.5 priority visual-sibling components have Storybook stories', () => {
  for (const rel of REQUIRED_STORIES) {
    const p = resolve(__dirname, rel);
    assert.ok(existsSync(p), `missing stories file: ${rel}`);
  }
});

test('Storybook config files exist (.storybook/main.{js,ts} + preview)', () => {
  const mainJs = resolve(__dirname, '.storybook/main.js');
  const mainTs = resolve(__dirname, '.storybook/main.ts');
  const previewJsx = resolve(__dirname, '.storybook/preview.jsx');
  const previewTsx = resolve(__dirname, '.storybook/preview.tsx');
  const previewTs = resolve(__dirname, '.storybook/preview.ts');
  assert.ok(existsSync(mainJs) || existsSync(mainTs), 'missing .storybook/main.{js,ts}');
  assert.ok(
    existsSync(previewJsx) || existsSync(previewTsx) || existsSync(previewTs),
    'missing .storybook/preview.{jsx,tsx,ts}',
  );
});
