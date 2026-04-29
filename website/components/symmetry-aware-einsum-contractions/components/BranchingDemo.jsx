import { useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor, notationLatex } from '../lib/notationSystem.js';

const TITLE = 'When projection branches, watch one orbit fan out';
const DECK = 'Branching = one product orbit reaches multiple stored output representatives. Pick a view, pick an orbit, see it.';

const INTRO_PARAGRAPHS = [
  // Relocated from content/main/section4.js intro ¶3 (verbatim).
  'A product-orbit representative can contain many full index assignments; their projections to the visible labels may land in one stored output representative or several. $\\alpha$ counts one update per stored output representative reached. The subtlety is that projection is not always a function from product orbits to output representatives: sometimes one product orbit reaches several. That is exactly why accumulation counting is harder than multiplication counting.',
  // Relocated from PartitionCountingExplainer.jsx BODY ¶1 (verbatim).
  'A product orbit may contain many full assignments. When those assignments are projected to the output labels, they may reach one stored output representative or several. Enumerating every concrete assignment is correct but can be wasteful.',
  // Relocated from PartitionCountingExplainer.jsx BODY ¶2 (verbatim).
  'Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives, so the accumulation count needs an extra reach factor on top of the orbit count.',
];

export default function BranchingDemo({
  componentData,
  costModel,
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
}) {
  const themeId = getActiveExplorerThemeId();
  const [activeView, setActiveView] = useState('fan');

  if (!componentData || !costModel) return null;

  return (
    <section
      id="branching-demo"
      className="mx-auto w-full max-w-[var(--prose-max)] rounded-xl border bg-white px-6 py-6 shadow-sm md:px-8 md:py-7 scroll-mt-24"
      style={{ borderColor: explorerThemeColor(themeId, 'border') }}
      aria-labelledby="branching-demo-title"
    >
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        Branching demo
      </div>
      <h3
        id="branching-demo-title"
        className="font-heading text-[20px] font-semibold leading-tight"
        style={{ color: explorerThemeColor(themeId, 'ink') }}
      >
        {TITLE}
      </h3>
      <p className="mt-2 max-w-[70ch] font-serif text-[15px] italic leading-7" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        {DECK}
      </p>

      <div className="mt-5 space-y-4 max-w-[78ch] font-serif text-[16px] leading-[1.75]" style={{ color: explorerThemeColor(themeId, 'body') }}>
        {INTRO_PARAGRAPHS.map((paragraph, idx) => (
          <p key={idx}>
            <InlineMathText>{paragraph}</InlineMathText>
          </p>
        ))}
      </div>

      <div className="mt-6 text-[12px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        view-mode tabs and the active renderer land in Task 2 onward; orbit selector + live α land in Task 6.
      </div>
    </section>
  );
}
