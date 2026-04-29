import { useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor, notationLatex } from '../lib/notationSystem.js';
import {
  generateTypedSetPartitions,
  partitionOrbitReps,
  typedLabelingCount,
  inducedBlockActionSize,
  inducedPrefixMaps,
  partitionKey,
  numBlocks,
} from '../engine/partition/typedPartitions.js';

const TITLE = 'When projection branches, count equality patterns';
const DECK = 'Partition counting is the exact compressed counter behind the general branching case.';

const INTRO_PARAGRAPHS = [
  // Relocated from content/main/section4.js intro ¶4 (verbatim).
  'When projection branches, the explorer can count exactly without expanding the full assignment grid by grouping assignments by typed equality pattern: which same-domain label positions are equal. Each pattern contributes a block-labeling factor and a count of stored output representatives it can reach. This is the partition-counting method explained below and proved in the appendix.',
  // Relocated from PartitionCountingExplainer.jsx BODY ¶3 (verbatim).
  'The partition-counting method groups assignments by equality pattern: which label positions carry the same value. With heterogeneous dimensions, this is typed — only labels from the same domain can be placed in the same equality block.',
  // Relocated from PartitionCountingExplainer.jsx BODY ¶4 (verbatim).
  String.raw`For a fixed typed equality pattern, the block-labeling factor counts how many product orbits live above that pattern. The map set $A_{\tilde{x}}$ records which input equality block supplies each output coordinate as we move through the product orbit. Quotienting $A_{\tilde{x}}$ by $H$ gives the number of stored output representatives reached.`,
];

const CLOSING_PARAGRAPH =
  // Relocated from PartitionCountingExplainer.jsx BODY ¶5 (verbatim).
  String.raw`This is not an approximation. When both are feasible, the typed partition count and corrected brute-force orbit enumeration must give the same $\alpha$.`;

const APPENDIX_POINTER =
  // Relocated from PartitionCountingExplainer.jsx footer (verbatim).
  String.raw`Theorem-level statement and proof live in the appendix. The notation registry encodes $\tilde{x}$, $A_{\tilde{x}}$, and $\overline{G}_{\tilde{x}}$ for use across the main page and modal.`;

const FORMULA = String.raw`\alpha_a = \sum_{\tilde{x}\in P_{\mathrm{typed}}(L_a)/G_a} \frac{\prod_s (n_s)_{b_s(\tilde{x})}}{|\overline{G}_{\tilde{x}}|}\, |A_{\tilde{x}}/H_a|`;

export default function TypedPartitionDemo({ componentData, costModel }) {
  const themeId = getActiveExplorerThemeId();
  const [selectedComponentIdx, setSelectedComponentIdx] = useState(0);
  const [selectedPatternKey, setSelectedPatternKey] = useState(null);

  if (!componentData || !costModel) return null;

  return (
    <section
      id="typed-partition-demo"
      className="mx-auto w-full max-w-[var(--prose-max)] rounded-xl border bg-white px-6 py-6 shadow-sm md:px-8 md:py-7 scroll-mt-24"
      style={{ borderColor: explorerThemeColor(themeId, 'border') }}
      aria-labelledby="typed-partition-demo-title"
    >
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        Typed partition counter
      </div>
      <h3
        id="typed-partition-demo-title"
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

      <div className="mt-6 overflow-x-auto">
        <Latex math={FORMULA} display />
      </div>

      <div className="mt-6 text-[12px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        pattern chips, breakdown panel, and cumulative table land in subsequent tasks.
      </div>

      <p className="mt-6 max-w-[78ch] font-serif text-[15px] leading-7" style={{ color: explorerThemeColor(themeId, 'body') }}>
        <InlineMathText>{CLOSING_PARAGRAPH}</InlineMathText>
      </p>
      <p className="mt-3 max-w-[78ch] text-[12.5px] leading-6" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        <InlineMathText>{APPENDIX_POINTER}</InlineMathText>
      </p>
    </section>
  );
}
