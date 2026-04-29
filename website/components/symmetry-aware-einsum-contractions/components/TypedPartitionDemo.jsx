import { useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor } from '../lib/notationSystem.js';
import {
  generateTypedSetPartitions,
  partitionOrbitReps,
  typedLabelingCount,
  inducedBlockActionSize,
  inducedPrefixMaps,
  countMapOrbitsUnderH,
  partitionKey,
  numBlocks,
} from '../engine/partition/typedPartitions.js';
import {
  restrictStabilizerToPositions,
} from '../engine/outputOrbit.js';

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

  const components = componentData.components ?? [];
  const activeComponent = components[selectedComponentIdx] ?? null;
  const sizes = activeComponent?.sizes ?? [];
  const elements = activeComponent?.elements ?? [];
  const visiblePositions = activeComponent
    ? activeComponent.va.map((label) => activeComponent.labels.indexOf(label))
    : [];

  const allPartitions = sizes.length > 0 ? generateTypedSetPartitions(sizes) : [];
  const orbitReps = elements.length > 0 ? partitionOrbitReps(allPartitions, elements) : allPartitions;

  const chips = orbitReps.map((partition) => ({
    key: partitionKey(partition),
    partition,
    blocks: numBlocks(partition),
    labelings: typedLabelingCount(partition, sizes),
    blockActionSize: inducedBlockActionSize(partition, elements),
  }));

  const hElements = elements.length > 0 ? restrictStabilizerToPositions(elements, visiblePositions) : [];
  const activeChip = chips.find((chip) => chip.key === selectedPatternKey) ?? chips[0] ?? null;
  const inducedMaps = activeChip
    ? inducedPrefixMaps(activeChip.partition, elements, visiblePositions)
    : new Set();
  const reachCount = activeChip
    ? countMapOrbitsUnderH(inducedMaps, hElements)
    : 0;
  const contribution = activeChip
    ? Math.round((activeChip.labelings / Math.max(activeChip.blockActionSize, 1)) * reachCount)
    : 0;

  const cumulativeRows = chips.map((chip) => {
    const maps = inducedPrefixMaps(chip.partition, elements, visiblePositions);
    const reach = countMapOrbitsUnderH(maps, hElements);
    const labelOver = chip.blockActionSize > 0 ? chip.labelings / chip.blockActionSize : 0;
    return {
      ...chip,
      reach,
      contribution: Math.round(labelOver * reach),
    };
  });
  const cumulativeAlpha = cumulativeRows.reduce((acc, row) => acc + row.contribution, 0);

  const componentRegimeId = activeComponent?.accumulation?.regimeId ?? null;
  const partitionBudgetExceeded =
    componentRegimeId === 'bruteForceOrbit' &&
    (activeComponent?.accumulation?.trace ?? []).some(
      (step) => step.regimeId === 'partitionCount' && step.decision === 'refused'
    );

  let caption;
  if (componentRegimeId === 'partitionCount') {
    caption = "this component fires partitionCount in the live engine; the breakdown above matches the engine's α directly.";
  } else if (componentRegimeId && componentRegimeId !== 'bruteForceOrbit') {
    caption = `this component fires ${componentRegimeId} in the live engine; partition counting would give the same α this way.`;
  } else if (partitionBudgetExceeded) {
    caption = 'the partition budget is exceeded for this preset; the engine falls back to corrected brute-force enumeration.';
  } else {
    caption = 'this component fires bruteForceOrbit in the live engine; partition counting fits the budget here and gives the same α.';
  }

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

      <div className="mt-5">
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          Equality pattern
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {chips.map((chip) => {
            const isActive = chip.key === selectedPatternKey;
            return (
              <button
                type="button"
                key={chip.key}
                data-pattern-chip={chip.key}
                onClick={() => setSelectedPatternKey(chip.key)}
                className="rounded px-2.5 py-1 font-mono text-[11px] font-semibold transition-colors"
                style={{
                  background: isActive ? explorerThemeTint(themeId, 'hero', 0.12) : 'transparent',
                  color: isActive ? explorerThemeColor(themeId, 'hero') : explorerThemeColor(themeId, 'body'),
                  border: `1px solid ${isActive ? explorerThemeColor(themeId, 'hero') : explorerThemeColor(themeId, 'border')}`,
                }}
              >
                {chip.key}
              </button>
            );
          })}
          {chips.length === 0 && (
            <span className="text-[12px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
              no partition data available for this preset
            </span>
          )}
        </div>
      </div>

      {activeChip && (
        <div
          data-testid="partition-breakdown-panel"
          className="mt-4 rounded-md border p-4"
          style={{ borderColor: explorerThemeColor(themeId, 'border'), background: explorerThemeColor(themeId, 'surfaceInset') }}
        >
          <div className="grid gap-5 lg:grid-cols-2">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                Block structure · {activeChip.key}
              </div>
              <div className="mt-2 font-mono text-[12px] leading-7" style={{ color: explorerThemeColor(themeId, 'body') }}>
                blocks: {activeChip.blocks}<br/>
                labelings ∏ₛ (nₛ)_{`{bₛ}`} = <strong>{activeChip.labelings}</strong><br/>
                |Ḡ_x̃| (block-action image) = <strong>{activeChip.blockActionSize}</strong>
              </div>
            </div>
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                Projection reach · |A_x̃ / H|
              </div>
              <div className="mt-2 font-mono text-[12px] leading-7" style={{ color: explorerThemeColor(themeId, 'body') }}>
                induced maps |A_x̃| = <strong>{inducedMaps?.size ?? 0}</strong><br/>
                quotient by H gives <strong>|A_x̃/H| = {reachCount}</strong>
              </div>
            </div>
          </div>
          <div className="mt-3 border-t pt-3 font-mono text-[12px]" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
            contribution = (∏ (nₛ)_{`{bₛ}`} / |Ḡ_x̃|) · |A_x̃/H| = ({activeChip.labelings} / {activeChip.blockActionSize}) · {reachCount} = <strong style={{ color: notationColor('alpha_total') }}>{contribution}</strong>
          </div>
        </div>
      )}

      <div data-testid="partition-cumulative-table" className="mt-4 border-t pt-3" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          Sum over patterns · live
        </div>
        <table className="mt-2 w-full font-mono text-[12px]" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: explorerThemeColor(themeId, 'surfaceInset') }}>
              <th className="p-1 text-left">pattern x̃</th>
              <th className="p-1 text-right">labelings / |Ḡ|</th>
              <th className="p-1 text-right">|A_x̃/H|</th>
              <th className="p-1 text-right">contribution</th>
            </tr>
          </thead>
          <tbody>
            {cumulativeRows.map((row) => (
              <tr
                key={row.key}
                onMouseEnter={() => setSelectedPatternKey(row.key)}
                style={{ background: row.key === selectedPatternKey ? explorerThemeTint(themeId, 'hero', 0.06) : 'transparent' }}
              >
                <td className="p-1">{row.key}</td>
                <td className="p-1 text-right">{row.labelings} / {row.blockActionSize}</td>
                <td className="p-1 text-right">{row.reach}</td>
                <td className="p-1 text-right">{row.contribution}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: `1.5px solid ${explorerThemeColor(themeId, 'ink')}` }}>
              <td className="p-1 font-semibold" colSpan={3}>α =</td>
              <td className="p-1 text-right font-semibold" data-testid="partition-alpha-total" style={{ color: notationColor('alpha_total') }}>{cumulativeAlpha}</td>
            </tr>
          </tfoot>
        </table>
        <p className="mt-3 text-[12px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          {caption}
        </p>
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
