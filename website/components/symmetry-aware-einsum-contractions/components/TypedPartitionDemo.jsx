import { useState } from 'react';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor } from '../lib/notationSystem.js';
import { useShowMore } from '../lib/useShowMore.js';
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

const TITLE = 'Typed Partition Counter';
const DECK = 'Partition counting is the exact compressed counter behind the general branching case.';

const FORMULA = String.raw`\alpha_a = \sum_{\tilde{x}\in P_{\mathrm{typed}}(L_a)/G_a} \frac{\prod_s (n_s)_{b_s(\tilde{x})}}{|\overline{G}_{\tilde{x}}|}\, |A_{\tilde{x}}/H_a|`;

// Show-N constants. The chip strip and the cumulative table both cap their
// visible counts at this number; the user can click "+N more" to expand.
// Eight matches the visual rhythm of the surrounding subsection cards.
const VISIBLE_LIMIT = 8;

export default function TypedPartitionDemo({ componentData, costModel }) {
  const themeId = getActiveExplorerThemeId();
  const [selectedComponentIdx] = useState(0);
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

  // Captions only for the genuinely-informative cases. Closed-form regimes
  // (functionalProjection / trivial / allVisible / allSummed / singleton /
  // young) intentionally produce no caption — partition counting agrees with
  // the engine but the line adds noise, not information.
  let caption = null;
  if (componentRegimeId === 'partitionCount') {
    caption = "this component fires partitionCount in the live engine; the breakdown above matches the engine's α directly.";
  } else if (partitionBudgetExceeded) {
    caption = 'the partition budget is exceeded for this preset; the engine falls back to corrected brute-force enumeration.';
  } else if (componentRegimeId === 'bruteForceOrbit') {
    caption = 'this component fires bruteForceOrbit in the live engine; partition counting fits the budget here and gives the same α.';
  }

  const {
    visible: visibleChips,
    showAll: showAllChips,
    toggle: toggleChips,
    hidden: hiddenChipCount,
  } = useShowMore(chips, VISIBLE_LIMIT);
  const {
    visible: visibleRows,
    showAll: showAllRows,
    toggle: toggleRows,
    hidden: hiddenRowCount,
  } = useShowMore(cumulativeRows, VISIBLE_LIMIT);

  return (
    <div id="typed-partition-demo" className="bg-white p-4 scroll-mt-24">
      <ExplorerSubsectionHeader anchorId="typed-partition-demo" labelText="Typed partition counter">
        {TITLE}
      </ExplorerSubsectionHeader>
      <p className="explorer-support-prose mt-2">
        {DECK}
      </p>

      <div className="mt-3 overflow-x-auto">
        <Latex math={FORMULA} display />
      </div>

      <div className="mt-4">
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          Equality pattern
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {/* Pattern chips render as design-system .chip pills:
              rounded-full, 1px border, mono 11–12px, gray text on white.
              Active state uses the .chip--coral recipe — coral-light fill,
              coral-tinted border, coral-hover text. */}
          {visibleChips.map((chip) => {
            const isActive = chip.key === selectedPatternKey;
            return (
              <button
                type="button"
                key={chip.key}
                data-pattern-chip={chip.key}
                onClick={() => setSelectedPatternKey(chip.key)}
                className="rounded-full border px-2.5 py-1 font-mono text-[11px] font-medium transition-colors"
                style={{
                  background: isActive ? explorerThemeTint(themeId, 'hero', 0.08) : explorerThemeColor(themeId, 'surface'),
                  color: isActive ? explorerThemeColor(themeId, 'hero') : explorerThemeColor(themeId, 'muted'),
                  borderColor: isActive ? explorerThemeTint(themeId, 'hero', 0.25) : explorerThemeColor(themeId, 'border'),
                }}
              >
                {chip.key}
              </button>
            );
          })}
          {hiddenChipCount > 0 && (
            <button
              type="button"
              data-action="toggle-all-chips"
              onClick={toggleChips}
              className="rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors"
              style={{
                color: explorerThemeColor(themeId, 'muted'),
                background: 'transparent',
              }}
            >
              {showAllChips ? 'show fewer patterns' : `+${hiddenChipCount} more patterns`}
            </button>
          )}
          {chips.length === 0 && (
            <span className="text-[12px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
              no partition data available for this preset
            </span>
          )}
        </div>
      </div>

      {/* Breakdown panel for the active chip. No inner card chrome; the
          surfaceInset background alone separates it from the surrounding
          prose. Two columns share an eyebrow + values pattern matching
          the diptych in design-system/preview/components.html. */}
      {activeChip && (
        <div
          data-testid="partition-breakdown-panel"
          className="mt-3 rounded-md p-3"
          style={{ background: explorerThemeColor(themeId, 'surfaceInset') }}
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                Block structure · {activeChip.key}
              </div>
              <div className="mt-1 font-mono text-[11px] leading-6" style={{ color: explorerThemeColor(themeId, 'body') }}>
                blocks: {activeChip.blocks}<br/>
                labelings ∏ₛ (nₛ)_{`{bₛ}`} = <strong>{activeChip.labelings}</strong><br/>
                |Ḡ_x̃| = <strong>{activeChip.blockActionSize}</strong>
              </div>
            </div>
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                Projection reach
              </div>
              <div className="mt-1 font-mono text-[11px] leading-6" style={{ color: explorerThemeColor(themeId, 'body') }}>
                |A_x̃| = <strong>{inducedMaps?.size ?? 0}</strong><br/>
                |A_x̃/H| = <strong>{reachCount}</strong>
              </div>
            </div>
          </div>
          <div className="mt-2 border-t pt-2 font-mono text-[11px]" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
            contribution = ({activeChip.labelings} / {activeChip.blockActionSize}) · {reachCount} = <strong style={{ color: notationColor('alpha_total') }}>{contribution}</strong>
          </div>
        </div>
      )}

      <div data-testid="partition-cumulative-table" className="mt-3 border-t pt-3" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          Sum over patterns · live
        </div>
        <table className="mt-2 w-full font-mono text-[11px]" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: explorerThemeColor(themeId, 'surfaceInset') }}>
              <th className="p-1 text-left">pattern x̃</th>
              <th className="p-1 text-right">labelings/|Ḡ|</th>
              <th className="p-1 text-right">|A_x̃/H|</th>
              <th className="p-1 text-right">contrib.</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr
                key={row.key}
                onMouseEnter={() => setSelectedPatternKey(row.key)}
                style={{ background: row.key === selectedPatternKey ? explorerThemeTint(themeId, 'hero', 0.06) : 'transparent' }}
              >
                <td className="p-1">{row.key}</td>
                <td className="p-1 text-right">{row.labelings}/{row.blockActionSize}</td>
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
        {hiddenRowCount > 0 && (
          <button
            type="button"
            data-action="toggle-all-rows"
            onClick={toggleRows}
            className="mt-2 text-[11px] font-medium underline-offset-2 hover:underline"
            style={{ color: explorerThemeColor(themeId, 'muted') }}
          >
            {showAllRows ? 'show fewer rows' : `+${hiddenRowCount} more rows`}
          </button>
        )}
        {caption && (
          <p className="mt-2 text-[11px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
            {caption}
          </p>
        )}
      </div>
    </div>
  );
}
