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
  fallingFactorial,
} from '../engine/partition/typedPartitions.js';
import {
  restrictStabilizerToPositions,
} from '../engine/outputOrbit.js';

const TITLE = 'Typed Partition Counter';
const DECK = 'Partition counting is the exact compressed counter behind the general branching case.';

// V3.1 §36 Appendix-Linked Partition Formula Card
// The formula is composed from four hover-target sub-expressions so each term
// in α_a = Σ_p̃ (falling-factorial / |Ḡ_p̃|) · |A_p̃/H_a| can highlight the
// per-pattern table column it drives. KaTeX renders each sub-expression
// inline; the surrounding spans are the hover bus.
const FORMULA_LEAD = String.raw`\alpha_a = `;
const FORMULA_PATTERN = String.raw`\sum_{\tilde{x}\in P_{\mathrm{typed}}(L_a)/G_a}`;
const FORMULA_FALLING_FACTORIAL = String.raw`\prod_s (n_s)_{b_s(\tilde{x})}`;
const FORMULA_DIVISOR = String.raw`|\overline{G}_{\tilde{x}}|`;
const FORMULA_OUTPUT_REACH = String.raw`|A_{\tilde{x}}/H_a|`;

// Column tint applied when a formula term is hovered. var(--coral-light) is
// the design-system token for the coral 50-tint surface.
const FORMULA_HOVER_TINT = 'var(--coral-light)';

// V3.1 §36 hover targets. Each entry maps a formula sub-expression to the
// per-pattern table column it drives, plus the aria-label that names what
// hovering will highlight (so screen-reader users get the same cue).
const FORMULA_HOVER_TARGETS = {
  pattern: {
    column: 'pattern',
    ariaLabel: 'Hover to highlight the pattern column in the table below',
  },
  fallingFactorial: {
    column: 'fallingFactorial',
    ariaLabel: 'Hover to highlight the concrete labelings column in the table below',
  },
  divisor: {
    column: 'divisor',
    ariaLabel: 'Hover to highlight the block-symmetry divisor column in the table below',
  },
  outputReach: {
    column: 'outputReach',
    ariaLabel: 'Hover to highlight the output reach column in the table below',
  },
};

// Show-N constants. The chip strip and the cumulative table both cap their
// visible counts at this number; the user can click "+N more" to expand.
// Eight matches the visual rhythm of the surrounding subsection cards.
const VISIBLE_LIMIT = 8;

// V3.1 §34 brute-force compare. The "compare brute force" toggle expands a
// tiny tuple-count table for n = 2..6: at each n we substitute the
// partition's per-position sizes with uniform n and compute (n)_b — the
// number of tuples whose equality pattern is exactly x̃. This is the same
// quantity α counts before the |Ḡ| / |A_p̃/H_a| collapse, so the column
// gives a direct read on how much partition counting saves vs raw tuple
// enumeration.
const BRUTE_FORCE_NS = [2, 3, 4, 5, 6];

// V3.1 §35 mini-ledger. Cap how many concrete member assignments we list
// in the prose ledger so the panel never grows past a glance — the
// underlying labelings count is already shown numerically.
const MINI_LEDGER_EXAMPLE_LIMIT = 5;

// V3.1 §35 helper. Enumerate concrete value-assignments whose equality
// pattern matches the partition. For partition [0,0,1] with sizes [3,3,3]
// this yields tuples like (0,0,1), (0,0,2), (1,1,0), … — exactly the
// rows the falling-factorial product counts. Stops at `limit` so callers
// don't pay for the full ∏(nₛ)_{bₛ}; when truncated the caller can show
// "+N more" rather than the entire universe.
function enumerateMemberAssignments(partition, sizes, limit) {
  const blocks = numBlocks(partition);
  if (blocks === 0) return [];
  // Build the list of positions per block in canonical order.
  const positionsByBlock = Array.from({ length: blocks }, () => []);
  partition.forEach((block, position) => positionsByBlock[block].push(position));
  // Each block draws a value from sizes[firstPositionInBlock] (all
  // positions in a block share a domain). Block values must all be
  // distinct within a same-domain group; cross-domain blocks are free.
  const blockDomain = positionsByBlock.map((positions) => sizes[positions[0]]);
  const examples = [];
  function visit(blockIdx, blockValues) {
    if (examples.length >= limit) return;
    if (blockIdx === blocks) {
      const tuple = new Array(partition.length);
      partition.forEach((block, position) => {
        tuple[position] = blockValues[block];
      });
      examples.push(tuple);
      return;
    }
    const domainSize = blockDomain[blockIdx];
    for (let value = 0; value < domainSize; value += 1) {
      // Distinctness only required against earlier blocks sharing this domain.
      let conflict = false;
      for (let prior = 0; prior < blockIdx; prior += 1) {
        if (blockDomain[prior] === domainSize && blockValues[prior] === value) {
          conflict = true;
          break;
        }
      }
      if (conflict) continue;
      blockValues[blockIdx] = value;
      visit(blockIdx + 1, blockValues);
      if (examples.length >= limit) return;
    }
  }
  visit(0, []);
  return examples;
}

// V3.1 §34 helper. Given a partition (block assignment per position) and a
// uniform domain size n, return ∏_s (n)_{b_s}. We treat every position as
// living in the same n-size domain so the blocks all share a single
// falling-factorial tier; this is the natural "what if every dimension
// were n?" thought experiment for the brute-force compare table.
function bruteForceTupleCount(partition, n) {
  const blocks = numBlocks(partition);
  return fallingFactorial(n, blocks);
}

// V3.1 §34 helper. Render a partition as a human-readable block list, e.g.
// partition [0,0,1] with labels ['a','b','c'] becomes "{a,b}, {c}". Falls
// back to numeric position indices when labels are missing so the detail
// card never shows undefined.
function describeBlocks(partition, labels) {
  const blockToLabels = new Map();
  partition.forEach((block, position) => {
    if (!blockToLabels.has(block)) blockToLabels.set(block, []);
    blockToLabels.get(block).push(labels[position] ?? `p${position}`);
  });
  const sortedBlocks = [...blockToLabels.keys()].sort((a, b) => a - b);
  return sortedBlocks
    .map((block) => `{${blockToLabels.get(block).join(',')}}`)
    .join(', ');
}

// Render the same partition as a readable equality family. The raw canonical
// key (0|0|1) remains in data attributes; the visible UI uses labels like
// "i=j | k" so the reader sees the mathematical grouping directly.
function describePatternFamily(partition, labels) {
  const blockToLabels = new Map();
  partition.forEach((block, position) => {
    if (!blockToLabels.has(block)) blockToLabels.set(block, []);
    blockToLabels.get(block).push(labels[position] ?? `p${position}`);
  });
  const sortedBlocks = [...blockToLabels.keys()].sort((a, b) => a - b);
  return sortedBlocks
    .map((block) => blockToLabels.get(block).join('='))
    .join(' | ');
}

function describeMemberAssignment(tuple, partition, labels) {
  const blockToPositions = new Map();
  partition.forEach((block, position) => {
    if (!blockToPositions.has(block)) blockToPositions.set(block, []);
    blockToPositions.get(block).push(position);
  });
  const sortedBlocks = [...blockToPositions.keys()].sort((a, b) => a - b);
  return sortedBlocks
    .map((block) => {
      const positions = blockToPositions.get(block);
      const lhs = positions.map((position) => labels[position] ?? `p${position}`).join('=');
      return `${lhs}=${tuple[positions[0]]}`;
    })
    .join(' · ');
}

// V3.1 §36 helper. Wraps a formula sub-expression in a hover-bus span that
// sets the parent's formulaTermHover state on mouse OR focus events (so
// keyboard users get the same column-highlight cue as mouse users). The
// surrounding state lives in the parent — this component is purely the
// event surface and the visible affordance (cursor-help + faint underline).
function FormulaHoverSpan({ term, activeTerm, setTerm, children }) {
  const meta = FORMULA_HOVER_TARGETS[term];
  const isActive = activeTerm === term;
  const enter = () => setTerm(term);
  const leave = () => setTerm(null);
  return (
    <span
      data-formula-term={term}
      tabIndex={0}
      role="button"
      aria-label={meta.ariaLabel}
      title={meta.ariaLabel}
      className="cursor-help rounded px-0.5 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-coral/40"
      style={{ background: isActive ? FORMULA_HOVER_TINT : 'transparent' }}
      onMouseEnter={enter}
      onMouseLeave={leave}
      onFocus={enter}
      onBlur={leave}
    >
      {children}
    </span>
  );
}

export default function TypedPartitionDemo({ componentData, costModel }) {
  const themeId = getActiveExplorerThemeId();
  const [selectedComponentIdx] = useState(0);
  const [selectedPatternKey, setSelectedPatternKey] = useState(null);
  // V3.1 §36 hover state. One of: 'pattern' | 'fallingFactorial' | 'divisor'
  // | 'outputReach' | null. Set by mouse/focus on the four formula spans;
  // drives a coral-light tint on the matching per-pattern table column so
  // readers can match each summand to its corresponding column without
  // squinting at the LaTeX.
  const [formulaTermHover, setFormulaTermHover] = useState(null);
  // V3.1 §34 "Compare brute force" toggle. Default-off; flipping it
  // expands a tiny per-n tuple-count strip beneath the selected-pattern
  // detail card so readers can see how (n)_b grows vs the partition
  // counter's compressed form.
  const [bruteForceOpen, setBruteForceOpen] = useState(false);
  // V3.1 §35 "Show full α sum" toggle. Default-off; flipping it expands
  // the mini-ledger from the single selected-pattern contribution to the
  // full alpha sum across every pattern family in this component, so
  // readers can see how the one prose line they're reading composes into
  // the cumulative total at the bottom of the table.
  const [showFullAlphaSum, setShowFullAlphaSum] = useState(false);

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
    familyLabel: describePatternFamily(partition, activeComponent?.labels ?? []),
    partition,
    blocks: numBlocks(partition),
    labelings: typedLabelingCount(partition, sizes),
    blockActionSize: inducedBlockActionSize(partition, elements),
  }));

  const hElements = elements.length > 0 ? restrictStabilizerToPositions(elements, visiblePositions) : [];
  // V3.1 §34. selectedPatternIdx is derived from the canonical
  // selectedPatternKey state above so chip clicks, row hovers, and the
  // detail card stay in lockstep. -1 means "no row selected yet"; the
  // detail card falls back to the first chip in that case so the panel is
  // never empty when chips exist.
  const selectedPatternIdx = chips.findIndex((chip) => chip.key === selectedPatternKey);
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

  // V3.1 §35 mini-ledger derived data. The ledger renders prose around
  // four numbers: the per-block falling-factorial multiplier (e.g.
  // "choose value for ab block: 3 choices"), the total concrete-labeling
  // count, up to MINI_LEDGER_EXAMPLE_LIMIT member assignments shown as
  // tuples, and the contribution this pattern family contributes to α.
  const activeLabels = activeComponent?.labels ?? [];
  const ledgerBlockChoices = activeChip
    ? (() => {
        const blocks = numBlocks(activeChip.partition);
        const positionsByBlock = Array.from({ length: blocks }, () => []);
        activeChip.partition.forEach((block, position) => {
          positionsByBlock[block].push(position);
        });
        // Within each domain, blocks are picked in order and must be
        // distinct from earlier blocks of the same domain. The choice
        // count for the k-th block of a size-n domain is therefore
        // (n - k_so_far_in_this_domain).
        const usedByDomain = new Map();
        return positionsByBlock.map((positions) => {
          const domain = sizes[positions[0]];
          const used = usedByDomain.get(domain) ?? 0;
          usedByDomain.set(domain, used + 1);
          return {
            blockLabel: positions.map((p) => activeLabels[p] ?? `p${p}`).join(''),
            domain,
            choices: domain - used,
          };
        });
      })()
    : [];
  const ledgerExamples = activeChip
    ? enumerateMemberAssignments(activeChip.partition, sizes, MINI_LEDGER_EXAMPLE_LIMIT)
    : [];

  // V3.1 §34 brute-force compare. For each n in BRUTE_FORCE_NS substitute
  // every position size with n and compute (n)_b for the active partition;
  // alongside, n^k is the raw "enumerate every tuple" count. The ratio
  // n^k / (n)_b shows the saving the equality pattern alone gives, before
  // |Ḡ| / |A_p̃/H_a| collapse the orbit further.
  const bruteForceRows = activeChip
    ? BRUTE_FORCE_NS.map((n) => {
        const tupleCount = bruteForceTupleCount(activeChip.partition, n);
        const positions = activeChip.partition.length;
        const rawCount = positions > 0 ? Math.pow(n, positions) : 0;
        return { n, tupleCount, rawCount };
      })
    : [];

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
    <div id="typed-partition-demo" className="bg-white p-4 scroll-mt-sticky">
      <ExplorerSubsectionHeader anchorId="typed-partition-demo" labelText="Typed partition counter">
        {TITLE}
      </ExplorerSubsectionHeader>
      <p className="explorer-support-prose mt-2">
        {DECK}
      </p>

      <div
        data-testid="partition-workbench"
        className="mt-3 grid gap-3 min-[1180px]:grid-cols-[minmax(0,1fr)_minmax(320px,0.66fr)]"
      >
        <div
          data-testid="partition-formula-panel"
          className="min-w-0 rounded-lg border bg-white p-3"
          style={{ borderColor: explorerThemeColor(themeId, 'border') }}
        >
          {/* V3.1 §36 Appendix-Linked Partition Formula Card.
              The formula is composed from four hover-target sub-expressions.
              Each span sets `formulaTermHover` on enter/leave/focus/blur and
              carries an aria-label describing what hovering will highlight.
              The matching column in the per-pattern table tints coral-light. */}
          <div className="overflow-x-auto">
            <div className="flex min-w-max items-center justify-center gap-x-1 gap-y-2 text-[18px]">
              <Latex math={FORMULA_LEAD} />
              <FormulaHoverSpan
                term="pattern"
                activeTerm={formulaTermHover}
                setTerm={setFormulaTermHover}
              >
                <Latex math={FORMULA_PATTERN} />
              </FormulaHoverSpan>
              <span className="inline-flex flex-col items-center px-1">
                <FormulaHoverSpan
                  term="fallingFactorial"
                  activeTerm={formulaTermHover}
                  setTerm={setFormulaTermHover}
                >
                  <Latex math={FORMULA_FALLING_FACTORIAL} />
                </FormulaHoverSpan>
                <span
                  aria-hidden="true"
                  className="my-0.5 block w-full"
                  style={{
                    borderTop: `1px solid ${explorerThemeColor(themeId, 'body')}`,
                    minWidth: '2.5rem',
                  }}
                />
                <FormulaHoverSpan
                  term="divisor"
                  activeTerm={formulaTermHover}
                  setTerm={setFormulaTermHover}
                >
                  <Latex math={FORMULA_DIVISOR} />
                </FormulaHoverSpan>
              </span>
              <span aria-hidden="true" className="px-0.5">·</span>
              <FormulaHoverSpan
                term="outputReach"
                activeTerm={formulaTermHover}
                setTerm={setFormulaTermHover}
              >
                <Latex math={FORMULA_OUTPUT_REACH} />
              </FormulaHoverSpan>
            </div>
          </div>
          <div className="mt-2 text-center text-[11px]">
            <a
              href="#appendix-section-6"
              data-formula-link="appendix-c"
              aria-label="Read Appendix C — Typed partition counting theorem"
              className="font-semibold text-coral underline decoration-coral/30 underline-offset-4 transition-colors hover:decoration-coral"
            >
              Full statement → Appendix C
            </a>
          </div>

          <div className="mt-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
              Equality groups
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {/* Pattern chips render as design-system .chip pills:
                  rounded-full, 1px border, mono 11–12px, gray text on white.
                  Active state uses the .chip--coral recipe — coral-light fill,
                  coral-tinted border, coral-hover text. */}
              {visibleChips.map((chip) => {
                const isActive = chip.key === activeChip?.key;
                return (
                  <button
                    type="button"
                    key={chip.key}
                    data-pattern-chip={chip.key}
                    onClick={() => setSelectedPatternKey(chip.key)}
                    className="rounded-full border px-2.5 py-1 font-mono text-[11px] font-medium transition-colors"
                    title={`Raw equality key: ${chip.key}`}
                    aria-label={`Select equality group ${chip.familyLabel}; raw key ${chip.key}`}
                    style={{
                      background: isActive ? explorerThemeTint(themeId, 'hero', 0.08) : explorerThemeColor(themeId, 'surface'),
                      color: isActive ? explorerThemeColor(themeId, 'hero') : explorerThemeColor(themeId, 'muted'),
                      borderColor: isActive ? explorerThemeTint(themeId, 'hero', 0.25) : explorerThemeColor(themeId, 'border'),
                    }}
                  >
                    {chip.familyLabel}
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
        </div>

        <div
          data-testid="partition-cumulative-table"
          className="min-w-0 self-start rounded-lg border p-3"
          style={{
            background: explorerThemeColor(themeId, 'surface'),
            borderColor: explorerThemeColor(themeId, 'border'),
          }}
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
              Sum over patterns · live
            </div>
            <div className="font-mono text-[11px] font-semibold" style={{ color: explorerThemeColor(themeId, 'body') }}>
              α = <strong data-testid="partition-alpha-total" style={{ color: notationColor('alpha_total') }}>{cumulativeAlpha}</strong>
            </div>
          </div>
          {/* V3.1 §36 per-pattern table.
              Each column carries a `data-pattern-column` matching the four
              FORMULA hover targets (pattern, fallingFactorial, divisor,
              outputReach). When `formulaTermHover` matches the column, the
              cells tint coral-light so readers can locate the column the
              hovered formula term contributes to.

              V3.1 §34 reverse direction. The four <th> headers are themselves
              keyboard-focusable hover surfaces — hovering or focusing a
              column header sets formulaTermHover, which highlights the
              matching term in the formula card above. C36 wires
              formula → table; this is the reverse arrow (table → formula). */}
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[280px] font-mono text-[11px]" style={{ borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: explorerThemeColor(themeId, 'surfaceInset') }}>
                  <th
                    className="p-1 text-left cursor-help"
                    data-pattern-column="pattern"
                    tabIndex={0}
                    aria-label="Hover or focus to highlight the pattern term in the formula above"
                    onMouseEnter={() => setFormulaTermHover('pattern')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('pattern')}
                    onBlur={() => setFormulaTermHover(null)}
                    style={{ background: formulaTermHover === 'pattern' ? FORMULA_HOVER_TINT : undefined }}
                  >
                    family
                  </th>
                  <th
                    className="p-1 text-right cursor-help"
                    data-pattern-column="fallingFactorial"
                    tabIndex={0}
                    aria-label="Hover or focus to highlight the concrete labelings term in the formula above"
                    onMouseEnter={() => setFormulaTermHover('fallingFactorial')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('fallingFactorial')}
                    onBlur={() => setFormulaTermHover(null)}
                    style={{ background: formulaTermHover === 'fallingFactorial' ? FORMULA_HOVER_TINT : undefined }}
                  >
                    labelings
                  </th>
                  <th
                    className="p-1 text-right cursor-help"
                    data-pattern-column="divisor"
                    tabIndex={0}
                    aria-label="Hover or focus to highlight the block-symmetry divisor term in the formula above"
                    onMouseEnter={() => setFormulaTermHover('divisor')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('divisor')}
                    onBlur={() => setFormulaTermHover(null)}
                    style={{ background: formulaTermHover === 'divisor' ? FORMULA_HOVER_TINT : undefined }}
                  >
                    |Ḡ|
                  </th>
                  <th
                    className="p-1 text-right cursor-help"
                    data-pattern-column="outputReach"
                    tabIndex={0}
                    aria-label="Hover or focus to highlight the output reach term in the formula above"
                    onMouseEnter={() => setFormulaTermHover('outputReach')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('outputReach')}
                    onBlur={() => setFormulaTermHover(null)}
                    style={{ background: formulaTermHover === 'outputReach' ? FORMULA_HOVER_TINT : undefined }}
                  >
                    |A_x̃/H|
                  </th>
                  <th className="p-1 text-right">contrib.</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((row) => {
                  const rowSelected = row.key === activeChip?.key;
                  const rowBackground = rowSelected
                    ? explorerThemeTint(themeId, 'hero', 0.06)
                    : 'transparent';
                  const tintFor = (column) =>
                    formulaTermHover === column ? FORMULA_HOVER_TINT : rowBackground;
                  return (
                    <tr
                      key={row.key}
                      onMouseEnter={() => setSelectedPatternKey(row.key)}
                    >
                      <td
                        className="p-1"
                        data-pattern-column="pattern"
                        title={`Raw equality key: ${row.key}`}
                        style={{ background: tintFor('pattern') }}
                      >
                        {row.familyLabel}
                      </td>
                      <td
                        className="p-1 text-right"
                        data-pattern-column="fallingFactorial"
                        style={{ background: tintFor('fallingFactorial') }}
                      >
                        {row.labelings}
                      </td>
                      <td
                        className="p-1 text-right"
                        data-pattern-column="divisor"
                        style={{ background: tintFor('divisor') }}
                      >
                        {row.blockActionSize}
                      </td>
                      <td
                        className="p-1 text-right"
                        data-pattern-column="outputReach"
                        style={{ background: tintFor('outputReach') }}
                      >
                        {row.reach}
                      </td>
                      <td
                        className="p-1 text-right"
                        style={{ background: rowBackground }}
                      >
                        {row.contribution}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr style={{ borderTop: `1.5px solid ${explorerThemeColor(themeId, 'ink')}` }}>
                  <td className="p-1 font-semibold" colSpan={4}>α =</td>
                  <td className="p-1 text-right font-semibold" style={{ color: notationColor('alpha_total') }}>{cumulativeAlpha}</td>
                </tr>
              </tfoot>
            </table>
          </div>
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
            <p className="mt-2 text-[11px] leading-5" style={{ color: explorerThemeColor(themeId, 'muted') }}>
              {caption}
            </p>
          )}
        </div>
      </div>

      <div
        data-testid="partition-detail-grid"
        className="mt-3 grid gap-3 min-[1180px]:grid-cols-2"
      >
        {activeChip && (
          <>
            {/* Breakdown + selected-pattern detail share one surface so the
                reader can keep the current pattern, formula terms, and
                contribution arithmetic in view together. */}
            <div
              data-testid="partition-breakdown-panel"
              className="rounded-lg border p-3"
              style={{
                background: explorerThemeColor(themeId, 'surfaceInset'),
                borderColor: explorerThemeColor(themeId, 'border'),
              }}
            >
              <div className="grid gap-3 sm:grid-cols-3">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                    Block structure
                  </div>
                  <div className="mt-1 font-mono text-[11px] leading-5" style={{ color: explorerThemeColor(themeId, 'body') }}>
                    groups: <strong>{activeChip.familyLabel}</strong><br/>
                    blocks: <strong>{activeChip.blocks}</strong><br/>
                    labelings: <strong>{activeChip.labelings}</strong><br/>
                    |Ḡ_x̃|: <strong>{activeChip.blockActionSize}</strong>
                  </div>
                </div>
                <div className="sm:border-l sm:pl-3" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                    Projection reach
                  </div>
                  <div className="mt-1 font-mono text-[11px] leading-5" style={{ color: explorerThemeColor(themeId, 'body') }}>
                    |A_x̃|: <strong>{inducedMaps?.size ?? 0}</strong><br/>
                    |A_x̃/H|: <strong>{reachCount}</strong>
                  </div>
                </div>
                <div className="sm:border-l sm:pl-3" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                    Contribution
                  </div>
                  <div className="mt-1 font-mono text-[11px] leading-5" style={{ color: explorerThemeColor(themeId, 'body') }}>
                    {activeChip.labelings} / {activeChip.blockActionSize}<br/>
                    · {reachCount} = <strong style={{ color: notationColor('alpha_total') }}>{contribution}</strong>
                  </div>
                </div>
              </div>

              {/* V3.1 §34 Pattern Contribution Explainer.
                  Six labeled fields lay out exactly what α gets from this one
                  partition, in the spec's canonical order. This compact grid
                  preserves the hover bus while avoiding a second full card. */}
              <div
                data-testid="pattern-contribution-explainer"
                className="mt-3 border-t pt-3"
                style={{ borderColor: explorerThemeColor(themeId, 'border') }}
              >
                <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                  Selected pattern detail · idx {selectedPatternIdx >= 0 ? selectedPatternIdx : 0}
                </div>
                <dl className="mt-2 grid gap-x-3 gap-y-1 font-mono text-[11px] leading-5 sm:grid-cols-2" style={{ color: explorerThemeColor(themeId, 'body') }}>
                  <div
                    data-detail-field="pattern"
                    tabIndex={0}
                    aria-label="Pattern field — hover or focus to highlight the pattern term in the formula above"
                    className="cursor-help rounded px-1 py-0.5 transition-colors"
                    style={{ background: formulaTermHover === 'pattern' ? FORMULA_HOVER_TINT : 'transparent' }}
                    onMouseEnter={() => setFormulaTermHover('pattern')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('pattern')}
                    onBlur={() => setFormulaTermHover(null)}
                  >
                    <dt className="inline font-semibold">Pattern:</dt> <dd className="inline">{activeChip.familyLabel}</dd>
                  </div>
                  <div
                    data-detail-field="blocks"
                    tabIndex={0}
                    aria-label="Blocks field — hover or focus to highlight the pattern term in the formula above"
                    className="cursor-help rounded px-1 py-0.5 transition-colors"
                    style={{ background: formulaTermHover === 'pattern' ? FORMULA_HOVER_TINT : 'transparent' }}
                    onMouseEnter={() => setFormulaTermHover('pattern')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('pattern')}
                    onBlur={() => setFormulaTermHover(null)}
                  >
                    <dt className="inline font-semibold">Blocks:</dt>{' '}
                    <dd className="inline">{describeBlocks(activeChip.partition, activeComponent?.labels ?? [])}</dd>
                  </div>
                  <div
                    data-detail-field="fallingFactorial"
                    tabIndex={0}
                    aria-label="Concrete labelings field — hover or focus to highlight the concrete labelings term in the formula above"
                    className="cursor-help rounded px-1 py-0.5 transition-colors"
                    style={{ background: formulaTermHover === 'fallingFactorial' ? FORMULA_HOVER_TINT : 'transparent' }}
                    onMouseEnter={() => setFormulaTermHover('fallingFactorial')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('fallingFactorial')}
                    onBlur={() => setFormulaTermHover(null)}
                  >
                    <dt className="inline font-semibold">Concrete labelings:</dt>{' '}
                    <dd className="inline">∏ₛ (nₛ)_{`{bₛ}`} = <strong>{activeChip.labelings}</strong></dd>
                  </div>
                  <div
                    data-detail-field="divisor"
                    tabIndex={0}
                    aria-label="Induced block-symmetry divisor field — hover or focus to highlight the divisor term in the formula above"
                    className="cursor-help rounded px-1 py-0.5 transition-colors"
                    style={{ background: formulaTermHover === 'divisor' ? FORMULA_HOVER_TINT : 'transparent' }}
                    onMouseEnter={() => setFormulaTermHover('divisor')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('divisor')}
                    onBlur={() => setFormulaTermHover(null)}
                  >
                    <dt className="inline font-semibold">Induced block-symmetry divisor:</dt>{' '}
                    <dd className="inline">|Ḡ_x̃| = <strong>{activeChip.blockActionSize}</strong></dd>
                  </div>
                  <div
                    data-detail-field="outputReach"
                    tabIndex={0}
                    aria-label="Output reach field — hover or focus to highlight the output reach term in the formula above"
                    className="cursor-help rounded px-1 py-0.5 transition-colors"
                    style={{ background: formulaTermHover === 'outputReach' ? FORMULA_HOVER_TINT : 'transparent' }}
                    onMouseEnter={() => setFormulaTermHover('outputReach')}
                    onMouseLeave={() => setFormulaTermHover(null)}
                    onFocus={() => setFormulaTermHover('outputReach')}
                    onBlur={() => setFormulaTermHover(null)}
                  >
                    <dt className="inline font-semibold">Output reach:</dt>{' '}
                    <dd className="inline">|A_x̃ / H_a| = <strong>{reachCount}</strong></dd>
                  </div>
                  <div
                    data-detail-field="contribution"
                    tabIndex={0}
                    aria-label="Contribution field — sum of labelings divided by divisor times reach"
                    className="cursor-help rounded px-1 py-0.5 transition-colors"
                  >
                    <dt className="inline font-semibold">Contribution:</dt>{' '}
                    <dd className="inline">{activeChip.labelings} / {activeChip.blockActionSize} · {reachCount} = <strong style={{ color: notationColor('alpha_total') }}>{contribution}</strong></dd>
                  </div>
                </dl>
                <div className="mt-2 border-t pt-2" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
                  <button
                    type="button"
                    data-action="toggle-brute-force"
                    aria-label={bruteForceOpen ? 'Hide the brute-force tuple comparison table' : 'Compare brute force — show tuple count for n = 2 through 6'}
                    aria-expanded={bruteForceOpen}
                    onClick={() => setBruteForceOpen((prev) => !prev)}
                    className="py-1 text-[11px] font-medium underline-offset-2 hover:underline"
                    style={{ color: explorerThemeColor(themeId, 'muted') }}
                  >
                    {bruteForceOpen ? 'hide brute-force compare' : 'compare brute force'}
                  </button>
                  {bruteForceOpen && (
                    <div data-testid="brute-force-compare" className="mt-2">
                      <div className="text-[10px] leading-5" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                        Equivalent tuple enumeration · uniform n substitution. (n)_b is the count of tuples that exhibit this exact equality pattern; n^k is the raw enumeration cost over all positions.
                      </div>
                      <table className="mt-2 w-full font-mono text-[11px]" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ background: explorerThemeColor(themeId, 'surface') }}>
                            <th className="p-1 text-left">n</th>
                            <th className="p-1 text-right">tuples in pattern (n)_b</th>
                            <th className="p-1 text-right">raw enum n^k</th>
                          </tr>
                        </thead>
                        <tbody>
                          {bruteForceRows.map((row) => (
                            <tr key={`bf-n-${row.n}`} data-brute-force-row={row.n}>
                              <td className="p-1">n = {row.n}</td>
                              <td className="p-1 text-right">{row.tupleCount}</td>
                              <td className="p-1 text-right">{row.rawCount}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* V3.1 §35 Worked Pattern Mini-Ledger. The prose ledger is now
                a compact companion to the detail grid: block choices and
                tuple examples use chip-like inline rows so the total stays
                visible beside the live sum table on desktop. */}
            <div
              data-testid="worked-pattern-mini-ledger"
              className="rounded-lg border p-3"
              style={{
                background: explorerThemeColor(themeId, 'surfaceInset'),
                borderColor: explorerThemeColor(themeId, 'border'),
              }}
            >
              <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                Worked mini-ledger · {activeChip.familyLabel}
              </div>
              <div className="mt-2 grid gap-3 font-mono text-[11px] leading-5 md:grid-cols-[minmax(0,1fr)_minmax(190px,0.65fr)]" style={{ color: explorerThemeColor(themeId, 'body') }}>
                <div className="min-w-0">
                  <div>
                    <span className="font-semibold">Pattern:</span> {activeChip.familyLabel}
                  </div>
                  <div>
                    <span className="font-semibold">Meaning:</span>{' '}
                    {describeBlocks(activeChip.partition, activeLabels)} — same-block positions share a value; different blocks may differ.
                  </div>
                  <div className="mt-2">
                    <span className="font-semibold">Concrete labelings:</span>
                  </div>
                  <ul className="mt-1 flex flex-wrap gap-1.5">
                    {ledgerBlockChoices.map((entry, idx) => (
                      <li
                        key={`mini-ledger-block-${idx}`}
                        data-mini-ledger-block={idx}
                        className="rounded-full border px-2 py-0.5"
                        style={{
                          background: explorerThemeColor(themeId, 'surface'),
                          borderColor: explorerThemeColor(themeId, 'border'),
                        }}
                      >
                        {entry.blockLabel}: <strong>{entry.choices}</strong> choices
                      </li>
                    ))}
                    <li
                      className="rounded-full border px-2 py-0.5 font-semibold"
                      style={{
                        background: 'var(--coral-light)',
                        borderColor: explorerThemeTint(themeId, 'hero', 0.25),
                        color: explorerThemeColor(themeId, 'hero'),
                      }}
                    >
                      total: {activeChip.labelings}
                    </li>
                  </ul>
                </div>
                <div className="min-w-0">
                  <div>
                    <span className="font-semibold">Member assignments:</span>
                  </div>
                  <ul className="mt-1 flex flex-wrap gap-1.5">
                    {ledgerExamples.map((tuple, idx) => (
                      <li
                        key={`mini-ledger-example-${idx}`}
                        data-mini-ledger-example={idx}
                        className="rounded-full border px-2 py-0.5"
                        style={{
                          background: explorerThemeColor(themeId, 'surface'),
                          borderColor: explorerThemeColor(themeId, 'border'),
                        }}
                      >
                        {describeMemberAssignment(tuple, activeChip.partition, activeLabels)}
                      </li>
                    ))}
                    {activeChip.labelings > ledgerExamples.length && (
                      <li
                        className="rounded-full px-2 py-0.5"
                        style={{ color: explorerThemeColor(themeId, 'muted') }}
                      >
                        +{activeChip.labelings - ledgerExamples.length} more
                      </li>
                    )}
                  </ul>
                  <div className="mt-2">
                    <span className="font-semibold">Output reach:</span> <strong>{reachCount}</strong>
                  </div>
                  <div>
                    <span className="font-semibold">Contribution shown for this pattern family:</span>{' '}
                    <strong style={{ color: notationColor('alpha_total') }}>{contribution}</strong>
                  </div>
                </div>
              </div>
              {/* Required caveat. The coral rail follows the Flopscope active
                  state recipe and keeps this note inside the single-accent
                  design-system vocabulary. */}
              <div
                data-testid="worked-pattern-ledger-caveat"
                role="note"
                className="mt-2 border-l-4 px-3 py-2 text-[11px] leading-5"
                style={{
                  background: 'var(--coral-light)',
                  borderColor: 'var(--coral)',
                  color: explorerThemeColor(themeId, 'body'),
                }}
              >
                This is one pattern-family contribution, not the full alpha; the full alpha also includes the all-distinct and all-equal pattern families.
              </div>
              <div className="mt-2 border-t pt-2" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>
                <button
                  type="button"
                  data-action="toggle-full-alpha-sum"
                  aria-label={showFullAlphaSum ? 'Hide the full alpha sum across every pattern family' : 'Show full α sum — stack every pattern-family contribution into the cumulative alpha'}
                  aria-expanded={showFullAlphaSum}
                  onClick={() => setShowFullAlphaSum((prev) => !prev)}
                  className="py-1 text-[11px] font-medium underline-offset-2 hover:underline"
                  style={{ color: explorerThemeColor(themeId, 'muted') }}
                >
                  {showFullAlphaSum ? 'hide full α sum' : 'show full α sum'}
                </button>
                {showFullAlphaSum && (
                  <div data-testid="worked-pattern-full-alpha-sum" className="mt-2 font-mono text-[11px] leading-5" style={{ color: explorerThemeColor(themeId, 'body') }}>
                    <div className="text-[10px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                      Stacked α: every pattern family contributes one term.
                    </div>
                    <div className="mt-1">
                      α_a ={' '}
                      {cumulativeRows.map((row, idx) => (
                        <span key={`full-alpha-term-${row.key}`} data-full-alpha-term={row.key}>
                          {idx > 0 ? ' + ' : ''}
                          <span title={`${row.familyLabel}: ${row.labelings}/${row.blockActionSize} · ${row.reach} = ${row.contribution}`}>
                            {row.contribution}
                            <span className="text-[10px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
                              {' '}({row.familyLabel})
                            </span>
                          </span>
                        </span>
                      ))}
                      {' = '}
                      <strong style={{ color: notationColor('alpha_total') }}>{cumulativeAlpha}</strong>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
