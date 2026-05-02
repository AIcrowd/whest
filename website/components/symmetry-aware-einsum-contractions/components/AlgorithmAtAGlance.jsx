import { useCallback, useMemo, useState } from 'react';
import { cn } from '@/lib/utils';
import ExplorerSectionCard, { AnchorLink } from './ExplorerSectionCard.jsx';
import EditorialCallout from './EditorialCallout.jsx';
import Latex from './Latex.jsx';
import MentalFrameworkCode from './MentalFrameworkCode.jsx';
import InlineMathText from './InlineMathText.jsx';
import SectionReferenceLink from './SectionReferenceLink.jsx';
import { FormulaHighlighted } from './StickyBar.jsx';
import { default as renderProseBlocks } from '../content/renderProseBlocks.jsx';
import mainPreamble from '../content/main/preamble.js';
import { getActiveExplorerThemeId, explorerThemeColor } from '../lib/explorerTheme.js';
import { notationLatex } from '../lib/notationSystem.js';
import { buildSection1ExampleView } from '../lib/section1ExampleView.js';

/**
 * Distill-style preamble for the symmetry-aware einsum explorer.
 *
 * Two-column top section:
 *   LEFT  — Einsum Contraction: prose + a 4-operand matrix-chain example
 *           rendered via color-coded KaTeX with explicit coral/slate label
 *           cues, followed by a "Where symmetry enters" transition block
 *           that names the pointwise group, product orbits, and output updates.
 *   RIGHT — The Mental Framework: two step panels (Multiplication · μ,
 *           Accumulation · α) above a single uninterrupted pseudocode block.
 *           Each code line carries a coloured left-rule tying it back to
 *           its step; the indentation of the inner loop stays visible.
 *
 * Below the two-column block, a single hand-off sentence leads into Act 1.
 * The concrete numeric payoff is left to the explorer itself — the reader
 * discovers it by picking a preset and watching the cost counts update.
 */

function renderSingleProseBlock(blocks = [], keyPrefix = 'main-prose-block') {
  return renderProseBlocks(blocks, { keyPrefix })[0] ?? null;
}

const JUSTIFIED_PROSE_STYLE = { textAlign: 'justify' };

// V3.1 §C04 tooltip strings (registries.md §4 Tooltips)
const TOOLTIP_VISIBLE_LABEL = 'Visible/output label. It survives as an axis of the result.';
const TOOLTIP_SUMMED_LABEL = 'Summed label. The evaluator loops over this label and accumulates it away.';
const TOOLTIP_DECLARED_SYMMETRY = 'Declared operand symmetry. This creates candidate product symmetries, but they still need certification.';

// V3.1 §C05 — Label Role Legend.
// Three roles: visible (V), summed (W), declared symmetric axes. Each chip:
//  - Hover/focus → fires onHoveredLabelsChange with the role's label set
//                  (broadcasts on the page-wide hoveredLabels bus, so the
//                   formula and the chip lists in the einsum prose light up).
//  - Click       → toggles a "lock" so the broadcast persists until clicked
//                   again. A second chip's lock supersedes any prior lock.
//  - Reverse     → if hoveredLabels (read from the same bus, passed in as a
//                   prop here) intersects this role's label set, the chip
//                   gets a coral pulse/ring. This is the symmetric direction:
//                   hovering a label in the formula lights up the legend role
//                   that label belongs to.
//
// The lock state is local to the legend (one shared "locked role" id). When
// locked, hover events from elsewhere still update reverse-highlight, but the
// outgoing broadcast is pinned to the locked set.
function LabelRoleLegend({
  view,
  freeLabelColor,
  summedLabelColor,
  hoveredLabels,
  onHoveredLabelsChange,
}) {
  const [lockedRole, setLockedRole] = useState(null);

  const items = useMemo(() => {
    const declaredCount = Array.isArray(view?.declaredSymmetricLabels)
      ? view.declaredSymmetricLabels.length
      : 0;
    return [
      {
        id: 'visible',
        labels: Array.isArray(view?.vFreeLabels) ? view.vFreeLabels : [],
        label: 'visible',
        suffix: ' labels (stay on output)',
        tooltip: TOOLTIP_VISIBLE_LABEL,
        ariaLabel: 'Visible labels — highlight the V-set across visible components',
      },
      {
        id: 'summed',
        labels: Array.isArray(view?.wSummedLabels) ? view.wSummedLabels : [],
        label: 'summed',
        suffix: ' labels (collapse under sum)',
        tooltip: TOOLTIP_SUMMED_LABEL,
        ariaLabel: 'Summed labels — highlight the W-set across visible components',
      },
      {
        id: 'declared',
        labels: declaredCount > 0
          ? view.declaredSymmetricLabels
          : [],
        // Coral is the role accent for declared symmetric axes — the
        // "where symmetry enters" callout already uses coral as its eyebrow,
        // so the visual mapping is consistent (see dotStyle below).
        label: 'declared',
        suffix: ' symmetric axes',
        tooltip: TOOLTIP_DECLARED_SYMMETRY,
        ariaLabel: 'Declared symmetric axes — highlight the labels under declared operand symmetries',
        emptyHint: declaredCount === 0 ? ' (none)' : null,
      },
    ];
  }, [view]);

  const hoveredSet = hoveredLabels instanceof Set ? hoveredLabels : null;

  const handleHover = useCallback((labels) => {
    if (!onHoveredLabelsChange) return;
    if (lockedRole) return; // lock supersedes hover broadcast
    if (!labels || labels.length === 0) {
      onHoveredLabelsChange(null);
      return;
    }
    onHoveredLabelsChange(new Set(labels));
  }, [onHoveredLabelsChange, lockedRole]);

  const handleLeave = useCallback(() => {
    if (!onHoveredLabelsChange) return;
    if (lockedRole) return;
    onHoveredLabelsChange(null);
  }, [onHoveredLabelsChange, lockedRole]);

  const handleClick = useCallback((roleId, labels) => {
    if (!onHoveredLabelsChange) return;
    if (lockedRole === roleId) {
      // unlock + clear
      setLockedRole(null);
      onHoveredLabelsChange(null);
      return;
    }
    setLockedRole(roleId);
    if (labels && labels.length > 0) {
      onHoveredLabelsChange(new Set(labels));
    } else {
      onHoveredLabelsChange(null);
    }
  }, [onHoveredLabelsChange, lockedRole]);

  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-stone-700">
      {items.map((item) => {
        // Reverse-highlight: any of this role's labels currently in the bus?
        const isReverseHit = hoveredSet
          && item.labels.length > 0
          && item.labels.some((ch) => hoveredSet.has(ch));
        const isLocked = lockedRole === item.id;
        const isInteractive = Boolean(onHoveredLabelsChange) && item.labels.length > 0;
        // Theme-bound dot color: visible→freeLabelColor, summed→summedLabelColor,
        // declared→coral. The first two literal references make the
        // theme-role contract grep-visible (preamble.test.mjs).
        const dotStyle = item.id === 'visible'
          ? { backgroundColor: freeLabelColor }
          : item.id === 'summed'
            ? { backgroundColor: summedLabelColor }
            : { backgroundColor: 'var(--coral)' };
        return (
          <button
            key={item.id}
            type="button"
            data-role={item.id}
            data-locked={isLocked ? 'true' : 'false'}
            data-reverse-hit={isReverseHit ? 'true' : 'false'}
            disabled={!isInteractive}
            aria-label={item.ariaLabel}
            aria-pressed={isLocked}
            title={item.tooltip}
            onMouseEnter={isInteractive ? () => handleHover(item.labels) : undefined}
            onMouseLeave={isInteractive ? () => handleLeave() : undefined}
            onFocus={isInteractive ? () => handleHover(item.labels) : undefined}
            onBlur={isInteractive ? () => handleLeave() : undefined}
            onClick={isInteractive ? () => handleClick(item.id, item.labels) : undefined}
            className={cn(
              'inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-left transition-all',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
              isInteractive
                ? 'cursor-pointer border-transparent hover:border-stone-200 hover:bg-stone-50'
                : 'cursor-default border-transparent opacity-80',
              (isReverseHit || isLocked)
                && 'border-[color:var(--coral)] bg-[color:color-mix(in_oklab,var(--coral)_8%,white)] ring-2 ring-[color:var(--coral)]/40 animate-pulse',
            )}
          >
            <span
              className="h-2.5 w-2.5 rounded-full"
              aria-hidden="true"
              style={dotStyle}
            />
            <span>
              <strong className="font-semibold text-stone-900">{item.label}</strong>
              {item.suffix}
              {item.emptyHint && <span className="text-stone-500">{item.emptyHint}</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// Renders V/W label chips with hover interaction and V3.1 tooltips.
// Each label is a span that fires onHoveredLabelsChange on enter/leave.
// Highlighted when the label is in hoveredLabels.
function LabelChipList({ labels, tooltip, hoveredLabels, onHoveredLabelsChange, colorStyle }) {
  if (!labels || labels.length === 0) return <span className="font-mono text-stone-500">∅</span>;
  const hasHover = hoveredLabels instanceof Set && hoveredLabels.size > 0;
  return (
    <>
      {labels.split(', ').map((label, idx) => {
        const ch = label.trim();
        const isHit = hasHover && hoveredLabels.has(ch);
        return (
          <span key={idx} className="contents">
            {idx > 0 && <span className="font-mono text-stone-400">, </span>}
            <span
              className={cn(
                'cursor-default font-mono font-semibold',
                isHit
                  ? 'text-black underline decoration-primary decoration-2 underline-offset-2'
                  : onHoveredLabelsChange
                    ? 'hover:text-black hover:underline hover:decoration-primary hover:decoration-2 hover:underline-offset-2'
                    : undefined,
              )}
              style={isHit ? undefined : colorStyle}
              aria-label={tooltip}
              title={tooltip}
              onMouseEnter={onHoveredLabelsChange ? () => onHoveredLabelsChange(new Set([ch])) : undefined}
              onMouseLeave={onHoveredLabelsChange ? () => onHoveredLabelsChange(null) : undefined}
              onFocus={onHoveredLabelsChange ? () => onHoveredLabelsChange(new Set([ch])) : undefined}
              onBlur={onHoveredLabelsChange ? () => onHoveredLabelsChange(null) : undefined}
              tabIndex={onHoveredLabelsChange ? 0 : undefined}
              role={onHoveredLabelsChange ? 'button' : undefined}
            >
              {ch}
            </span>
          </span>
        );
      })}
    </>
  );
}

function EinsumIntroColumn({ example, hoveredLabels, onHoveredLabelsChange }) {
  const explorerThemeId = getActiveExplorerThemeId();
  const freeLabelColor = explorerThemeColor(explorerThemeId, 'hero');
  const summedLabelColor = explorerThemeColor(explorerThemeId, 'summedSide');
  const view = buildSection1ExampleView(example, {
    freeLabelColor: explorerThemeColor(explorerThemeId, 'hero'),
    summedLabelColor: explorerThemeColor(explorerThemeId, 'summedSide'),
  });
  if (!view) return null;
  const coloredVFreeNotation = String.raw`\textcolor{${freeLabelColor}}{${notationLatex('v_free')}}`;
  const coloredWSummedNotation = String.raw`\textcolor{${summedLabelColor}}{${notationLatex('w_summed')}}`;

  return (
    <div id="einsum-contraction" className="flex h-full flex-col scroll-mt-sticky">
      <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
        <AnchorLink anchorId="einsum-contraction" labelText="Einsum contraction">
          Einsum contraction
        </AnchorLink>
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        A tensor operation, written as one equation
      </h3>

      <p
        className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700"
        style={JUSTIFIED_PROSE_STYLE}
      >
        {renderSingleProseBlock(mainPreamble.slots.einsumIntroBeforeSummed, 'einsum-intro-before-summed')}
        <strong className="font-semibold">summed over</strong>
        {renderSingleProseBlock(mainPreamble.slots.einsumIntroBetweenSummedAndFree, 'einsum-intro-between-summed-free')}
        <strong className="font-semibold">free</strong>
        {renderSingleProseBlock(mainPreamble.slots.einsumIntroAfterFree, 'einsum-intro-after-free')}
      </p>

      <div className="mt-6 overflow-x-auto rounded-2xl border border-stone-200 bg-white px-5 py-6">
        {/* Formula line — wired to the page-wide label-hover bus (C04 §C01 pattern).
            aria-label uses view.exactEinsumText as the canonical plain-text form
            (accessible to screen readers and tooltips). */}
        <div className="flex justify-center">
          <code
            className="rounded-md px-2 py-1 font-mono text-[16px] font-semibold tracking-[0.01em] text-stone-800"
            aria-label={view.exactEinsumText}
          >
            <FormulaHighlighted
              example={example}
              hoveredLabels={hoveredLabels}
              onHoveredLabelsChange={onHoveredLabelsChange}
            />
          </code>
        </div>
        <div className="mt-2.5 flex justify-center text-[19px]">
          <Latex display math={view.expandedEquationLatex} />
        </div>
        <p
          className="mx-auto mt-4 max-w-[46rem] text-[13px] leading-6 text-stone-600"
          style={JUSTIFIED_PROSE_STYLE}
        >
          <strong className="font-semibold text-stone-900">
            {view.operandCount} operand{view.operandCount === 1 ? '' : 's'}, {view.labelCount} label{view.labelCount === 1 ? '' : 's'}.
          </strong>{' '}
          <span>The summed labels </span>
          <Latex math={coloredWSummedNotation} />
          <span> = {'{'}</span>
          <LabelChipList
            labels={view.wSummedSummary}
            tooltip={TOOLTIP_SUMMED_LABEL}
            hoveredLabels={hoveredLabels}
            onHoveredLabelsChange={onHoveredLabelsChange}
            colorStyle={{ color: summedLabelColor }}
          />
          <span>{'}'} collapse under </span>
          <Latex math={String.raw`\sum`} />
          <span>; the free labels </span>
          <Latex math={coloredVFreeNotation} />
          <span> = {'{'}</span>
          <LabelChipList
            labels={view.vFreeSummary}
            tooltip={TOOLTIP_VISIBLE_LABEL}
            hoveredLabels={hoveredLabels}
            onHoveredLabelsChange={onHoveredLabelsChange}
            colorStyle={{ color: freeLabelColor }}
          />
          <span>{'}'} survive as the axes of </span>
          <Latex math="R" />
          <span>. Declared symmetries: </span>
          <span aria-label={TOOLTIP_DECLARED_SYMMETRY} title={TOOLTIP_DECLARED_SYMMETRY}>
            {view.declaredSymmetrySummary}
          </span>
          <span>. The dense direct grid has </span>
          <Latex math={view.denseGridScalingLatex} />
          <span> assignments before symmetry is used.</span>
        </p>
      </div>

      <LabelRoleLegend
        view={view}
        freeLabelColor={freeLabelColor}
        summedLabelColor={summedLabelColor}
        hoveredLabels={hoveredLabels}
        onHoveredLabelsChange={onHoveredLabelsChange}
      />

      {/* Spacer: anchors the callout to the bottom when the right column is taller
          (so both columns end at the same y), with a mt-6 minimum gap when not. */}
      <div className="mt-6 flex-1" aria-hidden="true" />

      <EditorialCallout
        id="where-symmetry-enters"
        className="scroll-mt-sticky"
        label={(
          <AnchorLink
            anchorId="where-symmetry-enters"
            labelText="Where symmetry enters"
            hashGlyphClassName="text-[12px] text-primary/70"
          >
            Where symmetry enters
          </AnchorLink>
        )}
        title="Not every product is distinct"
        bodyClassName="mt-2 text-[14px] leading-7 text-foreground"
        footer={(
          <>
            {renderSingleProseBlock(mainPreamble.slots.calloutFooter, 'symmetry-callout-footer')}
          </>
        )}
      >
        <>
          {renderSingleProseBlock(mainPreamble.slots.calloutBodyBeforeGroup, 'symmetry-callout-before-group')}
          <strong className="font-semibold">
            group <Latex math="G" />
          </strong>
          {renderSingleProseBlock(mainPreamble.slots.calloutBodyBetweenGroupAndOrbits, 'symmetry-callout-between-group-orbits')}
          <em>orbits</em>
          {renderSingleProseBlock(mainPreamble.slots.calloutBodyAfterOrbits, 'symmetry-callout-after-orbits')}
        </>
      </EditorialCallout>
    </div>
  );
}

function MentalFrameworkColumn({ example }) {
  return (
    <div id="mental-framework" className="flex h-full flex-col scroll-mt-sticky">
      <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
        <AnchorLink anchorId="mental-framework" labelText="The mental framework">
          The mental framework
        </AnchorLink>
      </span>
      <h3 className="mt-1 font-heading text-lg font-semibold text-foreground">
        The same two loops — dense or symmetric
      </h3>

      <p
        className="mt-3 font-serif text-[17px] leading-[1.75] text-gray-700"
        style={JUSTIFIED_PROSE_STYLE}
      >
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroBeforeRepSet, 'mental-framework-before-repset')}
        <code className="explorer-inline-code">RepSet</code>
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroBetweenRepSetAndOuts, 'mental-framework-between-repset-outs')}
        <code className="explorer-inline-code">Outs(rep)</code>
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroBetweenOutsAndCoeff, 'mental-framework-between-outs-coeff')}
        <code className="explorer-inline-code">coeff(rep, out_rep)</code>
        {renderSingleProseBlock(mainPreamble.slots.mentalFrameworkIntroAfterCoeff, 'mental-framework-after-coeff')}
      </p>

      <div className="mt-6 flex flex-1 flex-col">
        <MentalFrameworkCode example={example} />
      </div>
    </div>
  );
}

export default function AlgorithmAtAGlance({
  example,
  hoveredLabels = null,
  onHoveredLabelsChange = null,
}) {
  return (
    <section id="algorithm-at-a-glance" aria-labelledby="algorithm-at-a-glance-title" className="mb-10 scroll-mt-sticky">
      <ExplorerSectionCard
        eyebrow={
          <AnchorLink anchorId="algorithm-at-a-glance" labelText="Einsum at a Glance">
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">
              Einsum at a Glance
            </span>
          </AnchorLink>
        }
        title={<span id="algorithm-at-a-glance-title">{mainPreamble.title}</span>}
        description={<InlineMathText>{mainPreamble.deck}</InlineMathText>}
        className="border-gray-200 bg-white"
        contentClassName="pt-6"
      >
        {/* Two-column top: einsum notation (L) ↔ mental framework code (R).
            items-stretch makes both columns reach the same y by design:
            whichever side is naturally shorter grows a spacer to fill. On the
            left the "Where symmetry enters" callout sticks to the bottom; on
            the right the MentalFrameworkCode figure stretches so its Counting
            Convention band anchors to the bottom. */}
        <div className="editorial-two-col-divider-lg grid grid-cols-1 items-stretch gap-8 lg:grid-cols-2 lg:gap-10">
          <EinsumIntroColumn
            example={example}
            hoveredLabels={hoveredLabels}
            onHoveredLabelsChange={onHoveredLabelsChange}
          />
          <MentalFrameworkColumn example={example} />
        </div>

        <p
          className="mt-10 border-t border-stone-200 pt-8 font-serif text-[17px] leading-[1.75] text-gray-700"
          style={JUSTIFIED_PROSE_STYLE}
        >
          {renderSingleProseBlock(mainPreamble.slots.handoffBeforeSectionLink, 'handoff-before-section-link')}
          <SectionReferenceLink href="#setup">Section 1</SectionReferenceLink>
          {renderSingleProseBlock(mainPreamble.slots.handoffAfterSectionLink, 'handoff-after-section-link')}
        </p>
      </ExplorerSectionCard>
    </section>
  );
}
