/**
 * TuplePatternMeter — §8 Typed Partition Counting (C37)
 *
 * Motivates partition counting by side-by-siding four enumeration sizes:
 *   1. dense assignments  |X_a|              = n^k
 *   2. tuple/group touches |X_a| × |G_a|     (full Burnside enumeration)
 *   3. typed patterns      # typed equality patterns of L_a (from engine)
 *   4. pattern orbits      typed-patterns / |G_a| (the actual α via partitions)
 *
 * Same α, fewer objects counted. The first two bars are baseline (gray), the
 * last two are the savings story (coral).
 *
 * Bars are log-scaled (orders of magnitude differ; linear would compress the
 * smaller values to invisibility). Each bar carries a hover tooltip with a
 * short definition + the live numeric value.
 *
 * Over-budget caveat: when the typed-pattern enumeration would exceed the
 * interactive budget (or no componentData is available), the last two bars
 * display "—" and the caption appends "when pattern budget passes" — the
 * V3.1 usability rule against implying asymptotic universal superiority.
 *
 * All colours via design-system tokens. No raw hex outside TOKEN.
 * Reduced-motion respected via prefers-reduced-motion.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  generateTypedSetPartitions,
  partitionOrbitReps,
} from '../engine/partition/typedPartitions.js';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors NaiveAlphaCostMeter)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:      'var(--coral)',        // brand coral
  coralLight: 'var(--coral-light)',  // coral surface tint
  gray900:    'var(--gray-900)',     // body ink
  gray700:    'var(--gray-700)',     // strong label
  gray600:    'var(--gray-600)',     // secondary text
  gray500:    'var(--gray-500)',     // muted label
  gray400:    'var(--gray-400)',     // disabled
  gray200:    'var(--gray-200)',     // divider
  gray100:    'var(--gray-100)',     // bar track
  gray50:     'var(--gray-50)',      // surface inset
  white:      'var(--white)',
};

/* ─────────────────────────────────────────────────────────────────────────────
   Pattern-enumeration budget
   The engine's generateTypedSetPartitions runs over Bell numbers of |sizes|.
   Bell(12) ≈ 4 213 597; Bell(13) ≈ 27 644 437 — past 12 we stop and report.
   ───────────────────────────────────────────────────────────────────────────── */
const PATTERN_BUDGET_MAX_LABELS = 12;

/* ─────────────────────────────────────────────────────────────────────────────
   Bar definitions (id, label, accent role, tooltip)
   ───────────────────────────────────────────────────────────────────────────── */
const BARS = [
  {
    id: 'dense',
    label: 'dense assignments |X_a|',
    accent: 'gray',
    definition: 'Every (i,j,k,...) tuple of label values, with no symmetry quotient applied.',
  },
  {
    id: 'tupleGroup',
    label: 'tuple/group touches',
    accent: 'gray',
    definition: 'Full Burnside enumeration cost: each dense assignment touched by each pointwise group element.',
  },
  {
    id: 'typedPatterns',
    label: 'typed patterns',
    accent: 'coral',
    definition: 'Typed equality patterns of the label set: how many distinct ways labels can collide before quotienting.',
  },
  {
    id: 'patternOrbits',
    label: 'pattern orbits',
    accent: 'coral',
    definition: 'Typed patterns modulo the pointwise group action — the actual α produced by partition counting.',
  },
];

/* ─────────────────────────────────────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────────────────────────────────────── */

/** Format large numbers compactly (mirrors NaiveAlphaCostMeter.fmt). */
function fmt(n) {
  if (n === null || n === undefined) return '—';
  if (!Number.isFinite(n) || n > 1e18) return '> 10¹⁸';
  if (n >= 1e12) return `${(n / 1e12).toPrecision(3)}T`;
  if (n >= 1e9)  return `${(n / 1e9).toPrecision(3)}B`;
  if (n >= 1e6)  return `${(n / 1e6).toPrecision(3)}M`;
  if (n >= 1e3)  return `${(n / 1e3).toPrecision(3)}k`;
  return String(Math.round(n));
}

/**
 * Compute the four enumeration sizes from props.
 *
 * Returns:
 *   {
 *     denseAssignments,
 *     tupleGroupTouches,
 *     typedPatterns,    // null if over-budget or unavailable
 *     patternOrbits,    // null if over-budget or unavailable
 *     overBudget,       // true when pattern enumeration was refused
 *   }
 */
function computeMeter({ dimensionN, allLabels, groupSize, componentData }) {
  const n = Math.max(1, dimensionN);
  const k = allLabels.length;
  const denseAssignments = Math.pow(n, k);
  const tupleGroupTouches = denseAssignments * Math.max(1, groupSize);

  // Pattern enumeration: pull from the same primitive TypedPartitionDemo uses.
  // Safe access: when componentData is missing or labels exceed the budget,
  // we surface "—" + the V3.1 caveat — never silently report 0.
  const activeComponent = componentData?.components?.[0] ?? null;
  const sizes = activeComponent?.sizes ?? [];
  const elements = activeComponent?.elements ?? [];

  const haveData = sizes.length > 0;
  const withinBudget = sizes.length <= PATTERN_BUDGET_MAX_LABELS;

  let typedPatterns = null;
  let patternOrbits = null;
  let overBudget = !haveData || !withinBudget;

  if (haveData && withinBudget) {
    const allPartitions = generateTypedSetPartitions(sizes);
    typedPatterns = allPartitions.length;
    if (elements.length > 0) {
      patternOrbits = partitionOrbitReps(allPartitions, elements).length;
    } else {
      // No group elements available — pattern orbits collapse to typed patterns.
      patternOrbits = typedPatterns;
    }
  }

  return {
    denseAssignments,
    tupleGroupTouches,
    typedPatterns,
    patternOrbits,
    overBudget,
  };
}

/** Pick the bar's accent colour from its role. */
function accentColor(role) {
  return role === 'coral' ? TOKEN.coral : TOKEN.gray600;
}

/** Pick the bar's filled-track colour (slightly washed for the gray bars). */
function fillColor(role) {
  return role === 'coral' ? TOKEN.coral : TOKEN.gray400;
}

/* ─────────────────────────────────────────────────────────────────────────────
   Tooltip (fixed-position, follows pointer rect)
   ───────────────────────────────────────────────────────────────────────────── */
function BarTooltip({ anchorRect, content }) {
  const TOOLTIP_W = 300;
  const [pos, setPos] = useState(null);

  useEffect(() => {
    if (!anchorRect) { setPos(null); return; }
    const vw = document.documentElement.clientWidth;
    let x = anchorRect.left + anchorRect.width / 2;
    x = Math.max(TOOLTIP_W / 2 + 8, Math.min(x, vw - TOOLTIP_W / 2 - 8));
    const y = anchorRect.top - 8;
    setPos({ x, y });
  }, [anchorRect]);

  if (!anchorRect || !pos || !content) return null;

  return (
    <div
      className="pointer-events-none fixed z-[9999] max-w-[calc(100vw-2rem)] rounded-lg border border-stone-200 bg-white px-3 py-2.5 text-xs leading-5 shadow-[0_8px_32px_rgba(15,23,42,0.14)]"
      style={{
        left: pos.x,
        top: pos.y,
        width: TOOLTIP_W,
        transform: 'translateX(-50%) translateY(-100%)',
        color: TOKEN.gray700,
      }}
      role="tooltip"
      id="tuple-pattern-meter-tooltip"
    >
      {content}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Bar row — log-scaled horizontal bar + numeric badge
   ───────────────────────────────────────────────────────────────────────────── */
function BarRow({ bar, value, maxValue, prefersReducedMotion, onHover }) {
  const isUnavailable = value === null || value === undefined;
  // log10(x + 1) keeps small values visible; Math.max guards against
  // log(1)=0 producing NaN ratios.
  const logMax = Math.log10(Math.max(2, maxValue + 1));
  const logVal = isUnavailable ? 0 : Math.log10(value + 1);
  const widthPct = isUnavailable ? 0 : Math.max(2, (logVal / logMax) * 100);

  const fill = fillColor(bar.accent);
  const accent = accentColor(bar.accent);

  const tooltipContent = (
    <>
      <div style={{ color: TOKEN.gray900, fontWeight: 600 }} className="mb-1">
        {bar.label}
      </div>
      <div className="leading-5">{bar.definition}</div>
      <div className="mt-1 font-mono text-[11px]" style={{ color: TOKEN.gray500 }}>
        Currently: {isUnavailable ? '— (over budget)' : fmt(value)}
      </div>
    </>
  );

  return (
    <div
      className="flex items-center gap-3 py-1.5"
      role="listitem"
      data-bar-id={bar.id}
    >
      {/* Label (left column, fixed width) */}
      <span
        className="w-[170px] shrink-0 text-[11.5px] leading-5"
        style={{ color: TOKEN.gray700 }}
      >
        {bar.label}
      </span>

      {/* Bar track (flex column) */}
      <div
        className="relative h-5 flex-1 overflow-hidden rounded-sm"
        style={{ backgroundColor: TOKEN.gray100 }}
        tabIndex={0}
        role="button"
        aria-label={`${bar.label}: ${isUnavailable ? 'unavailable, over pattern budget' : fmt(value)}`}
        onMouseEnter={(e) => onHover(tooltipContent, e.currentTarget.getBoundingClientRect())}
        onMouseLeave={() => onHover(null, null)}
        onFocus={(e) => onHover(tooltipContent, e.currentTarget.getBoundingClientRect())}
        onBlur={() => onHover(null, null)}
      >
        <div
          style={{
            width: `${widthPct}%`,
            height: '100%',
            backgroundColor: fill,
            transition: prefersReducedMotion ? 'none' : 'width 0.35s ease-out',
            cursor: 'help',
          }}
          aria-hidden="true"
        />
      </div>

      {/* Live numeric badge (right column, fixed width) */}
      <span
        className="w-[78px] shrink-0 text-right font-mono text-[11.5px] font-semibold tabular-nums"
        style={{ color: isUnavailable ? TOKEN.gray400 : accent }}
        aria-live="polite"
        aria-atomic="true"
      >
        {isUnavailable ? '—' : fmt(value)}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * TuplePatternMeter
 *
 * Props:
 *   dimensionN     {number}          active dimension (positive integer)
 *   allLabels      {string[]}        all label strings
 *   groupSize      {number}          |G_a| — pointwise group size
 *   componentData  {object|null}     component data for typed-pattern enumeration
 */
export default function TuplePatternMeter({
  dimensionN = 3,
  allLabels = [],
  groupSize = 1,
  componentData = null,
}) {
  const [tooltipContent, setTooltipContent] = useState(null);
  const [tooltipAnchor, setTooltipAnchor] = useState(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  // Honour prefers-reduced-motion for bar animations.
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mq.matches);
    const handler = (e) => setPrefersReducedMotion(e.matches);
    mq.addEventListener?.('change', handler);
    return () => mq.removeEventListener?.('change', handler);
  }, []);

  const meter = useMemo(
    () => computeMeter({ dimensionN, allLabels, groupSize, componentData }),
    [dimensionN, allLabels, groupSize, componentData],
  );

  const handleHover = useCallback((content, rect) => {
    setTooltipContent(content);
    setTooltipAnchor(rect);
  }, []);

  // Dismiss tooltip on scroll / resize (parity with NaiveAlphaCostMeter).
  useEffect(() => {
    if (!tooltipAnchor) return undefined;
    const dismiss = () => { setTooltipContent(null); setTooltipAnchor(null); };
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
    };
  }, [tooltipAnchor]);

  const valueById = {
    dense: meter.denseAssignments,
    tupleGroup: meter.tupleGroupTouches,
    typedPatterns: meter.typedPatterns,
    patternOrbits: meter.patternOrbits,
  };

  // For the log scale we want one shared maximum so the relative magnitudes
  // are visually meaningful. Skip null values in the max computation.
  const maxValue = Math.max(
    1,
    ...Object.values(valueById).filter((v) => v !== null && v !== undefined && Number.isFinite(v)),
  );

  // Compose the caption: V3.1 spec base + over-budget caveat clause.
  const baseCaption = 'Same α, fewer objects counted.';
  const caveatCaption = ' (when pattern budget passes)';
  const captionText = meter.overBudget ? `${baseCaption}${caveatCaption}` : baseCaption;

  return (
    <div
      className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm"
      data-testid="tuple-pattern-meter"
    >
      {/* Header */}
      <div className="mb-3">
        <p
          className="text-[10px] font-semibold uppercase tracking-[0.15em]"
          style={{ color: TOKEN.gray500 }}
        >
          Tuple-vs-pattern enumeration
        </p>
        <p
          className="mt-0.5 text-[11.5px] leading-5"
          style={{ color: TOKEN.gray600 }}
        >
          Four enumeration sizes for the active component &mdash; the savings
          story is the gap between the gray bars and the coral bars.
        </p>
      </div>

      {/* Divider */}
      <div
        className="mb-3 h-px"
        style={{ backgroundColor: TOKEN.gray200 }}
        aria-hidden="true"
      />

      {/* Bars */}
      <div role="list" aria-label="Enumeration sizes for tuple vs pattern counting">
        {BARS.map((bar) => (
          <BarRow
            key={bar.id}
            bar={bar}
            value={valueById[bar.id]}
            maxValue={maxValue}
            prefersReducedMotion={prefersReducedMotion}
            onHover={handleHover}
          />
        ))}
      </div>

      {/* Divider */}
      <div
        className="mb-2 mt-3 h-px"
        style={{ backgroundColor: TOKEN.gray200 }}
        aria-hidden="true"
      />

      {/* Caption */}
      <p
        className="mt-1 text-[11.5px] leading-5"
        style={{ color: TOKEN.gray500 }}
        data-testid="tuple-pattern-meter-caption"
      >
        {captionText}
      </p>

      {/* Over-budget secondary line — explicit, non-marketing reading of
          why the last two bars are blank. Kept short to avoid bloating
          the card; full definitions live in the per-bar tooltips. */}
      {meter.overBudget && (
        <p
          className="mt-1 text-[11px] leading-5"
          style={{ color: TOKEN.gray500 }}
          data-testid="tuple-pattern-meter-over-budget-note"
        >
          Pattern enumeration unavailable for this preset &mdash; the typed
          partition count exceeds the interactive budget. The α via partitions
          would still match the engine, but we do not enumerate it here.
        </p>
      )}

      {/* Portal tooltip */}
      <BarTooltip anchorRect={tooltipAnchor} content={tooltipContent} />
    </div>
  );
}
