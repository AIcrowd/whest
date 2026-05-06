/**
 * UnavailableDetailsPanel — §49 V3.1 polish (C49)
 *
 * Renders the V3.1 "α unavailable under the current interactive budget"
 * surfacing for a single per-row Unavailable case. The amber pill in
 * ComponentCostView is preserved as the visual trigger; this panel sits
 * directly beneath the row inside a <details>/<summary> collapsible so the
 * table layout is undisturbed when no one expands it.
 *
 * Live numbers — both budget conditions:
 *   • typed-pattern count : generateTypedSetPartitions(sizes).length
 *                           (Bell-number-like; guarded by a max-position cap
 *                            so we never enumerate past 12 positions).
 *   • brute-force count   : bruteForceEstimate(sizes, groupSize)
 *                           (= |G_a| · ∏ n_ℓ, the corrected pair-touch cost).
 *
 * "Feature, not an error" framing — the panel reinforces exactness. We use
 * neutral gray + amber-warning tones (no red), and the heading is the
 * verbatim V3.1 caption.
 *
 * No engine touches: we reuse `bruteForceEstimate`, `BRUTE_FORCE_BUDGET`, and
 * `generateTypedSetPartitions` exactly as they are exported.
 *
 * All colours via design-system tokens (CSS variables). No raw notation hex
 * literals — the notation-system audit forbids brand/notation colours as
 * string literals in any source file (use var(--coral) etc. instead).
 */

import { useMemo } from 'react';
import {
  bruteForceEstimate,
  BRUTE_FORCE_BUDGET,
} from '../engine/budget.js';
import { generateTypedSetPartitions } from '../engine/partition/typedPartitions.js';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors NaiveAlphaCostMeter / TuplePatternMeter)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  warning:    'var(--warning)',      // amber — pill register
  // Amber surface tint composed from --warning via color-mix; keeps token
  // discipline (no hard-coded RGB values) and inherits theme overrides.
  warningBg:  'color-mix(in srgb, var(--warning) 12%, transparent)',
  gray900:    'var(--gray-900)',     // body ink
  gray700:    'var(--gray-700)',     // strong label
  gray600:    'var(--gray-600)',     // secondary text
  gray500:    'var(--gray-500)',     // muted label
  gray400:    'var(--gray-400)',     // disabled / divider strong
  gray200:    'var(--gray-200)',     // divider
  gray100:    'var(--gray-100)',     // surface inset
  gray50:     'var(--gray-50)',      // surface deep
  white:      'var(--white)',
};

/* ─────────────────────────────────────────────────────────────────────────────
   Pattern-enumeration guard
   generateTypedSetPartitions runs over Bell numbers of |sizes|. Bell(12) is
   ~4M; Bell(13) ~28M — past 12 positions we stop and report "over budget"
   without enumerating. The same guard pattern as TuplePatternMeter (C37).
   ───────────────────────────────────────────────────────────────────────────── */
const PATTERN_BUDGET_MAX_POSITIONS = 12;

/**
 * Compute live numbers for both budget conditions.
 *
 * Returns:
 *   {
 *     typedPartitionCount   : number | null   // null when uncomputable
 *     typedPartitionOverBudget : boolean
 *     bruteForceCount       : number          // groupSize · Π sizes
 *     bruteForceOverBudget  : boolean
 *   }
 */
function computeLiveNumbers(sizes, groupSize) {
  const safeSizes = Array.isArray(sizes) ? sizes : [];
  const safeGroupSize = Math.max(1, Number(groupSize) || 1);

  // Brute-force is always cheap to compute (it's a product).
  const bruteForceCount = bruteForceEstimate(safeSizes, safeGroupSize);
  const bruteForceOverBudget = bruteForceCount > BRUTE_FORCE_BUDGET;

  // Typed-partition enumeration is Bell-number-like; guard by position count.
  let typedPartitionCount = null;
  let typedPartitionOverBudget = false;
  if (safeSizes.length > PATTERN_BUDGET_MAX_POSITIONS) {
    typedPartitionOverBudget = true;
  } else if (safeSizes.length > 0) {
    try {
      const partitions = generateTypedSetPartitions(safeSizes);
      typedPartitionCount = partitions.length;
      // Compare against the same interactive budget — these are pure-count
      // pattern objects, but the budget is the unit ceiling for either.
      typedPartitionOverBudget = typedPartitionCount > BRUTE_FORCE_BUDGET;
    } catch {
      // Defensive: if the enumeration throws, treat as over-budget.
      typedPartitionOverBudget = true;
    }
  } else {
    // No sizes — degenerate; the partition count is trivially 1 (empty
    // partition) but we surface n/a since there's nothing meaningful to
    // report.
    typedPartitionCount = 1;
  }

  return {
    typedPartitionCount,
    typedPartitionOverBudget,
    bruteForceCount,
    bruteForceOverBudget,
  };
}

/** Format a numeric live count with locale separators or "n/a" when null. */
function fmtCount(n) {
  if (n === null || n === undefined) return 'n/a';
  if (!Number.isFinite(n)) return 'n/a';
  return n.toLocaleString();
}

/* ─────────────────────────────────────────────────────────────────────────────
   Failed-condition badge — small inline marker next to the matching row
   ───────────────────────────────────────────────────────────────────────────── */
function ConditionBadge() {
  return (
    <span
      className="ml-2 inline-flex items-center rounded-full px-1.5 py-0.5 align-middle text-[9.5px] font-semibold uppercase tracking-[0.12em]"
      style={{ backgroundColor: TOKEN.warningBg, color: TOKEN.warning }}
      aria-label="This condition triggered the unavailable state"
    >
      this one
    </span>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * UnavailableDetailsPanel
 *
 * Props:
 *   componentId      {string}                   — the comma-joined label list, used in aria-label
 *   sizes            {number[]}                 — per-position label sizes (for live numbers)
 *   groupSize        {number}                   — |G_a|, the local pointwise group order
 *   failedCondition  {'typed-partition'|'brute-force'|'no-shortcut'|null}
 *                                               — heuristic from the engine trace
 *   onLowerN         {((n: number) => void)|null}
 *                                               — when present, the "Try n=…" CTA dispatches this
 *   currentN         {number}                   — current dimensionN (for the suggested-n math)
 */
export default function UnavailableDetailsPanel({
  componentId = '',
  sizes = [],
  groupSize = 1,
  failedCondition = 'no-shortcut',
  onLowerN = null,
  currentN = 3,
}) {
  const live = useMemo(
    () => computeLiveNumbers(sizes, groupSize),
    [sizes, groupSize],
  );

  // Suggested smaller n: drop two, clamped to ≥ 2. The math works because the
  // brute-force cost scales like n^k · |G|, so even a 2-step drop is a
  // dramatic reduction in most heterogeneous-product cases.
  const suggestedN = Math.max(2, Math.floor(currentN) - 2);
  const canLowerN = typeof onLowerN === 'function' && suggestedN < currentN;

  const isFailedTypedPartition = failedCondition === 'typed-partition';
  const isFailedBruteForce = failedCondition === 'brute-force';
  // 'no-shortcut' is the catch-all when no specific budget condition was
  // recorded — typically when no regime fired at all.
  const isFailedNoShortcut = failedCondition === 'no-shortcut' || failedCondition == null;

  const ariaLabel = componentId
    ? `Unavailable count details for component ${componentId}`
    : 'Unavailable count details';

  return (
    <section
      role="region"
      aria-label="Unavailable count details"
      className="mt-2 rounded-md border px-3 py-2.5 text-[12px] leading-5"
      style={{
        backgroundColor: TOKEN.gray50,
        borderColor: TOKEN.gray200,
        color: TOKEN.gray700,
      }}
      data-testid="unavailable-details-panel"
      data-component-id={componentId}
      data-failed-condition={failedCondition ?? 'no-shortcut'}
    >
      {/* Heading — verbatim V3.1 caption */}
      <p
        className="mb-2 font-semibold"
        style={{ color: TOKEN.gray900 }}
        data-testid="unavailable-details-heading"
      >
        α unavailable under the current interactive budget
      </p>

      {/* Four V3.1 detail lines, each as its own paragraph. The matching
          condition gets a small "this one" badge so the user can see which
          gate fired. */}
      <p className="mb-1" data-testid="unavailable-detail-no-shortcut">
        No exact shortcut applies.
        {isFailedNoShortcut ? <ConditionBadge /> : null}
      </p>
      <p className="mb-1" data-testid="unavailable-detail-typed-partition">
        <span>
          Typed partition patterns exceed budget:{' '}
          <code
            className="font-mono font-semibold"
            style={{ color: TOKEN.gray900 }}
          >
            {fmtCount(live.typedPartitionCount)}
          </code>
          {live.typedPartitionCount !== null ? (
            <span style={{ color: TOKEN.gray500 }}>
              {' '}/ {BRUTE_FORCE_BUDGET.toLocaleString()}
            </span>
          ) : null}
          .
        </span>
        {isFailedTypedPartition ? <ConditionBadge /> : null}
      </p>
      <p className="mb-1" data-testid="unavailable-detail-brute-force">
        <span>
          Corrected brute force pair touches exceed budget:{' '}
          <code
            className="font-mono font-semibold"
            style={{ color: TOKEN.gray900 }}
          >
            {fmtCount(live.bruteForceCount)}
          </code>
          <span style={{ color: TOKEN.gray500 }}>
            {' '}/ {BRUTE_FORCE_BUDGET.toLocaleString()}
          </span>
          .
        </span>
        {isFailedBruteForce ? <ConditionBadge /> : null}
      </p>
      <p className="mb-2" data-testid="unavailable-detail-honest-report">
        The page reports unavailable rather than guessing.
      </p>

      {/* CTA row — lower-n button + appendix link. Both inherit from the
          design-system token palette so the focus rings match the rest of
          the explorer. */}
      <div className="mt-2.5 flex flex-wrap items-center gap-3">
        {canLowerN ? (
          <button
            type="button"
            onClick={() => onLowerN?.(suggestedN)}
            aria-label={`Try a smaller dimension: lower n to ${suggestedN}`}
            className="rounded-md border px-2.5 py-1 text-[11.5px] font-semibold focus:outline-none focus-visible:ring-2"
            style={{
              backgroundColor: TOKEN.white,
              borderColor: TOKEN.gray400,
              color: TOKEN.gray900,
            }}
            data-testid="unavailable-lower-n-cta"
          >
            Try n = {suggestedN}
          </button>
        ) : null}
        <a
          href="#appendix-section-7"
          aria-label="Read Appendix B.9 — typed partition counting theorem"
          className="rounded-md text-[11.5px] font-semibold underline decoration-dotted underline-offset-[3px] focus:outline-none focus-visible:ring-2"
          style={{ color: TOKEN.gray700 }}
          data-testid="unavailable-appendix-link"
        >
          Read Appendix B.9 →
        </a>
      </div>

      {/* Honest framing footer — V3.1 usability rule: this is a feature,
          not an error. Kept tiny so it doesn't dominate the panel. */}
      <p
        className="mt-2 text-[10.5px]"
        style={{ color: TOKEN.gray500 }}
        data-testid="unavailable-feature-not-error"
      >
        This is a feature, not an error — the page reinforces exactness by
        refusing to guess past the interactive budget.
      </p>

      {/* The aria-label dependency is rendered via the section role above;
          we expose the panel id via data-testid for the test suite. */}
      <span className="sr-only">{ariaLabel}</span>
    </section>
  );
}
