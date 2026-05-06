// V3.1 §41 — Live Result Sentence.
//
// A standalone four-line prose summary of the selected preset's final
// accounting, retained for the V3.1 §41 component contract. Section 9's
// publish-ready page now avoids repeating the dense-vs-symmetry result and
// mounts the later EditorialComparisonSpread as its single summary instead.
//
// Spec template (V3.1 §41, verbatim phrasing the test pins):
//   For the selected contraction, product representatives are counted by
//   [product method], giving M = [live].
//   Accumulation updates are counted by [alpha method], giving alpha = [live].
//   Therefore total = mu + alpha = [live].
//   Compared with the dense direct baseline [live], this is a [live] reduction.
//
// When α is unavailable for any component under the current interactive
// budget, the second line is replaced by the unavailable-explanation
// sentence so the prose never lies about a number we don't have.
//
// Visual register: serif body matching SectionFiveIntroBlock, with method
// names rendered as inline CaseBadge pills (clickable: opens regime
// tooltip). All color via CSS variables on the explorer theme — no raw
// notation hex.

import CaseBadge from './CaseBadge.jsx';

const SECTION_FIVE_THEME_OVERRIDE = 'editorial-noir-math';

// Default product-method label. M_a in this engine is always the Burnside
// orbit count of full label assignments under the local symmetry group;
// the formula `(1/|G_a|) ∑_g ∏_c n_c` is the Burnside ingredient. Callers
// can override via the `productMethod` prop if a regime ever ships an
// alternative product-orbit counter.
const DEFAULT_PRODUCT_METHOD = 'Burnside orbit count';

/**
 * Format a non-negative integer with locale-aware thousands separators.
 * Falls back to the input if it isn't a finite number — defensive against
 * unavailable counts upstream.
 */
function formatCount(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return String(n);
  return n.toLocaleString();
}

/**
 * Compute "% reduction" relative to a dense baseline.
 *   ((dense - actual) / dense * 100).toFixed(1)
 * Returns the formatted percent string ("42.7%") or "0.0%" when dense is
 * zero. Negative reductions (actual > dense) are reported honestly — this
 * is a comparison, not a marketing number.
 */
function reductionPct(dense, actual) {
  if (typeof dense !== 'number' || dense <= 0) return '0.0%';
  if (typeof actual !== 'number' || !Number.isFinite(actual)) return '0.0%';
  const pct = ((dense - actual) / dense) * 100;
  return `${pct.toFixed(1)}%`;
}

/**
 * Inline method badge — wraps CaseBadge in `xs` size with passthrough text
 * so the method name reads as part of the sentence flow rather than as a
 * floating chip. Clickable via CaseBadge's tooltip surface.
 *
 * If `regimeId` is null/undefined we fall back to a styled span so the
 * sentence still renders meaningfully when a method is supplied as plain
 * text (e.g. the default "Burnside orbit count").
 */
function MethodBadge({ regimeId, label }) {
  if (regimeId) {
    return (
      <CaseBadge
        regimeId={regimeId}
        size="xs"
        themeOverride={SECTION_FIVE_THEME_OVERRIDE}
        presentationThemeOverride={null}
      />
    );
  }
  return (
    <span
      className="inline-flex items-center rounded-full border border-border bg-surface-raised px-2 py-0.5 font-mono text-[10px] font-semibold text-foreground"
    >
      {label}
    </span>
  );
}

/**
 * LiveResultSentence — V3.1 §41 hero prose block.
 *
 * Props
 * ─────
 * presetName            — display name of the active example (e.g. "Pairwise dot")
 * productMethod         — { label, regimeId? } for the product-orbit method
 * alphaMethod           — { label, regimeId? } for the accumulation method
 * mu, alpha, total      — live integer counts
 * denseBaseline         — dense-direct baseline integer count
 * componentUnavailable  — { reason } when α is unavailable for any component;
 *                         when present, the second line is replaced by the
 *                         V3.1 unavailable-explanation sentence.
 */
export default function LiveResultSentence({
  presetName,
  productMethod,
  alphaMethod,
  mu,
  alpha,
  total,
  denseBaseline,
  componentUnavailable,
}) {
  const productLabel = productMethod?.label ?? DEFAULT_PRODUCT_METHOD;
  const productRegimeId = productMethod?.regimeId ?? null;
  const alphaLabel = alphaMethod?.label ?? 'the selected α regime';
  const alphaRegimeId = alphaMethod?.regimeId ?? null;

  const reduction = reductionPct(denseBaseline, total);

  // The middle paragraph (alpha line) is replaced with the V3.1
  // unavailable-explanation sentence when α can't be computed. The reason
  // string is supplied by the caller — typically "the brute-force budget
  // was exceeded" or "no regime fired for this mixed component".
  const unavailable = Boolean(componentUnavailable);
  const unavailableReason = componentUnavailable?.reason ?? 'no regime applied within the current interactive budget';

  return (
    <div className="mx-auto max-w-[44rem] space-y-3 text-center font-serif text-[15.5px] leading-[1.75] text-gray-700">
      {/* Line 1 — product representatives */}
      <p>
        For the selected contraction
        {presetName ? (
          <>
            {' '}
            (<span className="font-mono italic text-gray-900">{presetName}</span>)
          </>
        ) : null}
        , product representatives are counted by{' '}
        <MethodBadge regimeId={productRegimeId} label={productLabel} />
        , giving{' '}
        <span
          className="font-mono font-semibold text-gray-900"
          aria-live="polite"
          aria-atomic="true"
          data-testid="live-result-mu"
        >
          M = {formatCount(mu)}
        </span>
        .
      </p>

      {/* Line 2 — accumulation updates (or unavailable explanation) */}
      {unavailable ? (
        <p data-testid="live-result-unavailable">
          The exact alpha count is unavailable for this component under the
          current interactive budget because {unavailableReason}.
        </p>
      ) : (
        <p>
          Accumulation updates are counted by{' '}
          <MethodBadge regimeId={alphaRegimeId} label={alphaLabel} />
          , giving{' '}
          <span
            className="font-mono font-semibold text-gray-900"
            aria-live="polite"
            aria-atomic="true"
            data-testid="live-result-alpha"
          >
            alpha = {formatCount(alpha)}
          </span>
          .
        </p>
      )}

      {/* Line 3 — total = mu + alpha */}
      <p>
        Therefore total = mu + alpha ={' '}
        <span
          className="font-mono font-semibold text-coral"
          aria-live="polite"
          aria-atomic="true"
          data-testid="live-result-total"
        >
          {formatCount(total)}
        </span>
        .
      </p>

      {/* Line 4 — reduction vs. dense direct baseline */}
      <p>
        Compared with the dense direct baseline{' '}
        <span
          className="font-mono font-semibold text-gray-900"
          data-testid="live-result-dense"
        >
          {formatCount(denseBaseline)}
        </span>
        , this is a{' '}
        <span
          className="font-mono font-semibold"
          style={{ color: 'var(--explorer-color-quantity, currentColor)' }}
          aria-live="polite"
          aria-atomic="true"
          data-testid="live-result-reduction"
        >
          {reduction}
        </span>
        {' '}reduction.
      </p>
    </div>
  );
}
