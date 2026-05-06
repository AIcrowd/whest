/**
 * CertificationCard — V3.1 §21 (C21)
 *
 * Compact standalone witness card for an accepted (σ, π) pair admitted
 * into G_pt. Sits next to the IncidenceMatrix walk in the σ-loop section
 * and surfaces the five labeled fields the V3.1 spec calls out:
 *
 *   candidate row move sigma:        [cycle notation on operand-axis slots]
 *   matching label relabeling pi:    [cycle notation on labels]
 *   domain check:                    passed
 *   incidence recovery:              π(σ(M)) = M
 *   result:                          accepted into G_pt
 *
 * Interactions:
 *   • Hover the σ row → onHoverSigma fires with the moved-row Set so the
 *     parent can highlight those rows in the matrix walk.
 *   • Hover the π row → onHoverPi fires with the moved-label Set so the
 *     parent can highlight those columns in the matrix walk.
 *   • "Show in matrix" button → onScrollToMatrix scrolls the matrix area
 *     into view (parent owns the ref / scrollIntoView call).
 *   • "Read Appendix A" link → href="#appendix-section-1" (Appendix A is
 *     the product-side certification body, mounted from
 *     content/appendix/section1.ts).
 *
 * Reject case: when `pair.kind === 'rejected'` the card returns null —
 * the existing in-line failure visualization (recovery-badge-below
 * recovery-fail) inside SigmaLoop already covers that branch.
 *
 * Accessibility:
 *   • Each labeled field is keyboard-focusable (tabIndex=0) and carries
 *     an aria-label that reads the field's own name + value.
 *   • The card itself has role="region" + aria-label so screen readers
 *     announce it as the "Certification card".
 *   • All hover handlers are also wired to focus / blur so keyboard
 *     navigation drives the same matrix highlight.
 *
 * Token discipline:
 *   • Colours are sourced exclusively from CSS design-system tokens
 *     (var(--coral), var(--success), var(--gray-*)) — no raw notation
 *     hex literals appear in this source (the notation-system audit
 *     forbids them).
 *
 * No engine touches — the card is pure presentation over the witness
 * object that SigmaLoop already builds.
 */

import { useMemo } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors UnavailableDetailsPanel / NaiveAlphaCostMeter)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:      'var(--coral)',          // primary witness accent
  coralBg:    'color-mix(in srgb, var(--coral) 8%, transparent)',
  coralBorder:'color-mix(in srgb, var(--coral) 32%, transparent)',
  success:    'var(--success)',        // "passed" / "accepted" lane
  successBg:  'color-mix(in srgb, var(--success) 12%, transparent)',
  gray900:    'var(--gray-900)',
  gray700:    'var(--gray-700)',
  gray600:    'var(--gray-600)',
  gray500:    'var(--gray-500)',
  gray400:    'var(--gray-400)',
  gray200:    'var(--gray-200)',
  gray100:    'var(--gray-100)',
  gray50:     'var(--gray-50)',
  white:      'var(--white)',
};

/* ─────────────────────────────────────────────────────────────────────────────
   Cycle-notation formatting helpers
   ─────────────────────────────────────────────────────────────────────────────
   These mirror the σ/π formatters that SigmaLoop already uses for the chip
   buttons so the card and the chip read identically. Duplicating them here
   keeps the card self-contained — no dependency on SigmaLoop internals.
   ───────────────────────────────────────────────────────────────────────────── */
function fmtSigmaCycles(sigmaRowPerm, uLabels) {
  if (!sigmaRowPerm || sigmaRowPerm.length === 0) return 'e';
  const n = sigmaRowPerm.length;
  const visited = new Set();
  const cycles = [];
  for (let i = 0; i < n; i++) {
    if (visited.has(i) || sigmaRowPerm[i] === i) continue;
    const cycle = [];
    let cur = i;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(uLabels?.[cur] ?? `r${cur}`);
      cur = sigmaRowPerm[cur];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }
  return cycles.map((c) => '(' + c.join(' ') + ')').join('') || 'e';
}

function fmtPiCycles(pi) {
  if (!pi) return 'e';
  const entries = Object.entries(pi).filter(([k, v]) => k !== v);
  if (entries.length === 0) return 'e';
  const visited = new Set();
  const cycles = [];
  for (const [k] of entries) {
    if (visited.has(k)) continue;
    const cycle = [];
    let cur = k;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = pi[cur];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }
  return cycles.map((c) => '(' + c.join(' ') + ')').join('') || 'e';
}

/* ─────────────────────────────────────────────────────────────────────────────
   Hover-set extractors
   ─────────────────────────────────────────────────────────────────────────────
   The card emits the same shape that IncidenceMatrix consumes in its
   movedRows / movedCols props — a Set of indices for rows and a Set of
   labels for columns — so the parent can wire the bus directly.
   ───────────────────────────────────────────────────────────────────────────── */
function buildSigmaRowSet(sigmaRowPerm) {
  const set = new Set();
  if (!sigmaRowPerm) return set;
  sigmaRowPerm.forEach((uIdx, k) => {
    if (uIdx !== k) set.add(k);
  });
  return set;
}

function buildPiLabelSet(pi) {
  const set = new Set();
  if (!pi) return set;
  for (const [from, to] of Object.entries(pi)) {
    if (from !== to) set.add(from);
  }
  return set;
}

/* ─────────────────────────────────────────────────────────────────────────────
   Field row — labelled value with hover/focus handlers.

   The simpler redesign (in response to user feedback "not sure how to read
   the certification card") replaces the prior label-on-left/value-on-right
   grid with a vertical pairing: small kicker label above, generous value
   below. Within a card that's 320 px wide this reads as a stack of clearly
   delimited proof-receipt items rather than a 5-row table.
   ───────────────────────────────────────────────────────────────────────────── */
function FieldRow({
  testId,
  fieldLabel,
  fieldValue,
  ariaLabel,
  onMouseEnter,
  onMouseLeave,
  onFocus,
  onBlur,
  valueColor,
  valueIcon = null,
  highlightOnHover = false,
}) {
  return (
    <div
      data-testid={testId}
      tabIndex={0}
      role="group"
      aria-label={ariaLabel}
      className="cert-card-row"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onFocus={onFocus}
      onBlur={onBlur}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        padding: '5px 4px',
        borderRadius: 4,
        cursor: highlightOnHover ? 'pointer' : 'default',
        outline: 'none',
      }}
    >
      <span
        className="cert-card-label"
        style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: TOKEN.gray500,
        }}
      >
        {fieldLabel}
      </span>
      <span
        className="cert-card-value"
        style={{
          display: 'inline-flex',
          alignItems: 'baseline',
          gap: 6,
          fontFamily: 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace)',
          fontSize: 13,
          fontWeight: 600,
          color: valueColor || TOKEN.gray900,
          wordBreak: 'break-word',
        }}
      >
        {valueIcon ? (
          <span aria-hidden="true" style={{ flexShrink: 0, fontSize: 12 }}>{valueIcon}</span>
        ) : null}
        <span>{fieldValue}</span>
      </span>
    </div>
  );
}

/* Small section kicker that visually groups related field rows
   ("witness inputs" vs. "verification proof") within the card. */
function CardSectionKicker({ children }) {
  return (
    <div
      className="cert-card-section-kicker"
      style={{
        marginTop: 8,
        marginBottom: 2,
        fontSize: 9.5,
        fontWeight: 700,
        letterSpacing: '0.16em',
        textTransform: 'uppercase',
        color: TOKEN.gray400,
        borderTop: `1px dashed ${TOKEN.gray200}`,
        paddingTop: 6,
      }}
    >
      {children}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * CertificationCard
 *
 * Props:
 *   pair           {object}   — the witness object SigmaLoop already builds.
 *                                Expected shape:
 *                                  {
 *                                    isValid       : boolean,
 *                                    sigmaRowPerm  : number[],
 *                                    pi            : { [label]: label },
 *                                    kind?         : 'rejected' | string,
 *                                  }
 *   uLabels        {string[]} — operand-axis slot labels (for σ cycle text)
 *   onHoverSigma   {((set: Set<number>|null) => void)|null}
 *                              — fires with moved-row Set on hover/focus,
 *                                null on leave/blur.
 *   onHoverPi      {((set: Set<string>|null) => void)|null}
 *                              — fires with moved-label Set on hover/focus,
 *                                null on leave/blur.
 *   onScrollToMatrix {(() => void)|null}
 *                              — fires when the user clicks "Show in matrix".
 */
export default function CertificationCard({
  pair = null,
  uLabels = [],
  onHoverSigma = null,
  onHoverPi = null,
  onScrollToMatrix = null,
}) {
  // Reject case: render nothing — the existing in-line failure visualization
  // already handles rejected pairs.
  if (!pair) return null;
  if (pair.kind === 'rejected') return null;
  if (pair.isValid === false) return null;

  const sigmaCycles = useMemo(
    () => fmtSigmaCycles(pair.sigmaRowPerm, uLabels),
    [pair.sigmaRowPerm, uLabels],
  );
  const piCycles = useMemo(() => fmtPiCycles(pair.pi), [pair.pi]);

  const sigmaRowSet = useMemo(
    () => buildSigmaRowSet(pair.sigmaRowPerm),
    [pair.sigmaRowPerm],
  );
  const piLabelSet = useMemo(() => buildPiLabelSet(pair.pi), [pair.pi]);

  const fireHoverSigma = (set) => {
    if (typeof onHoverSigma === 'function') onHoverSigma(set);
  };
  const fireHoverPi = (set) => {
    if (typeof onHoverPi === 'function') onHoverPi(set);
  };

  return (
    <section
      role="region"
      aria-label="Certification card — accepted (σ, π) witness"
      data-testid="certification-card"
      data-pair-status="accepted"
      className="certification-card"
      style={{
        // Fill the parent column (was hard-coded 320 px which left dead
        // space on either side of the card inside the σ-Loop column).
        width: '100%',
        boxSizing: 'border-box',
        backgroundColor: TOKEN.white,
        border: `1px solid ${TOKEN.coralBorder}`,
        borderLeft: `3px solid ${TOKEN.coral}`,
        borderRadius: 8,
        padding: '12px 12px 10px',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)',
      }}
    >
      {/* Heading */}
      <header
        className="cert-card-header"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 6,
        }}
      >
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: 999,
            backgroundColor: TOKEN.coral,
          }}
        />
        <h5
          style={{
            margin: 0,
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: TOKEN.gray900,
          }}
        >
          Certification card
        </h5>
      </header>

      {/* WITNESS — the inputs to the certification test (σ, π).
          Hover handlers here drive the matrix highlight bus. */}
      <CardSectionKicker>Witness</CardSectionKicker>
      <FieldRow
        testId="cert-field-sigma"
        fieldLabel="candidate row move sigma"
        fieldValue={sigmaCycles}
        ariaLabel={`Candidate row move sigma: ${sigmaCycles}`}
        valueColor={TOKEN.coral}
        highlightOnHover={true}
        onMouseEnter={() => fireHoverSigma(sigmaRowSet)}
        onMouseLeave={() => fireHoverSigma(null)}
        onFocus={() => fireHoverSigma(sigmaRowSet)}
        onBlur={() => fireHoverSigma(null)}
      />
      <FieldRow
        testId="cert-field-pi"
        fieldLabel="matching label relabeling pi"
        fieldValue={piCycles}
        ariaLabel={`Matching label relabeling pi: ${piCycles}`}
        valueColor={TOKEN.coral}
        highlightOnHover={true}
        onMouseEnter={() => fireHoverPi(piLabelSet)}
        onMouseLeave={() => fireHoverPi(null)}
        onFocus={() => fireHoverPi(piLabelSet)}
        onBlur={() => fireHoverPi(null)}
      />

      {/* PROOF — the two checks the certifier ran. Each carries a ✓ icon
          so the verdict reads as a checklist instead of a data table. */}
      <CardSectionKicker>Proof</CardSectionKicker>
      <FieldRow
        testId="cert-field-domain"
        fieldLabel="domain check"
        fieldValue="passed"
        ariaLabel="Domain check: passed"
        valueColor={TOKEN.success}
        valueIcon="✓"
      />
      <FieldRow
        testId="cert-field-incidence"
        fieldLabel="incidence recovery"
        fieldValue="π(σ(M)) = M"
        ariaLabel="Incidence recovery: π(σ(M)) = M"
        valueColor={TOKEN.success}
        valueIcon="✓"
      />

      {/* RESULT — the outcome. `→` icon points at the conclusion. */}
      <CardSectionKicker>Result</CardSectionKicker>
      <FieldRow
        testId="cert-field-result"
        fieldLabel="result"
        fieldValue="accepted into G_pt"
        ariaLabel="Result: accepted into G_pt"
        valueColor={TOKEN.success}
        valueIcon="→"
      />

      {/* CTA row */}
      <div
        className="cert-card-cta-row"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: 10,
          marginTop: 8,
          paddingTop: 8,
          borderTop: `1px solid ${TOKEN.gray200}`,
        }}
      >
        <button
          type="button"
          onClick={() => {
            if (typeof onScrollToMatrix === 'function') onScrollToMatrix();
          }}
          aria-label="Show this witness in the incidence matrix above"
          data-testid="cert-show-in-matrix-btn"
          className="cert-card-show-in-matrix"
          style={{
            backgroundColor: TOKEN.white,
            border: `1px solid ${TOKEN.gray400}`,
            borderRadius: 6,
            padding: '4px 10px',
            fontSize: 11.5,
            fontWeight: 600,
            color: TOKEN.gray900,
            cursor: 'pointer',
          }}
        >
          Show in matrix
        </button>
        <a
          href="#appendix-section-1"
          aria-label="Read Appendix A — pointwise group certification"
          data-testid="cert-appendix-link"
          className="cert-card-appendix-link"
          style={{
            fontSize: 11.5,
            fontWeight: 600,
            color: TOKEN.gray700,
            textDecoration: 'underline',
            textDecorationStyle: 'dotted',
            textUnderlineOffset: 3,
          }}
        >
          Read Appendix A →
        </a>
      </div>
    </section>
  );
}
