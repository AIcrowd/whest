/**
 * WitnessGallery — V3.1 §25 (C25)
 *
 * Side-by-side accepted vs rejected witness comparison gallery. Lives above
 * the existing single-pair animation panel inside SigmaLoop and surfaces one
 * representative accepted (σ, π) pair next to one representative rejected σ
 * so the reader can see — at one glance — what the certifier accepts and what
 * it rejects, and why.
 *
 * Card layout (V3.1 §25 verbatim field labels):
 *
 *   ── Accepted card (5 fields) ────────────────
 *     candidate                : cyclic rotation
 *     sigma row move           : [cycle on operand-axis slots]
 *     pi label relabeling      : [cycle on labels]
 *     result                   : accepted
 *     group contribution       : generator of <group>
 *
 *   ── Rejected card (5 fields) ────────────────
 *     candidate                : reflection
 *     sigma row move           : [cycle on operand-axis slots]
 *     attempted pi             : none compatible
 *     reason                   : incidence fingerprint mismatch
 *     result                   : rejected
 *
 * Interactions:
 *   • Hovering a card's sigma row → fires onHoverSigma with the moved-row Set
 *     so the parent can highlight those rows in the matrix walk.
 *   • Hovering a card's pi row (accepted only) → fires onHoverPi with the
 *     moved-label Set so the parent can highlight those columns.
 *   • Click "why rejected?" on the rejected card → expands the diagnostic
 *     detail inline (incidence fingerprint mismatch, mismatched columns).
 *   • If the current preset has neither an accepted nor a rejected pair
 *     (e.g. trivial group) → render a "Switch to Directed triangle" CTA that
 *     calls onSwitchToDirectedTriangle.
 *
 * Accessibility:
 *   • The gallery is a <section role="region" aria-label="Witness gallery">.
 *   • Each card is its own role="region" with an aria-label naming the verdict.
 *   • Every labeled field is keyboard-focusable (tabIndex=0).
 *   • The "why rejected?" button is a real <button> with aria-expanded.
 *   • The CTA is a real <button> labelled "Switch to Directed triangle".
 *
 * Token discipline:
 *   • All colours come from CSS design-system tokens (var(--coral),
 *     var(--success), var(--gray-*)) — no raw notation hex literals appear
 *     in this source.
 *
 * No engine touches — the gallery is pure presentation over the witness
 * objects SigmaLoop already builds.
 */

import { useMemo, useState } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors CertificationCard)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:        'var(--coral)',
  coralBg:      'color-mix(in srgb, var(--coral) 8%, transparent)',
  coralBorder:  'color-mix(in srgb, var(--coral) 32%, transparent)',
  success:      'var(--success)',
  successBg:    'color-mix(in srgb, var(--success) 12%, transparent)',
  warning:      'var(--warning)',
  warningBg:    'color-mix(in srgb, var(--warning) 12%, transparent)',
  warningBorder:'color-mix(in srgb, var(--warning) 32%, transparent)',
  gray900:      'var(--gray-900)',
  gray700:      'var(--gray-700)',
  gray600:      'var(--gray-600)',
  gray500:      'var(--gray-500)',
  gray400:      'var(--gray-400)',
  gray200:      'var(--gray-200)',
  gray100:      'var(--gray-100)',
  gray50:       'var(--gray-50)',
  white:        'var(--white)',
};

/* ─────────────────────────────────────────────────────────────────────────────
   Cycle-notation formatting helpers (mirrors CertificationCard / SigmaLoop)
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
   Candidate-flavour heuristics — readable English label for the σ shape.
   Matches the V3.1 §25 examples ("cyclic rotation", "reflection") but falls
   back to neutral phrasing for shapes that do not match the canonical
   triangle preset.
   ───────────────────────────────────────────────────────────────────────────── */
function describeCandidate(sigmaRowPerm) {
  if (!sigmaRowPerm || sigmaRowPerm.length === 0) return 'identity';
  const n = sigmaRowPerm.length;
  // Count cycles of length > 1
  const visited = new Set();
  const cycleLengths = [];
  for (let i = 0; i < n; i++) {
    if (visited.has(i)) continue;
    let cur = i;
    let len = 0;
    while (!visited.has(cur)) {
      visited.add(cur);
      cur = sigmaRowPerm[cur];
      len += 1;
    }
    if (len > 1) cycleLengths.push(len);
  }
  if (cycleLengths.length === 0) return 'identity';
  // Single n-cycle that touches every row → cyclic rotation
  if (cycleLengths.length === 1 && cycleLengths[0] === n) return 'cyclic rotation';
  // Single 2-cycle → reflection / transposition
  if (cycleLengths.length === 1 && cycleLengths[0] === 2) return 'reflection';
  // Multiple disjoint 2-cycles → reflection composite
  if (cycleLengths.every((l) => l === 2)) return 'reflection composite';
  // Mixed → generic permutation
  return 'permutation';
}

/* ─────────────────────────────────────────────────────────────────────────────
   Group contribution helper — short readable phrase for the accepted card's
   "group contribution" field. Uses the group display name when available.
   ───────────────────────────────────────────────────────────────────────────── */
function describeGroupContribution(group) {
  const name = group?.displayName || group?.name || group?.label;
  if (typeof name === 'string' && name.length > 0) {
    return `generator of ${name}`;
  }
  return 'generator of G_pt';
}

/* ─────────────────────────────────────────────────────────────────────────────
   Reason display + diagnostic detail helpers
   ─────────────────────────────────────────────────────────────────────────────
   The engine stores reject reasons as raw strings (e.g.
   "No matching π (fingerprint mismatch)"). The V3.1 §25 spec asks for a more
   human "incidence fingerprint mismatch" phrasing on the headline field with
   the original sentence revealed inside the inline diagnostic.
   ───────────────────────────────────────────────────────────────────────────── */
function shortReason(rawReason) {
  if (!rawReason) return 'no compatible π found';
  if (/fingerprint/i.test(rawReason)) return 'incidence fingerprint mismatch';
  if (/domain/i.test(rawReason))      return 'domain mismatch';
  return rawReason;
}

/* ─────────────────────────────────────────────────────────────────────────────
   FieldRow — labelled value with optional hover/focus highlight handlers.
   ───────────────────────────────────────────────────────────────────────────── */
function FieldRow({
  testId,
  fieldLabel,
  fieldValue,
  ariaLabel,
  valueColor,
  onMouseEnter,
  onMouseLeave,
  onFocus,
  onBlur,
  highlightOnHover = false,
}) {
  return (
    <div
      data-testid={testId}
      tabIndex={0}
      role="group"
      aria-label={ariaLabel}
      className="witness-gallery-row"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onFocus={onFocus}
      onBlur={onBlur}
      style={{
        display: 'grid',
        gridTemplateColumns: '160px 1fr',
        columnGap: 10,
        alignItems: 'baseline',
        padding: '4px 6px',
        borderRadius: 4,
        cursor: highlightOnHover ? 'pointer' : 'default',
        outline: 'none',
      }}
    >
      <span
        className="witness-gallery-row-label"
        style={{
          fontSize: 10.5,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: TOKEN.gray500,
        }}
      >
        {fieldLabel}
      </span>
      <span
        className="witness-gallery-row-value"
        style={{
          fontFamily: 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace)',
          fontSize: 12.5,
          fontWeight: 600,
          color: valueColor || TOKEN.gray900,
          wordBreak: 'break-word',
        }}
      >
        {fieldValue}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Accepted card — 5 fields per V3.1 §25
   ───────────────────────────────────────────────────────────────────────────── */
function AcceptedCard({ pair, uLabels, group, onHoverSigma, onHoverPi }) {
  const sigmaCycles = useMemo(() => fmtSigmaCycles(pair.sigmaRowPerm, uLabels), [pair.sigmaRowPerm, uLabels]);
  const piCycles = useMemo(() => fmtPiCycles(pair.pi), [pair.pi]);
  const sigmaRowSet = useMemo(() => buildSigmaRowSet(pair.sigmaRowPerm), [pair.sigmaRowPerm]);
  const piLabelSet = useMemo(() => buildPiLabelSet(pair.pi), [pair.pi]);
  const candidateLabel = useMemo(() => describeCandidate(pair.sigmaRowPerm), [pair.sigmaRowPerm]);
  const groupContribution = useMemo(() => describeGroupContribution(group), [group]);

  const fireSigma = (set) => { if (typeof onHoverSigma === 'function') onHoverSigma(set); };
  const firePi = (set) => { if (typeof onHoverPi === 'function') onHoverPi(set); };

  return (
    <section
      role="region"
      aria-label="Accepted witness card"
      data-testid="witness-gallery-accepted-card"
      data-pair-status="accepted"
      className="witness-gallery-card witness-gallery-card-accepted"
      style={{
        flex: '1 1 280px',
        minWidth: 280,
        backgroundColor: TOKEN.white,
        border: `1px solid ${TOKEN.coralBorder}`,
        borderLeft: `3px solid ${TOKEN.success}`,
        borderRadius: 8,
        padding: '12px 12px 10px',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)',
      }}
    >
      <header
        className="witness-gallery-card-header"
        style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}
      >
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: 999,
            backgroundColor: TOKEN.success,
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
          Accepted witness
        </h5>
      </header>

      <FieldRow
        testId="witness-gallery-accepted-candidate"
        fieldLabel="candidate"
        fieldValue={candidateLabel}
        ariaLabel={`Candidate: ${candidateLabel}`}
        valueColor={TOKEN.gray900}
      />
      <FieldRow
        testId="witness-gallery-accepted-sigma"
        fieldLabel="sigma row move"
        fieldValue={sigmaCycles}
        ariaLabel={`Sigma row move: ${sigmaCycles}`}
        valueColor={TOKEN.coral}
        highlightOnHover={true}
        onMouseEnter={() => fireSigma(sigmaRowSet)}
        onMouseLeave={() => fireSigma(null)}
        onFocus={() => fireSigma(sigmaRowSet)}
        onBlur={() => fireSigma(null)}
      />
      <FieldRow
        testId="witness-gallery-accepted-pi"
        fieldLabel="pi label relabeling"
        fieldValue={piCycles}
        ariaLabel={`Pi label relabeling: ${piCycles}`}
        valueColor={TOKEN.coral}
        highlightOnHover={true}
        onMouseEnter={() => firePi(piLabelSet)}
        onMouseLeave={() => firePi(null)}
        onFocus={() => firePi(piLabelSet)}
        onBlur={() => firePi(null)}
      />
      <FieldRow
        testId="witness-gallery-accepted-result"
        fieldLabel="result"
        fieldValue="accepted"
        ariaLabel="Result: accepted"
        valueColor={TOKEN.success}
      />
      <FieldRow
        testId="witness-gallery-accepted-contribution"
        fieldLabel="group contribution"
        fieldValue={groupContribution}
        ariaLabel={`Group contribution: ${groupContribution}`}
        valueColor={TOKEN.gray900}
      />
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Rejected card — 5 fields per V3.1 §25 + inline "why rejected?" diagnostic
   ───────────────────────────────────────────────────────────────────────────── */
function RejectedCard({ pair, uLabels, onHoverSigma }) {
  const [showWhy, setShowWhy] = useState(false);

  const sigmaCycles = useMemo(() => fmtSigmaCycles(pair.sigmaRowPerm, uLabels), [pair.sigmaRowPerm, uLabels]);
  const sigmaRowSet = useMemo(() => buildSigmaRowSet(pair.sigmaRowPerm), [pair.sigmaRowPerm]);
  const candidateLabel = useMemo(() => describeCandidate(pair.sigmaRowPerm), [pair.sigmaRowPerm]);
  const reasonShort = useMemo(() => shortReason(pair.reason), [pair.reason]);

  const fireSigma = (set) => { if (typeof onHoverSigma === 'function') onHoverSigma(set); };

  return (
    <section
      role="region"
      aria-label="Rejected witness card"
      data-testid="witness-gallery-rejected-card"
      data-pair-status="rejected"
      className="witness-gallery-card witness-gallery-card-rejected"
      style={{
        flex: '1 1 280px',
        minWidth: 280,
        backgroundColor: TOKEN.white,
        border: `1px solid ${TOKEN.warningBorder}`,
        borderLeft: `3px solid ${TOKEN.warning}`,
        borderRadius: 8,
        padding: '12px 12px 10px',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)',
      }}
    >
      <header
        className="witness-gallery-card-header"
        style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}
      >
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: 999,
            backgroundColor: TOKEN.warning,
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
          Rejected witness
        </h5>
      </header>

      <FieldRow
        testId="witness-gallery-rejected-candidate"
        fieldLabel="candidate"
        fieldValue={candidateLabel}
        ariaLabel={`Candidate: ${candidateLabel}`}
        valueColor={TOKEN.gray900}
      />
      <FieldRow
        testId="witness-gallery-rejected-sigma"
        fieldLabel="sigma row move"
        fieldValue={sigmaCycles}
        ariaLabel={`Sigma row move: ${sigmaCycles}`}
        valueColor={TOKEN.warning}
        highlightOnHover={true}
        onMouseEnter={() => fireSigma(sigmaRowSet)}
        onMouseLeave={() => fireSigma(null)}
        onFocus={() => fireSigma(sigmaRowSet)}
        onBlur={() => fireSigma(null)}
      />
      <FieldRow
        testId="witness-gallery-rejected-attempted-pi"
        fieldLabel="attempted pi"
        fieldValue="none compatible"
        ariaLabel="Attempted pi: none compatible"
        valueColor={TOKEN.gray700}
      />
      <FieldRow
        testId="witness-gallery-rejected-reason"
        fieldLabel="reason"
        fieldValue={reasonShort}
        ariaLabel={`Reason: ${reasonShort}`}
        valueColor={TOKEN.warning}
      />
      <FieldRow
        testId="witness-gallery-rejected-result"
        fieldLabel="result"
        fieldValue="rejected"
        ariaLabel="Result: rejected"
        valueColor={TOKEN.warning}
      />

      {/* "why rejected?" inline diagnostic ── */}
      <div
        className="witness-gallery-card-cta-row"
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
          onClick={() => setShowWhy((v) => !v)}
          aria-expanded={showWhy}
          aria-controls="witness-gallery-rejected-diagnostic"
          aria-label="Show diagnostic detail explaining why this candidate was rejected"
          data-testid="witness-gallery-why-rejected-btn"
          className="witness-gallery-why-rejected"
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
          {showWhy ? '▾ why rejected?' : '▸ why rejected?'}
        </button>
      </div>

      {showWhy && (
        <div
          id="witness-gallery-rejected-diagnostic"
          data-testid="witness-gallery-rejected-diagnostic"
          role="note"
          aria-label="Detailed rejection diagnostic"
          className="witness-gallery-rejected-diagnostic"
          style={{
            marginTop: 8,
            padding: '8px 10px',
            borderRadius: 6,
            backgroundColor: TOKEN.warningBg,
            border: `1px solid ${TOKEN.warningBorder}`,
            fontSize: 11.5,
            lineHeight: 1.5,
            color: TOKEN.gray900,
          }}
        >
          <div
            style={{
              fontSize: 10.5,
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: TOKEN.gray700,
              marginBottom: 4,
            }}
          >
            diagnostic detail
          </div>
          <div>
            {pair.reason || 'No matching π (fingerprint mismatch)'}
          </div>
          <div style={{ marginTop: 4, fontSize: 11, color: TOKEN.gray700 }}>
            σ rearranged the rows of M, but no relabeling π over the active labels
            could realign the resulting columns to recover M. This row move therefore
            does not lift to a pointwise symmetry of the contraction.
          </div>
        </div>
      )}
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Pair selection — pick one accepted + one rejected from SigmaLoop's pairs.
   ─────────────────────────────────────────────────────────────────────────────
   We exclude identity / skipped pairs from the accepted side so the gallery
   surfaces something interesting, and prefer the first non-skipped accepted
   pair (which mirrors how SigmaLoop already selects an initial valid pair).
   ───────────────────────────────────────────────────────────────────────────── */
function pickWitnessPair(pairs, predicate) {
  if (!Array.isArray(pairs)) return null;
  for (const p of pairs) {
    if (p && !p.skipped && predicate(p)) return p;
  }
  return null;
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * WitnessGallery
 *
 * Props:
 *   pairs           {object[]}  — full pair list as built by SigmaLoop
 *                                 (each entry has at minimum: skipped, isValid,
 *                                 sigmaRowPerm, pi, reason).
 *   uLabels         {string[]}  — operand-axis slot labels (for σ cycle text)
 *   group           {object}    — symmetry group object (for "group contribution")
 *   onHoverSigma    {((set: Set<number>|null) => void)|null}
 *                               — fires with moved-row Set on hover/focus,
 *                                 null on leave/blur.
 *   onHoverPi       {((set: Set<string>|null) => void)|null}
 *                               — fires with moved-label Set on hover/focus.
 *   onSwitchToDirectedTriangle  {(() => void)|null}
 *                               — fires when the user clicks the CTA shown
 *                                 when neither an accepted nor rejected pair
 *                                 exists in the current preset.
 */
export default function WitnessGallery({
  pairs = [],
  uLabels = [],
  group = null,
  onHoverSigma = null,
  onHoverPi = null,
  onSwitchToDirectedTriangle = null,
}) {
  const acceptedPair = useMemo(
    () => pickWitnessPair(pairs, (p) => p.isValid === true && p.pi),
    [pairs],
  );
  const rejectedPair = useMemo(
    () => pickWitnessPair(pairs, (p) => p.isValid === false),
    [pairs],
  );

  // Empty-state CTA: no accepted AND no rejected → suggest the canonical
  // Directed triangle preset which has a clean accepted/rejected pair.
  if (!acceptedPair && !rejectedPair) {
    return (
      <section
        role="region"
        aria-label="Witness gallery"
        data-testid="witness-gallery"
        data-state="empty"
        className="witness-gallery witness-gallery-empty"
        style={{
          padding: '12px 14px',
          borderRadius: 8,
          border: `1px dashed ${TOKEN.gray400}`,
          backgroundColor: TOKEN.gray50,
          color: TOKEN.gray700,
          fontSize: 12.5,
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <span>
          The current preset has no rejected or accepted (σ, π) pair to compare.
        </span>
        <button
          type="button"
          onClick={() => {
            if (typeof onSwitchToDirectedTriangle === 'function') onSwitchToDirectedTriangle();
          }}
          aria-label="Switch to the Directed triangle preset to see an accepted vs rejected witness comparison"
          data-testid="witness-gallery-switch-preset-btn"
          className="witness-gallery-switch-preset"
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
          Switch to Directed triangle
        </button>
      </section>
    );
  }

  return (
    <section
      role="region"
      aria-label="Witness gallery — accepted vs rejected (σ, π) candidates"
      data-testid="witness-gallery"
      data-state={acceptedPair && rejectedPair ? 'paired' : (acceptedPair ? 'accepted-only' : 'rejected-only')}
      className="witness-gallery"
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 12,
        alignItems: 'stretch',
      }}
    >
      {acceptedPair && (
        <AcceptedCard
          pair={acceptedPair}
          uLabels={uLabels}
          group={group}
          onHoverSigma={onHoverSigma}
          onHoverPi={onHoverPi}
        />
      )}
      {rejectedPair && (
        <RejectedCard
          pair={rejectedPair}
          uLabels={uLabels}
          onHoverSigma={onHoverSigma}
        />
      )}

      {/* When only one side is present, expose the CTA so the reader can
          jump to a preset that has both. */}
      {(!acceptedPair || !rejectedPair) && (
        <div
          className="witness-gallery-cta-strip"
          style={{
            flex: '0 0 auto',
            display: 'flex',
            alignItems: 'center',
            paddingLeft: 4,
          }}
        >
          <button
            type="button"
            onClick={() => {
              if (typeof onSwitchToDirectedTriangle === 'function') onSwitchToDirectedTriangle();
            }}
            aria-label="Switch to the Directed triangle preset to see both an accepted and a rejected witness side-by-side"
            data-testid="witness-gallery-switch-preset-btn"
            className="witness-gallery-switch-preset"
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
            Switch to Directed triangle
          </button>
        </div>
      )}
    </section>
  );
}
