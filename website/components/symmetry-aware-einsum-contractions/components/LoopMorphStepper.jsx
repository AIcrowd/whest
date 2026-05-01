/**
 * LoopMorphStepper — V3.1 §7 — C07 (NEW)
 *
 * Side-by-side pseudocode morph from the dense loop to the representative loop,
 * driven by a 5-step stepper that the reader advances with Prev / Next or the
 * keyboard arrow keys. Each step focuses a different transformation:
 *
 *   1. dense assignments   — every full assignment touched
 *   2. group equal products — orbits revealed beneath the assignments
 *   3. choose representative — one representative per orbit selected
 *   4. project orbit members — each member projected onto its output rep
 *   5. accumulate into output — the representative loop in its final form
 *
 *   Left side (dense)
 *     for full_assignment in X:
 *         R[project(full_assignment)] += product_at(full_assignment)
 *
 *   Right side (representative)
 *     for rep in RepSet:
 *         base_val = product_at(rep)
 *         for out_rep in Outs(rep):
 *             R[out_rep] += coeff(rep, out_rep) * base_val
 *
 * Hover interactions — three "tokens" on the right side (RepSet, Outs(rep),
 * coeff) light up the visual columns in a small companion strip on hover or
 * keyboard focus, so the reader can connect identifier-to-object directly.
 *
 * Reduced-motion: morph transition collapses to instant; opacity/transition
 * styles fall back to `transition: 'none'`.
 *
 * All colours come from the design-system token map. No raw hex.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors colors_and_type.css tier 1 + 3A)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:      'var(--coral)',        // #F0524D
  coralLight: 'var(--coral-light)',  // #FEF2F1
  gray900:    'var(--gray-900)',     // #292C2D
  gray700:    'var(--gray-700)',     // #3F4142
  gray600:    'var(--gray-600)',     // #5D5F60
  gray500:    'var(--gray-500)',     // #888B8D
  gray400:    'var(--gray-400)',     // #AAACAD
  gray300:    'var(--gray-300)',     // #C4C7C8
  gray200:    'var(--gray-200)',     // #D9DCDC
  gray100:    'var(--gray-100)',     // #F1F3F5
  gray50:     'var(--gray-50)',      // #F8F9F9
  white:      'var(--white)',        // #FFFFFF
  einV:       'var(--ein-v)',        // #4A7CFF (free / visible)
  einW:       'var(--ein-w)',        // #64748B (summed)
};

/* ─────────────────────────────────────────────────────────────────────────────
   Pseudocode strings — referenced verbatim by the V3.1 §7 spec
   ───────────────────────────────────────────────────────────────────────────── */
const DENSE_PSEUDOCODE = [
  'for full_assignment in X:',
  '    R[project(full_assignment)] += product_at(full_assignment)',
];

const REP_PSEUDOCODE = [
  'for rep in RepSet:',
  '    base_val = product_at(rep)',
  '    for out_rep in Outs(rep):',
  '        R[out_rep] += coeff(rep, out_rep) * base_val',
];

/* ─────────────────────────────────────────────────────────────────────────────
   The five stepper stages
   ───────────────────────────────────────────────────────────────────────────── */
const STEPS = [
  {
    id: 'dense',
    title: 'dense assignments',
    caption:
      'Every full assignment in X is touched once; the dense loop multiplies and accumulates per assignment.',
    densePulse: 0,
    repPulse: 0,
    leftWeight: 1,
    rightWeight: 0.35,
  },
  {
    id: 'group',
    title: 'group equal products',
    caption:
      'Assignments that share a product value are gathered into orbits — equal products under the symmetry group.',
    densePulse: 1,
    repPulse: 0,
    leftWeight: 1,
    rightWeight: 0.5,
  },
  {
    id: 'choose',
    title: 'choose representative',
    caption:
      'One representative is chosen per orbit; product_at(rep) computes the shared value once.',
    densePulse: 1,
    repPulse: 1,
    leftWeight: 0.85,
    rightWeight: 0.7,
  },
  {
    id: 'project',
    title: 'project orbit members',
    caption:
      'Each orbit member projects onto an output representative; coeff(rep, out_rep) records the multiplicity.',
    densePulse: 0.5,
    repPulse: 2,
    leftWeight: 0.5,
    rightWeight: 0.9,
  },
  {
    id: 'accumulate',
    title: 'accumulate into output',
    caption:
      'R[out_rep] receives coeff(rep, out_rep) · base_val once per output rep — multiply once, accumulate many.',
    densePulse: 0,
    repPulse: 3,
    leftWeight: 0.35,
    rightWeight: 1,
  },
];

const N_STEPS = STEPS.length; // 5

/* ─────────────────────────────────────────────────────────────────────────────
   Hover-bus tokens — match the right-side identifiers
   ───────────────────────────────────────────────────────────────────────────── */
const HOVER_TOKENS = {
  RepSet:    { label: 'RepSet',    description: 'product orbits — one rep per equal-product class' },
  Outs:      { label: 'Outs(rep)', description: 'output representatives reached by an orbit' },
  coeff:     { label: 'coeff',     description: 'multiplicity of orbit members landing in an output rep' },
};

/* ─────────────────────────────────────────────────────────────────────────────
   Reduced-motion hook — mirrors the pattern in TwoQuotientSchematic
   ───────────────────────────────────────────────────────────────────────────── */
function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e) => setReduced(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return reduced;
}

/* ─────────────────────────────────────────────────────────────────────────────
   PseudocodeBlock — renders a list of code lines with optional highlight on
   each line. `pulseLine` (0-indexed) draws a coral marker bar to the left.
   ───────────────────────────────────────────────────────────────────────────── */
function PseudocodeBlock({
  title,
  subtitle,
  lines,
  pulseLine,
  weight,
  reducedMotion,
  onTokenHover,
  hoverTokenId,
  side,
}) {
  // Tokenization for the right side — only meaningful identifiers receive
  // interactive hover handlers.
  const renderLine = (line, idx) => {
    const isPulse = pulseLine !== null && pulseLine !== undefined && idx === pulseLine;
    const indent = line.match(/^\s*/)?.[0] ?? '';

    // Build either a plain code line or a tokenized code line with hoverable
    // identifiers. Only the right side has hoverable tokens.
    let body;
    if (side === 'rep') {
      body = (
        <RepLineWithHoverTokens
          line={line}
          onTokenHover={onTokenHover}
          hoverTokenId={hoverTokenId}
        />
      );
    } else {
      body = <span>{line || ' '}</span>;
    }

    return (
      <div
        key={idx}
        className="flex items-baseline"
        data-testid={`loop-morph-stepper-${side}-line`}
        data-line-idx={idx}
        data-pulsed={isPulse ? 'true' : 'false'}
        style={{
          opacity: weight,
          transition: reducedMotion ? 'none' : 'opacity 0.32s ease',
        }}
      >
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: 6,
            height: 18,
            marginRight: 8,
            borderRadius: 2,
            background: isPulse ? TOKEN.coral : 'transparent',
            transition: reducedMotion ? 'none' : 'background 0.2s ease',
          }}
        />
        <code
          className="whitespace-pre font-mono text-[12.5px] leading-7"
          style={{ color: isPulse ? TOKEN.gray900 : TOKEN.gray700 }}
        >
          <span aria-hidden="true">{indent.replace(/ /g, ' ')}</span>
          {body}
        </code>
      </div>
    );
  };

  return (
    <div
      data-testid={`loop-morph-stepper-${side}`}
      className="flex h-full flex-col rounded-md border"
      style={{
        borderColor: TOKEN.gray200,
        background: TOKEN.gray50,
      }}
    >
      <div
        className="border-b px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.08em]"
        style={{
          borderColor: TOKEN.gray200,
          color: side === 'dense' ? TOKEN.gray500 : TOKEN.coral,
          background: TOKEN.white,
        }}
      >
        <div>{title}</div>
        {subtitle ? (
          <div className="mt-0.5 font-mono text-[10px] font-normal normal-case tracking-normal" style={{ color: TOKEN.gray500 }}>
            {subtitle}
          </div>
        ) : null}
      </div>
      <div className="px-3 py-3">
        {lines.map((line, idx) => renderLine(line.replace(/^\s+/, (m) => m), idx))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   RepLineWithHoverTokens — splits a representative-side line into runs and
   wraps RepSet / Outs(rep) / coeff identifiers in interactive spans.
   ───────────────────────────────────────────────────────────────────────────── */
function RepLineWithHoverTokens({ line, onTokenHover, hoverTokenId }) {
  const trimmed = line.replace(/^\s+/, '');
  // Pattern matches the three identifiers — order matters (longest first to
  // avoid Outs being matched as part of a larger token).
  const pattern = /(RepSet|Outs\(rep\)|coeff)/g;
  const parts = [];
  let lastIdx = 0;
  let m;
  let key = 0;
  while ((m = pattern.exec(trimmed)) !== null) {
    if (m.index > lastIdx) {
      parts.push({ kind: 'plain', text: trimmed.slice(lastIdx, m.index), key: key++ });
    }
    const matched = m[0];
    let tokenId = null;
    if (matched === 'RepSet') tokenId = 'RepSet';
    else if (matched === 'Outs(rep)') tokenId = 'Outs';
    else if (matched === 'coeff') tokenId = 'coeff';
    parts.push({ kind: 'token', text: matched, tokenId, key: key++ });
    lastIdx = m.index + matched.length;
  }
  if (lastIdx < trimmed.length) {
    parts.push({ kind: 'plain', text: trimmed.slice(lastIdx), key: key++ });
  }

  return (
    <>
      {parts.map((p) => {
        if (p.kind === 'plain') {
          return <span key={p.key}>{p.text}</span>;
        }
        const isActive = hoverTokenId === p.tokenId;
        const tokenInfo = HOVER_TOKENS[p.tokenId];
        return (
          <span
            key={p.key}
            role="button"
            tabIndex={0}
            data-testid="loop-morph-stepper-token"
            data-token-id={p.tokenId}
            data-active={isActive ? 'true' : 'false'}
            aria-label={`Highlight ${tokenInfo?.label ?? p.text}: ${tokenInfo?.description ?? ''}`}
            aria-pressed={isActive}
            onMouseEnter={() => onTokenHover(p.tokenId)}
            onMouseLeave={() => onTokenHover(null)}
            onFocus={() => onTokenHover(p.tokenId)}
            onBlur={() => onTokenHover(null)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onTokenHover(isActive ? null : p.tokenId);
              }
            }}
            style={{
              cursor: 'pointer',
              fontWeight: 600,
              color: isActive ? TOKEN.coral : TOKEN.einV,
              borderBottom: isActive
                ? `1.5px solid ${TOKEN.coral}`
                : `1px dashed ${TOKEN.gray300}`,
              padding: '0 1px',
            }}
          >
            {p.text}
          </span>
        );
      })}
    </>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   HoverObjectStrip — the visual companion that lights up when a token is
   hovered. Three columns (RepSet / Outs(rep) / coeff). The active column
   gets a coral border + label; the others stay muted.
   ───────────────────────────────────────────────────────────────────────────── */
function HoverObjectStrip({ hoverTokenId, reducedMotion }) {
  const cols = [
    { id: 'RepSet', label: 'RepSet', glyphs: ['rep₁', 'rep₂', 'rep₃'] },
    { id: 'Outs',   label: 'Outs(rep)', glyphs: ['out₁', 'out₂'] },
    { id: 'coeff',  label: 'coeff',  glyphs: ['c₁=2', 'c₂=1'] },
  ];

  return (
    <div
      data-testid="loop-morph-stepper-hover-strip"
      className="mt-3 grid gap-2"
      style={{
        gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
      }}
    >
      {cols.map((col) => {
        const isActive = hoverTokenId === col.id;
        return (
          <div
            key={col.id}
            data-testid="loop-morph-stepper-hover-col"
            data-token-id={col.id}
            data-active={isActive ? 'true' : 'false'}
            className="rounded-md border px-2 py-1.5"
            style={{
              borderColor: isActive ? TOKEN.coral : TOKEN.gray200,
              background: isActive ? TOKEN.coralLight : TOKEN.white,
              transition: reducedMotion
                ? 'none'
                : 'border-color 0.18s ease, background 0.18s ease',
            }}
          >
            <div
              className="font-mono text-[10px] font-semibold uppercase tracking-[0.08em]"
              style={{ color: isActive ? TOKEN.coral : TOKEN.gray500 }}
            >
              {col.label}
            </div>
            <div className="mt-1 flex flex-wrap gap-1">
              {col.glyphs.map((g) => (
                <span
                  key={g}
                  className="rounded px-1.5 py-0.5 font-mono text-[10px]"
                  style={{
                    background: isActive ? TOKEN.white : TOKEN.gray100,
                    color: isActive ? TOKEN.gray900 : TOKEN.gray600,
                    border: `1px solid ${isActive ? TOKEN.coral : TOKEN.gray200}`,
                  }}
                >
                  {g}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   StepDots — five circles with the current step highlighted; serves as both
   visual indicator and keyboard-focusable selector.
   ───────────────────────────────────────────────────────────────────────────── */
function StepDots({ stepIdx, onSelect, reducedMotion }) {
  return (
    <div
      role="tablist"
      aria-label="Loop morph stepper position"
      data-testid="loop-morph-stepper-dots"
      className="flex items-center gap-2"
    >
      {STEPS.map((step, idx) => {
        const isCurrent = idx === stepIdx;
        return (
          <button
            key={step.id}
            type="button"
            role="tab"
            aria-current={isCurrent ? 'step' : undefined}
            aria-selected={isCurrent}
            aria-label={`Step ${idx + 1} of ${N_STEPS}: ${step.title}`}
            data-testid="loop-morph-stepper-dot"
            data-step-idx={idx}
            onClick={() => onSelect(idx)}
            style={{
              width: 18,
              height: 18,
              borderRadius: '50%',
              border: `1.5px solid ${isCurrent ? TOKEN.coral : TOKEN.gray300}`,
              background: isCurrent ? TOKEN.coral : TOKEN.white,
              cursor: 'pointer',
              padding: 0,
              transition: reducedMotion
                ? 'none'
                : 'background 0.18s ease, border-color 0.18s ease',
            }}
          />
        );
      })}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */
function LoopMorphStepper() {
  const [stepIdx, setStepIdx] = useState(0);
  const [hoverTokenId, setHoverTokenId] = useState(null);
  const reducedMotion = usePrefersReducedMotion();
  const containerRef = useRef(null);

  const step = STEPS[stepIdx];

  const goPrev = useCallback(() => {
    setStepIdx((prev) => Math.max(0, prev - 1));
  }, []);
  const goNext = useCallback(() => {
    setStepIdx((prev) => Math.min(N_STEPS - 1, prev + 1));
  }, []);

  // Arrow-key navigation when focus is anywhere inside the stepper container.
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goPrev();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        goNext();
      } else if (e.key === 'Home') {
        e.preventDefault();
        setStepIdx(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        setStepIdx(N_STEPS - 1);
      }
    },
    [goPrev, goNext],
  );

  // Pulse-line indices: the dense side pulses line 1 (the project += line) on
  // steps 1–3; the rep side cycles line 0 → 1 → 2 → 3 across the stepper.
  const densePulseLine = useMemo(() => {
    if (step.densePulse > 0) return 1;
    return null;
  }, [step.densePulse]);
  const repPulseLine = useMemo(() => {
    if (step.repPulse === 0) return null;
    if (step.repPulse === 1) return 1; // base_val
    if (step.repPulse === 2) return 2; // for out_rep
    return 3; // accumulate
  }, [step.repPulse]);

  return (
    <div
      ref={containerRef}
      data-testid="loop-morph-stepper"
      className="rounded-lg border p-4"
      style={{
        borderColor: TOKEN.gray200,
        background: TOKEN.white,
      }}
      onKeyDown={handleKeyDown}
    >
      {/* Header: title + step indicator (1..5) */}
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-2">
        <div
          className="font-mono text-[12px] tracking-[0.04em]"
          style={{ color: TOKEN.gray700 }}
        >
          dense → representative loop morph
        </div>
        <div
          className="font-mono text-[11px]"
          data-testid="loop-morph-stepper-progress"
          style={{ color: TOKEN.gray500 }}
        >
          step <strong style={{ color: TOKEN.gray900 }}>{stepIdx + 1}</strong> / {N_STEPS}
        </div>
        <div className="ml-auto">
          <StepDots
            stepIdx={stepIdx}
            onSelect={setStepIdx}
            reducedMotion={reducedMotion}
          />
        </div>
      </div>

      {/* Side-by-side pseudocode columns */}
      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
      >
        <PseudocodeBlock
          side="dense"
          title="dense loop"
          subtitle="every full assignment touched once"
          lines={DENSE_PSEUDOCODE}
          pulseLine={densePulseLine}
          weight={step.leftWeight}
          reducedMotion={reducedMotion}
          onTokenHover={setHoverTokenId}
          hoverTokenId={hoverTokenId}
        />
        <PseudocodeBlock
          side="rep"
          title="representative loop"
          subtitle="multiply once · accumulate many"
          lines={REP_PSEUDOCODE}
          pulseLine={repPulseLine}
          weight={step.rightWeight}
          reducedMotion={reducedMotion}
          onTokenHover={setHoverTokenId}
          hoverTokenId={hoverTokenId}
        />
      </div>

      {/* Visual hover strip — three columns lit by the right-side tokens */}
      <HoverObjectStrip hoverTokenId={hoverTokenId} reducedMotion={reducedMotion} />

      {/* Caption — describes the current transformation */}
      <div
        className="mt-3 rounded-md border px-3 py-2 font-mono text-[12px] leading-6"
        data-testid="loop-morph-stepper-caption"
        style={{
          borderColor: TOKEN.gray200,
          background: TOKEN.gray50,
          color: TOKEN.gray700,
        }}
        aria-live="polite"
      >
        <span style={{ color: TOKEN.coral, fontWeight: 600 }}>{step.title}</span>
        <span style={{ color: TOKEN.gray400 }}> — </span>
        <span>{step.caption}</span>
      </div>

      {/* Prev / Next stepper controls */}
      <div className="mt-3 flex items-center justify-between gap-3">
        <button
          type="button"
          data-testid="loop-morph-stepper-prev"
          onClick={goPrev}
          disabled={stepIdx === 0}
          aria-label="Previous step"
          style={{
            padding: '6px 14px',
            fontSize: '12px',
            fontWeight: 500,
            fontFamily: "'Inter', sans-serif",
            borderRadius: '6px',
            border: `1.5px solid ${stepIdx === 0 ? TOKEN.gray200 : TOKEN.gray300}`,
            background: stepIdx === 0 ? TOKEN.gray100 : TOKEN.white,
            color: stepIdx === 0 ? TOKEN.gray400 : TOKEN.gray700,
            cursor: stepIdx === 0 ? 'not-allowed' : 'pointer',
            transition: reducedMotion
              ? 'none'
              : 'background 0.15s ease, color 0.15s ease, border-color 0.15s ease',
          }}
        >
          ← Prev
        </button>
        <div
          className="font-mono text-[11px]"
          style={{ color: TOKEN.gray500 }}
        >
          arrow keys ←/→ to step · Home/End for first/last
        </div>
        <button
          type="button"
          data-testid="loop-morph-stepper-next"
          onClick={goNext}
          disabled={stepIdx === N_STEPS - 1}
          aria-label="Next step"
          style={{
            padding: '6px 14px',
            fontSize: '12px',
            fontWeight: 500,
            fontFamily: "'Inter', sans-serif",
            borderRadius: '6px',
            border: `1.5px solid ${stepIdx === N_STEPS - 1 ? TOKEN.gray200 : TOKEN.coral}`,
            background: stepIdx === N_STEPS - 1 ? TOKEN.gray100 : TOKEN.coralLight,
            color: stepIdx === N_STEPS - 1 ? TOKEN.gray400 : TOKEN.coral,
            cursor: stepIdx === N_STEPS - 1 ? 'not-allowed' : 'pointer',
            transition: reducedMotion
              ? 'none'
              : 'background 0.15s ease, color 0.15s ease, border-color 0.15s ease',
          }}
        >
          Next →
        </button>
      </div>
    </div>
  );
}

export default LoopMorphStepper;
