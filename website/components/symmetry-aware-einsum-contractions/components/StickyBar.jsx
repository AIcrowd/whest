import Link from 'next/link';
import { createPortal } from 'react-dom';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { formatGeneratorNotation } from '../lib/symmetryLabel.js';
import { EXPLORER_ACTS } from './explorerNarrative.js';
import SymmetryBadge from './SymmetryBadge.jsx';

// Character-by-character renderer for a subscript/output *label region*
// (e.g. "ia,ib,ic" or "abc"). Letters matching `hoveredLabels` swap to
// pitch black with a coral underline; everything else passes through.
// Layout is stable: no bold (glyph widths stay fixed), no padding/ring
// (no per-char circle outlines).
function SubscriptTokens({ text, hoveredLabels }) {
  const chars = Array.from(String(text ?? ''));
  const hasHover = hoveredLabels instanceof Set && hoveredLabels.size > 0;
  return (
    <>
      {chars.map((ch, idx) => {
        const isLetter = /[A-Za-z]/.test(ch);
        const isHit = hasHover && isLetter && hoveredLabels.has(ch);
        if (!isHit) return <span key={idx}>{ch}</span>;
        // `decoration-coral` / `text-coral` are NOT defined in this Tailwind
        // theme — only `--primary` / `--color-primary` resolve to the coral
        // hex. `text-black` gives the strong contrast against the coral-tint
        // badge background that plain coral text can't.
        return (
          <span
            key={idx}
            className="text-black underline decoration-primary decoration-2 underline-offset-2"
          >
            {ch}
          </span>
        );
      })}
    </>
  );
}

// Renders the einsum formula with per-region tokenization. Only the
// subscript + output label regions are highlight-able — so hovering a
// label like `i` no longer lights up the `i` in "einsum". We rebuild from
// `example.expression` (structured) rather than walking the display string,
// so the visible formula always matches the authoritative subscripts.
export function FormulaHighlighted({ example, hoveredLabels }) {
  const expr = example?.expression;
  if (!expr || typeof expr.subscripts !== 'string') {
    // Pre-normalized or malformed example — fall back to the raw string
    // with no highlighting. Better than crashing or silently highlighting
    // the wrong characters.
    return <>{example?.formula ?? ''}</>;
  }
  return (
    <>
      <span>{"einsum('"}</span>
      <SubscriptTokens text={expr.subscripts} hoveredLabels={hoveredLabels} />
      {/* Arrow is the one coral accent inside the neutral stadium-pill —
          matches `.formula-live .arr { color: var(--coral) }` in
          design-system/preview/components.html. */}
      <span className="mx-1 text-coral">{'→'}</span>
      <SubscriptTokens text={expr.output ?? ''} hoveredLabels={hoveredLabels} />
      <span>{`', ${expr.operandNames ?? ''})`}</span>
    </>
  );
}

function symmetryLabel(variable) {
  if (!variable || variable.symmetry === 'none') return 'dense';
  const k = (variable.symAxes && variable.symAxes.length) || variable.rank;
  const axes = Array.isArray(variable.symAxes) ? variable.symAxes : null;
  const hasExplicitPartialAxes = axes && axes.length > 0 && axes.length < variable.rank;

  switch (variable.symmetry) {
    case 'symmetric':
      return hasExplicitPartialAxes ? `S${k}{${axes.join(',')}}` : `S${k}`;
    case 'cyclic':
      return hasExplicitPartialAxes ? `C${k}{${axes.join(',')}}` : `C${k}`;
    case 'dihedral':
      return hasExplicitPartialAxes ? `D${k}{${axes.join(',')}}` : `D${k}`;
    case 'custom':
      return formatGeneratorNotation(variable.generators) ?? (hasExplicitPartialAxes ? `custom{${axes.join(',')}}` : 'custom');
    default:
      return null;
  }
}

function symmetryLabelFromPerOp(perOpSymmetry) {
  if (!perOpSymmetry) return 'dense';
  if (perOpSymmetry === 'symmetric') return 'dense';
  if (typeof perOpSymmetry === 'string') return perOpSymmetry;

  const axes = Array.isArray(perOpSymmetry.axes) ? perOpSymmetry.axes : null;
  const k = axes?.length || null;
  switch (perOpSymmetry.type) {
    case 'cyclic':
      return k ? `C${k}` : 'cyclic';
    case 'dihedral':
      return k ? `D${k}` : 'dihedral';
    case 'custom':
      return formatGeneratorNotation(perOpSymmetry.generators) ?? (axes?.length ? `custom{${axes.join(',')}}` : 'custom');
    case 'symmetric':
      return k ? `S${k}` : 'symmetric';
    default:
      return 'dense';
  }
}

function buildMetadataItems({ example, group }) {
  const operandNames = Array.isArray(example?.operandNames)
    ? example.operandNames
    : typeof example?.expression?.operandNames === 'string'
      ? example.expression.operandNames.split(',').map((part) => part.trim()).filter(Boolean)
      : [];
  const perOpSymmetry = Array.isArray(example?.perOpSymmetry) ? example.perOpSymmetry : [];
  const variablesByName = new Map(
    (example?.variables ?? []).map((variable) => [variable.name, variable]),
  );
  const operands = operandNames.map((name, idx) => ({
    name,
    symmetry:
      idx < perOpSymmetry.length && perOpSymmetry[idx] !== undefined
        ? symmetryLabelFromPerOp(perOpSymmetry[idx])
        : symmetryLabel(variablesByName.get(name)),
  }));
  return {
    operands,
    groupLabel: group?.fullGroupName || 'trivial',
  };
}

export function SymmetryChip({ name, symmetry }) {
  return (
    <span className="inline-flex h-6 items-center gap-1 rounded-full border border-primary/18 bg-white px-2.5 text-xs font-mono text-foreground shadow-sm">
      <span className="font-semibold text-primary">{name}</span>
      <span className="text-muted-foreground">:</span>
      <span className="text-coral">{symmetry}</span>
    </span>
  );
}

function StickyMetadataPopover({ anchorRect, operands, groupLabel }) {
  if (!anchorRect || typeof document === 'undefined') return null;

  const viewportWidth = document.documentElement.clientWidth;
  const clampedCenterX = Math.min(
    Math.max(anchorRect.left + anchorRect.width / 2, 180),
    viewportWidth - 180,
  );
  const x = Math.min(
    Math.max(clampedCenterX, 32),
    viewportWidth - 32,
  );
  const y = anchorRect.bottom + 10;

  return createPortal(
    <div
      role="dialog"
      aria-label="Einsum metadata"
      className="pointer-events-none fixed z-[9999] inline-flex w-max max-w-[calc(100vw-2rem)] rounded-xl border border-stone-200 bg-white px-4 py-3 text-stone-900 shadow-[0_24px_60px_rgba(15,23,42,0.12)]"
      style={{
        left: x,
        top: y,
        transform: 'translateX(-50%)',
      }}
    >
      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[12px] leading-6 text-stone-700">
        {operands.map((operand, idx) => (
          <span key={`sticky-metadata-${operand.name}-${idx}`} className="contents">
            {idx > 0 && <span className="text-stone-400">,</span>}
            <SymmetryChip name={operand.name} symmetry={operand.symmetry} />
          </span>
        ))}
        <span className="text-stone-500">→</span>
        <SymmetryBadge value={groupLabel} className="h-6 px-2.5 text-[11px] leading-5 shadow-none" />
      </div>
      <div
        className="absolute left-1/2 top-[-6px] h-1.5 w-3 bg-white"
        style={{
          clipPath: 'polygon(50% 0, 0 100%, 100% 100%)',
          transform: 'translateX(-50%)',
        }}
      />
    </div>,
    document.body,
  );
}

export default function StickyBar({ example, group, activeActId, hoveredLabels = null, dimensionN = null }) {
  const [showMetadataPopover, setShowMetadataPopover] = useState(false);
  const [metadataAnchorRect, setMetadataAnchorRect] = useState(null);
  const metadataTriggerRef = useRef(null);
  const closeTimerRef = useRef(null);
  const metadataItems = useMemo(
    () => buildMetadataItems({ example, group }),
    [example, group],
  );

  const clearCloseTimer = () => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const openMetadataPopover = () => {
    clearCloseTimer();
    if (!metadataTriggerRef.current) return;
    setMetadataAnchorRect(metadataTriggerRef.current.getBoundingClientRect());
    setShowMetadataPopover(true);
  };

  const scheduleMetadataClose = () => {
    clearCloseTimer();
    closeTimerRef.current = window.setTimeout(() => setShowMetadataPopover(false), 80);
  };

  useEffect(() => () => clearCloseTimer(), []);

  useEffect(() => {
    if (!showMetadataPopover) return undefined;

    const dismiss = () => setShowMetadataPopover(false);
    const dismissOnEscape = (event) => {
      if (event.key === 'Escape') setShowMetadataPopover(false);
    };
    const dismissIfOutside = (event) => {
      if (!metadataTriggerRef.current) return;
      if (event.target instanceof Node && metadataTriggerRef.current.contains(event.target)) return;
      setShowMetadataPopover(false);
    };

    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('keydown', dismissOnEscape);
    window.addEventListener('pointerdown', dismissIfOutside);
    window.addEventListener('blur', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('keydown', dismissOnEscape);
      window.removeEventListener('pointerdown', dismissIfOutside);
      window.removeEventListener('blur', dismiss);
    };
  }, [showMetadataPopover]);

  return (
    <div className="sticky top-0 z-40 border-b border-border/70 bg-background/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1460px] flex-col gap-4 px-6 py-4 md:flex-row md:items-center md:justify-between md:px-8">
        {/* Flopscope wordmark — the one in-product brand anchor. Matches the
            `.brand` slot of design-system/Flopscope Einsum Explorer.html and
            the nav wordmark in lib/layout.shared.tsx. Reuses the global
            `.flopscope-wordmark` utility (Newsreader 700 opsz32, coral dot);
            do not reimplement. */}
        <Link
          href="/"
          aria-label="flopscope."
          className="flopscope-wordmark mr-4 shrink-0 text-[20px] no-underline"
        >
          <span className="flopscope-wordmark__flop">flop</span>scope<span className="flopscope-wordmark__dot">.</span>
        </Link>
        <nav className="flex shrink-0 items-center gap-1 overflow-x-auto pb-1 md:pb-0">
          {EXPLORER_ACTS.map((act, idx) => {
            const isActive = activeActId === act.id;
            return (
              <a
                key={act.id}
                href={`#${act.id}`}
                className={cn(
                  buttonVariants({ size: 'sm', variant: 'ghost' }),
                  'inline-flex h-9 min-h-9 items-center gap-2 rounded-full border px-3 transition-colors',
                  isActive
                    ? 'border-[var(--coral)] bg-white text-[var(--coral-hover)] hover:border-[var(--coral)] hover:bg-[color:color-mix(in_oklab,var(--coral)_8%,white)]'
                    : 'border-transparent text-muted-foreground hover:border-primary/25 hover:bg-primary/8 hover:text-primary',
                )}
              >
                <Badge
                  variant={isActive ? 'default' : 'outline'}
                  className={cn(
                    'flex h-5 min-w-5 items-center justify-center rounded-full border px-1.5 py-0 text-[11px] font-semibold leading-none',
                    isActive
                      ? 'border-[var(--coral)] bg-[var(--coral)] text-white'
                      : 'border-primary/20 bg-background text-muted-foreground',
                  )}
                >
                  {idx + 1}
                </Badge>
                {act.navTitle}
              </a>
            );
          })}
        </nav>

        <div className="flex min-w-0 shrink flex-col items-start gap-2 self-end md:max-w-[42rem] md:items-end md:self-auto">
          {example && (
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="outline"
                className="shrink-0 border-gray-200 bg-white text-gray-500 shadow-none"
              >
                {`n=${dimensionN ?? '—'}`}
              </Badge>
              {/* Neutral stadium pill per design-system `.formula-live`:
                  gray-100 ground, gray-600 ink, 1px gray-200 border,
                  rounded-full (=20px stadium). The only coral moment
                  inside is the arrow (see FormulaHighlighted above). */}
              <button
                ref={metadataTriggerRef}
                type="button"
                aria-haspopup="dialog"
                aria-expanded={showMetadataPopover}
                onPointerEnter={openMetadataPopover}
                onPointerLeave={scheduleMetadataClose}
                onFocus={openMetadataPopover}
                onBlur={scheduleMetadataClose}
                onClick={() => {
                  if (showMetadataPopover) {
                    setShowMetadataPopover(false);
                    return;
                  }
                  openMetadataPopover();
                }}
                className="block rounded-full border border-gray-200 bg-gray-100 px-3 py-1 text-left text-sm font-mono font-medium text-gray-600 shadow-sm transition-colors hover:border-stone-300 hover:bg-stone-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/25"
              >
                <FormulaHighlighted example={example} hoveredLabels={hoveredLabels} />
              </button>
              {showMetadataPopover && (
                <StickyMetadataPopover
                  anchorRect={metadataAnchorRect}
                  operands={metadataItems.operands}
                  groupLabel={metadataItems.groupLabel}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
