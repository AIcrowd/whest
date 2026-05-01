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
function SubscriptTokens({ text, hoveredLabels, onHoveredLabelsChange }) {
  const chars = Array.from(String(text ?? ''));
  const hasHover = hoveredLabels instanceof Set && hoveredLabels.size > 0;
  return (
    <>
      {chars.map((ch, idx) => {
        const isLetter = /[A-Za-z]/.test(ch);
        const isHit = hasHover && isLetter && hoveredLabels.has(ch);
        if (!isLetter || !onHoveredLabelsChange) {
          return (
            <span
              key={idx}
              className={isHit ? 'text-black underline decoration-primary decoration-2 underline-offset-2' : undefined}
            >
              {ch}
            </span>
          );
        }
        // Interactive letter: hover fires the cross-highlighting bus
        return (
          <span
            key={idx}
            className={cn(
              'cursor-default',
              isHit
                ? 'text-black underline decoration-primary decoration-2 underline-offset-2'
                : 'hover:text-black hover:underline hover:decoration-primary hover:decoration-2 hover:underline-offset-2',
            )}
            onMouseEnter={() => onHoveredLabelsChange(new Set([ch]))}
            onMouseLeave={() => onHoveredLabelsChange(null)}
            onFocus={() => onHoveredLabelsChange(new Set([ch]))}
            onBlur={() => onHoveredLabelsChange(null)}
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
export function FormulaHighlighted({ example, hoveredLabels, onHoveredLabelsChange }) {
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
      <SubscriptTokens text={expr.subscripts} hoveredLabels={hoveredLabels} onHoveredLabelsChange={onHoveredLabelsChange} />
      {/* Arrow is the one coral accent inside the neutral stadium-pill —
          matches `.formula-live .arr { color: var(--coral) }` in
          design-system/preview/components.html. */}
      <span className="mx-1 text-coral">{'→'}</span>
      <SubscriptTokens text={expr.output ?? ''} hoveredLabels={hoveredLabels} onHoveredLabelsChange={onHoveredLabelsChange} />
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

// Inline dimension stepper — 28px tall, gray border, mono font.
// Keyboard arrow-up/down increments within [min, max].
// Fires onDimensionNChange(newValue) when the value changes.
// Falls back to static display when onDimensionNChange is null.
function DimensionStepper({ dimensionN, onDimensionNChange, min = 2, max = 8 }) {
  const value = dimensionN ?? '—';
  const canInteract = typeof onDimensionNChange === 'function' && typeof dimensionN === 'number';

  const decrement = () => {
    if (!canInteract) return;
    onDimensionNChange(Math.max(min, dimensionN - 1));
  };
  const increment = () => {
    if (!canInteract) return;
    onDimensionNChange(Math.min(max, dimensionN + 1));
  };

  if (!canInteract) {
    // Read-only badge (legacy: no setter provided)
    return (
      <Badge
        variant="outline"
        aria-label={`n=${dimensionN ?? '—'}`}
        className="shrink-0 border-gray-200 bg-white font-mono text-gray-500 shadow-none"
      >
        {`n=${dimensionN ?? '—'}`}
      </Badge>
    );
  }

  return (
    <div
      className="inline-flex h-7 items-center gap-0 overflow-hidden rounded-md border border-gray-200 bg-white font-mono text-xs text-gray-700 shadow-none"
      role="group"
      aria-label={`n=${dimensionN ?? '—'}`}
    >
      <button
        type="button"
        aria-label="Decrease dimension"
        onClick={decrement}
        disabled={dimensionN <= min}
        className="flex h-7 w-6 items-center justify-center border-r border-gray-200 text-gray-400 transition-colors hover:bg-gray-50 hover:text-gray-700 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/25"
      >
        −
      </button>
      <span
        aria-live="polite"
        aria-atomic="true"
        className="min-w-[2.5rem] px-2 text-center text-[12px] font-mono leading-7 text-gray-700"
      >
        {`n=${dimensionN ?? '—'}`}
      </span>
      <button
        type="button"
        aria-label="Increase dimension"
        onClick={increment}
        disabled={dimensionN >= max}
        className="flex h-7 w-6 items-center justify-center border-l border-gray-200 text-gray-400 transition-colors hover:bg-gray-50 hover:text-gray-700 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/25"
      >
        +
      </button>
    </div>
  );
}

export default function StickyBar({
  example,
  group,
  activeActId,
  hoveredLabels = null,
  dimensionN = null,
  onDimensionNChange = null,
  onHoveredLabelsChange = null,
  activeAlphaMethod = null,
  onActiveAlphaMethodHoverChange = null,
}) {
  const [showMetadataPopover, setShowMetadataPopover] = useState(false);
  const [metadataAnchorRect, setMetadataAnchorRect] = useState(null);
  const metadataTriggerRef = useRef(null);
  const closeTimerRef = useRef(null);
  const metadataItems = useMemo(
    () => buildMetadataItems({ example, group }),
    [example, group],
  );

  // Behavior 4 — Compact-on-scroll
  // Detect prefers-reduced-motion once at mount (no re-render; purely CSS gate)
  const prefersReducedMotionRef = useRef(
    typeof window !== 'undefined'
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false,
  );
  const [isCompact, setIsCompact] = useState(false);
  const rafRef = useRef(null);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;

    const onScroll = () => {
      if (rafRef.current) return;
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null;
        setIsCompact(window.scrollY > 200);
      });
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    // Sync immediately in case the page is already scrolled (e.g., back-navigation)
    onScroll();
    return () => {
      window.removeEventListener('scroll', onScroll, { passive: true });
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
    };
  }, []);

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

  // Behavior 4 — CSS transition guard: no transition if prefers-reduced-motion
  const compactTransitionClass = prefersReducedMotionRef.current
    ? ''
    : 'transition-[height,opacity] duration-200 ease-out';

  return (
    <div
      className={cn(
        'sticky top-0 z-40 border-b border-border/70 bg-background/95 backdrop-blur-sm',
        compactTransitionClass,
        isCompact ? 'sticky-bar--compact' : 'sticky-bar--expanded',
      )}
      data-compact={isCompact ? 'true' : 'false'}
    >
      <div
        className={cn(
          'mx-auto flex max-w-[1460px] px-6 py-4 md:px-8',
          isCompact
            ? 'flex-row items-center justify-between gap-4'
            : 'flex-col gap-4 md:flex-row md:items-center md:justify-between',
        )}
      >
        {/* Whest wordmark — the one in-product brand anchor. Matches the
            `.brand` slot of design-system/Whest Einsum Explorer.html and
            the nav wordmark in lib/layout.shared.tsx. Reuses the global
            `.whest-wordmark` utility (Newsreader 700 opsz32, coral dot);
            do not reimplement. */}
        <Link
          href="/"
          aria-label="Whest."
          className="whest-wordmark mr-4 shrink-0 text-[20px] no-underline"
        >
          Whest<span className="whest-wordmark__dot">.</span>
        </Link>
        <nav
          className={cn(
            'flex shrink-0 items-center gap-1 overflow-x-auto pb-1 md:pb-0',
            isCompact && 'hidden md:flex',
          )}
        >
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
              {/* Behavior 2 — Dimension knob: interactive stepper when
                  onDimensionNChange is provided, static badge otherwise. */}
              <DimensionStepper
                dimensionN={dimensionN}
                onDimensionNChange={onDimensionNChange}
              />

              {/* Behavior 3 — α-method badge: hover fires the tree-leaf bus */}
              {activeAlphaMethod && (
                <Badge
                  variant="outline"
                  className="shrink-0 cursor-default border-gray-200 bg-white font-mono text-[11px] text-gray-600 shadow-none transition-colors hover:border-primary/30 hover:bg-primary/5"
                  aria-label={`Alpha method: ${activeAlphaMethod}`}
                  onMouseEnter={() => onActiveAlphaMethodHoverChange?.(activeAlphaMethod)}
                  onMouseLeave={() => onActiveAlphaMethodHoverChange?.(null)}
                  onFocus={() => onActiveAlphaMethodHoverChange?.(activeAlphaMethod)}
                  onBlur={() => onActiveAlphaMethodHoverChange?.(null)}
                >
                  α: {activeAlphaMethod}
                </Badge>
              )}

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
                {/* Behavior 1 — label-hover bus: FormulaHighlighted now also
                    calls onHoveredLabelsChange when hovering letter tokens */}
                <FormulaHighlighted
                  example={example}
                  hoveredLabels={hoveredLabels}
                  onHoveredLabelsChange={onHoveredLabelsChange}
                />
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
