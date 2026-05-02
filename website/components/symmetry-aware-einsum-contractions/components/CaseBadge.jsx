import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Badge } from '@/components/ui/badge';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import GlossaryList from './GlossaryList.jsx';
import { getRegimePresentation } from './regimePresentation.js';
import { cn } from '../lib/utils.js';

const TOOLTIP_WIDTH = 520;
const TOOLTIP_HEIGHT = 340;
const VIEWPORT_PADDING = 16;
const TOOLTIP_OFFSET = 8;
const TOOLTIP_CLOSE_DELAY_MS = 160;

/**
 * Module-level coordinator: at most one CaseBadge tooltip is visible at a time.
 * When a badge opens its tooltip it broadcasts its identity; any other
 * subscriber that is currently open closes itself. This prevents overlapping
 * tooltips from fighting each other when the mouse path crosses multiple
 * pills (the original "overlap weirdness" bug).
 */
const tooltipSubscribers = new Set();
function broadcastTooltipOpen(ownerId) {
  for (const notify of tooltipSubscribers) notify(ownerId);
}

/**
 * Mix a #RRGGBB hex color with white to produce a soft background tint.
 * `amount` in [0,1]: 0 = pure color, 1 = white.
 */
function mixWithWhite(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mix = (c) => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

function mixWithBlack(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mix = (c) => Math.round(c * (1 - amount));
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

function relativeLuminance(hex) {
  const rgb = [hex.slice(1, 3), hex.slice(3, 5), hex.slice(5, 7)]
    .map((part) => parseInt(part, 16) / 255)
    .map((channel) => (
      channel <= 0.03928
        ? channel / 12.92
        : ((channel + 0.055) / 1.055) ** 2.4
    ));
  return (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2]);
}

function isDarkColor(hex) {
  return relativeLuminance(hex) < 0.3;
}

function colorsFor(baseColor, { outlined = false } = {}) {
  const darkSurface = isDarkColor(baseColor);
  const lightSurface = relativeLuminance(baseColor) > 0.58;
  if (outlined) {
    return {
      bg: '#FFFFFF',
      text: darkSurface
        ? mixWithWhite(baseColor, 0.08)
        : (lightSurface ? mixWithBlack(baseColor, 0.28) : mixWithBlack(baseColor, 0.18)),
      border: darkSurface
        ? mixWithWhite(baseColor, 0.16)
        : (lightSurface ? mixWithBlack(baseColor, 0.12) : mixWithBlack(baseColor, 0.08)),
    };
  }
  return {
    bg: darkSurface
      ? mixWithWhite(baseColor, 0.7)
      : (lightSurface ? mixWithBlack(baseColor, 0.06) : mixWithWhite(baseColor, 0.38)),
    text: darkSurface ? mixWithBlack(baseColor, 0.06) : mixWithBlack(baseColor, 0.56),
    border: darkSurface ? mixWithWhite(baseColor, 0.16) : mixWithBlack(baseColor, 0.12),
  };
}

function getBadgeClasses(variant, size) {
  if (variant === 'compact') {
    return size === 'xs'
      ? 'h-5 w-5 justify-center rounded-full px-0.5 py-0 leading-none text-[11px] font-bold'
      : 'h-6 w-6 justify-center rounded-full px-0.5 py-0 leading-none text-[11px] font-bold';
  }

  return size === 'xs'
    ? 'rounded-full px-2 py-0.5 text-[10px] font-semibold'
    : 'rounded-full px-2.5 py-0.5 text-xs font-semibold';
}

export default function CaseBadge({
  regimeId,
  size = 'sm',
  variant = 'pill',
  interactive = true,
  active = false,
  themeOverride = null,
  presentationThemeOverride = themeOverride,
  className,
  children = null,
}) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({
    x: 0,
    y: 0,
    flipped: false,
    scale: 1,
  });
  const ref = useRef(null);
  const tooltipRef = useRef(null);
  const closeTimerRef = useRef(null);
  const ownerIdRef = useRef(null);
  if (ownerIdRef.current === null) ownerIdRef.current = Symbol('case-badge');

  // Subscribe to peer openings: when any other CaseBadge opens its tooltip,
  // close ours so that only one is ever visible at a time.
  useEffect(() => {
    const notify = (openedOwnerId) => {
      if (openedOwnerId !== ownerIdRef.current) setShowTooltip(false);
    };
    tooltipSubscribers.add(notify);
    return () => { tooltipSubscribers.delete(notify); };
  }, []);

  useEffect(() => () => {
    if (closeTimerRef.current !== null) window.clearTimeout(closeTimerRef.current);
  }, []);

  const outlinedPill = variant !== 'compact';
  const presentation = getRegimePresentation(regimeId, presentationThemeOverride);
  const colors = colorsFor(presentation.color ?? '#94A3B8', { outlined: outlinedPill });
  const tooltip = interactive ? presentation.tooltip : null;

  const clearCloseTimer = () => {
    if (closeTimerRef.current !== null) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const scheduleClose = () => {
    clearCloseTimer();
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null;
      setShowTooltip(false);
    }, TOOLTIP_CLOSE_DELAY_MS);
  };

  const handleEnter = () => {
    if (!tooltip || !ref.current) return;
    clearCloseTimer();

    const rect = ref.current.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const tooltipWidth = Math.min(TOOLTIP_WIDTH, vw - (VIEWPORT_PADDING * 2));
    let x = rect.left + rect.width / 2;
    x = Math.max(
      tooltipWidth / 2 + VIEWPORT_PADDING,
      Math.min(x, vw - tooltipWidth / 2 - VIEWPORT_PADDING),
    );

    const roomAbove = rect.top - VIEWPORT_PADDING - TOOLTIP_OFFSET;
    const roomBelow = vh - rect.bottom - VIEWPORT_PADDING - TOOLTIP_OFFSET;
    const placeBelow = roomBelow >= TOOLTIP_HEIGHT || roomBelow >= roomAbove;

    const y = placeBelow
      ? rect.bottom + TOOLTIP_OFFSET
      : Math.max(VIEWPORT_PADDING, rect.top - TOOLTIP_OFFSET - TOOLTIP_HEIGHT);

    setTooltipPos({ x, y, flipped: placeBelow, scale: 1 });
    broadcastTooltipOpen(ownerIdRef.current);
    setShowTooltip(true);
  };

  useLayoutEffect(() => {
    if (!showTooltip || !tooltip || !ref.current || !tooltipRef.current) return;

    const triggerRect = ref.current.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const naturalWidth = tooltipRef.current.offsetWidth;
    const naturalHeight = tooltipRef.current.offsetHeight;
    const availableWidth = vw - (VIEWPORT_PADDING * 2);
    const availableHeight = vh - (VIEWPORT_PADDING * 2);
    const heightScale = Math.min(1, availableHeight / Math.max(naturalHeight, 1));
    const widthScale = Math.min(1, availableWidth / Math.max(naturalWidth, 1));
    const scale = Math.min(heightScale, widthScale);
    const renderedWidth = naturalWidth * scale;
    const renderedHeight = naturalHeight * scale;

    let x = triggerRect.left + triggerRect.width / 2;
    x = Math.max(
      renderedWidth / 2 + VIEWPORT_PADDING,
      Math.min(x, vw - renderedWidth / 2 - VIEWPORT_PADDING),
    );

    const roomAbove = triggerRect.top - VIEWPORT_PADDING - TOOLTIP_OFFSET;
    const roomBelow = vh - triggerRect.bottom - VIEWPORT_PADDING - TOOLTIP_OFFSET;
    const placeBelow = roomBelow >= renderedHeight || roomBelow >= roomAbove;
    const unclampedY = placeBelow
      ? triggerRect.bottom + TOOLTIP_OFFSET
      : triggerRect.top - TOOLTIP_OFFSET - renderedHeight;
    const y = Math.max(
      VIEWPORT_PADDING,
      Math.min(unclampedY, vh - VIEWPORT_PADDING - renderedHeight),
    );

    setTooltipPos((prev) => {
      if (
        Math.abs(prev.x - x) < 0.5
        && Math.abs(prev.y - y) < 0.5
        && Math.abs(prev.scale - scale) < 0.01
        && prev.flipped === placeBelow
      ) {
        return prev;
      }
      return { x, y, flipped: placeBelow, scale };
    });
  }, [showTooltip, tooltip]);

  // Defensive dismissal: pointerleave isn't always guaranteed to fire (scroll,
  // focus change, programmatic events, touch). Close the tooltip whenever the
  // user interacts elsewhere so it can't get stuck floating on the page.
  useEffect(() => {
    if (!showTooltip) return undefined;

    const dismiss = () => {
      clearCloseTimer();
      setShowTooltip(false);
    };
    const dismissOnEscape = (event) => {
      if (event.key === 'Escape') setShowTooltip(false);
    };
    const dismissIfOutside = (event) => {
      if (!ref.current) return;
      if (event.target instanceof Node && ref.current.contains(event.target)) return;
      setShowTooltip(false);
    };
    const dismissOnOutsideScroll = (event) => {
      if (
        tooltipRef.current
        && event.target instanceof Node
        && tooltipRef.current.contains(event.target)
      ) {
        return;
      }
      dismiss();
    };

    window.addEventListener('scroll', dismissOnOutsideScroll, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('keydown', dismissOnEscape);
    window.addEventListener('pointerdown', dismissIfOutside);
    window.addEventListener('blur', dismiss);

    return () => {
      window.removeEventListener('scroll', dismissOnOutsideScroll, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('keydown', dismissOnEscape);
      window.removeEventListener('pointerdown', dismissIfOutside);
      window.removeEventListener('blur', dismiss);
    };
  }, [showTooltip]);

  const label = variant === 'compact' ? presentation.shortLabel : presentation.label;

  const triggerProps = tooltip
    ? {
      onPointerEnter: handleEnter,
      onPointerLeave: scheduleClose,
    }
    : {};

  const trigger = children ? (
    // Passthrough: wrap arbitrary children as the tooltip trigger. Used by
    // ComponentSummaryTable's Method cell so hovering the formulas opens
    // the same tooltip that the pill in the Case band shows.
    <span
      ref={ref}
      {...triggerProps}
      aria-label={presentation.label}
      className={cn(tooltip && 'cursor-help inline-block', className)}
    >
      {children}
    </span>
  ) : (
    <span {...triggerProps}>
      <Badge
        ref={ref}
        variant="outline"
        className={cn(
          'inline-flex items-center border font-mono',
          variant === 'compact' ? 'shrink-0' : '',
          getBadgeClasses(variant, size),
          tooltip && 'cursor-help',
          className,
        )}
        style={{
          backgroundColor: colors.bg,
          color: colors.text,
          borderColor: colors.border,
          boxShadow: active
            ? `0 0 0 4px #FFFFFF, 0 0 0 5px ${colors.border}`
            : undefined,
        }}
        aria-label={presentation.label}
      >
        {label}
      </Badge>
    </span>
  );

  return (
    <>
      {trigger}

      {showTooltip && tooltip && typeof document !== 'undefined' && createPortal(
        <div
          ref={tooltipRef}
          className="pointer-events-auto fixed z-[9999] w-[520px] max-w-[calc(100vw-2rem)] rounded-xl border border-stone-200 bg-white px-5 py-4 text-base text-stone-900 shadow-[0_24px_60px_rgba(15,23,42,0.16)]"
          role="tooltip"
          onPointerEnter={clearCloseTimer}
          onPointerLeave={scheduleClose}
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: `translateX(-50%) scale(${tooltipPos.scale})`,
            transformOrigin: 'top center',
          }}
        >
          <div className="mb-1 whitespace-normal break-words text-lg font-semibold leading-7">
            <InlineMathText themeOverride={themeOverride}>{tooltip.title}</InlineMathText>
          </div>
          {tooltip.whenText && (
            <div className="mb-2 text-xs uppercase tracking-wider text-stone-500">
              Applies when: <InlineMathText themeOverride={themeOverride}>{tooltip.whenText}</InlineMathText>
            </div>
          )}
          <div className="mb-2 text-xs uppercase tracking-wider text-stone-500">
            Counts: filled O → Q cells for this component
          </div>
          <div className="max-w-full whitespace-normal break-words text-base leading-7 text-stone-700">
            <InlineMathText themeOverride={themeOverride}>{tooltip.body}</InlineMathText>
          </div>
          {tooltip.latex && (
            <div className="mt-3 max-w-full border-t border-stone-200 pt-3 text-base text-stone-900">
              <div className="min-w-0">
                <Latex math={tooltip.latex} display themeOverride={themeOverride} />
              </div>
            </div>
          )}
          {tooltip.glossary && (
            <div className="mt-3 whitespace-normal break-words border-t border-stone-200 pt-3 text-sm leading-6 text-stone-700">
              <div className="mb-1.5 text-xs uppercase tracking-wider text-stone-500">Where</div>
              <GlossaryList entries={tooltip.glossary} themeOverride={themeOverride} />
            </div>
          )}
          <div
            className={cn(
              'absolute left-1/2 h-1.5 w-3 bg-white',
              tooltipPos.flipped ? 'top-[-6px]' : 'bottom-[-6px]',
            )}
            style={{
              clipPath: 'polygon(0 0, 100% 0, 50% 100%)',
              transform: tooltipPos.flipped
                ? 'translateX(-50%) rotate(180deg)'
                : 'translateX(-50%)',
            }}
          />
        </div>,
        document.body,
      )}
    </>
  );
}
