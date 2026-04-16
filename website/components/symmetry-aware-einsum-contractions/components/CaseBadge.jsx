import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Badge } from '@/components/ui/badge';
import Latex from './Latex.jsx';
import GlossaryList from './GlossaryList.jsx';
import { getRegimePresentation } from './regimePresentation.js';
import { cn } from '../lib/utils.js';

const TOOLTIP_WIDTH = 460;
const TOOLTIP_HEIGHT = 340;
const VIEWPORT_PADDING = 16;
const TOOLTIP_OFFSET = 8;

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

function colorsFor(baseColor) {
  return {
    bg: mixWithWhite(baseColor, 0.88),
    text: baseColor,
    border: mixWithWhite(baseColor, 0.55),
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
  caseType,
  regimeId,
  size = 'sm',
  variant = 'pill',
  interactive = true,
  active = false,
  className,
}) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const ref = useRef(null);
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

  // Prefer explicit regimeId; fall back to legacy caseType for callers not yet migrated.
  const id = regimeId ?? caseType;
  const presentation = getRegimePresentation(id);
  const colors = colorsFor(presentation.color ?? '#94A3B8');
  const tooltip = interactive ? presentation.tooltip : null;

  const handleEnter = () => {
    if (!tooltip || !ref.current) return;

    const rect = ref.current.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    let x = rect.left + rect.width / 2;
    x = Math.max(TOOLTIP_WIDTH / 2 + TOOLTIP_OFFSET, Math.min(x, vw - TOOLTIP_WIDTH / 2 - VIEWPORT_PADDING));

    let y = rect.top - TOOLTIP_OFFSET;
    let flipped = false;
    if (y - TOOLTIP_HEIGHT < TOOLTIP_OFFSET) {
      y = rect.bottom + TOOLTIP_OFFSET;
      flipped = true;
    }

    setTooltipPos({ x, y, flipped });
    broadcastTooltipOpen(ownerIdRef.current);
    setShowTooltip(true);
  };

  // Defensive dismissal: pointerleave isn't always guaranteed to fire (scroll,
  // focus change, programmatic events, touch). Close the tooltip whenever the
  // user interacts elsewhere so it can't get stuck floating on the page.
  useEffect(() => {
    if (!showTooltip) return undefined;

    const dismiss = () => setShowTooltip(false);
    const dismissOnEscape = (event) => {
      if (event.key === 'Escape') setShowTooltip(false);
    };
    const dismissIfOutside = (event) => {
      if (!ref.current) return;
      if (event.target instanceof Node && ref.current.contains(event.target)) return;
      setShowTooltip(false);
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
  }, [showTooltip]);

  const label = variant === 'compact' ? presentation.shortLabel : presentation.label;

  return (
    <>
      <span
        onPointerEnter={tooltip ? handleEnter : undefined}
        onPointerLeave={tooltip ? () => setShowTooltip(false) : undefined}
      >
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
              ? `0 0 0 4px ${colors.bg}, 0 0 0 5px ${colors.border}`
              : undefined,
          }}
          aria-label={presentation.label}
        >
          {label}
        </Badge>
      </span>

      {showTooltip && tooltip && typeof document !== 'undefined' && createPortal(
        <div
          className="pointer-events-none fixed z-[9999] w-[460px] max-w-[calc(100vw-2rem)] rounded-lg bg-gray-900 px-4 py-3.5 text-white shadow-2xl"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: tooltipPos.flipped ? 'translateX(-50%)' : 'translateX(-50%) translateY(-100%)',
          }}
        >
          <div className="mb-1 whitespace-normal break-words text-sm font-semibold leading-6">{tooltip.title}</div>
          {tooltip.whenText && (
            <div className="mb-2 text-[11px] uppercase tracking-wider text-gray-400">
              When: {tooltip.whenText}
            </div>
          )}
          <div className="max-w-full whitespace-normal break-words text-sm leading-6 text-gray-300">
            {tooltip.body}
          </div>
          {tooltip.latex && (
            <div className="mt-3 overflow-x-auto border-t border-gray-700 pt-3 text-sm text-gray-100">
              <div className="min-w-0">
                <Latex math={tooltip.latex} display />
              </div>
            </div>
          )}
          {tooltip.glossary && (
            <div className="mt-3 whitespace-normal break-words border-t border-gray-700 pt-3 text-xs leading-relaxed text-gray-300">
              <div className="mb-1.5 text-[10px] uppercase tracking-wider text-gray-500">Where</div>
              <GlossaryList entries={tooltip.glossary} />
            </div>
          )}
          <div
            className={cn(
              'absolute left-1/2 h-1.5 w-3 bg-gray-900',
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
