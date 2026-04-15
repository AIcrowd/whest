import { useRef, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import Latex from './Latex.jsx';
import { getCasePresentation } from './casePresentation.js';
import { cn } from '../lib/utils.js';

const TOOLTIP_WIDTH = 280;
const TOOLTIP_HEIGHT = 128;
const VIEWPORT_PADDING = 16;
const TOOLTIP_OFFSET = 8;

const CASE_COLORS = {
  A: { bg: 'rgba(74, 124, 255, 0.12)', text: '#4A7CFF', border: 'rgba(74, 124, 255, 0.35)' },
  B: { bg: 'rgba(148, 163, 184, 0.12)', text: '#64748B', border: 'rgba(148, 163, 184, 0.35)' },
  C: { bg: 'rgba(250, 158, 51, 0.12)', text: '#D97706', border: 'rgba(250, 158, 51, 0.35)' },
  D: { bg: 'rgba(35, 183, 97, 0.12)', text: '#15803D', border: 'rgba(35, 183, 97, 0.35)' },
  E: { bg: 'rgba(240, 82, 77, 0.12)', text: '#DC2626', border: 'rgba(240, 82, 77, 0.35)' },
};

function getBadgeClasses(variant, size) {
  if (variant === 'compact') {
    return size === 'xs'
      ? 'h-4 w-4 justify-center rounded text-[9px] font-bold'
      : 'h-[18px] w-[18px] justify-center rounded text-[10px] font-bold';
  }

  return size === 'xs'
    ? 'rounded-full px-2 py-0.5 text-[10px] font-semibold'
    : 'rounded-full px-2.5 py-0.5 text-xs font-semibold';
}

export default function CaseBadge({
  caseType,
  size = 'sm',
  variant = 'pill',
  interactive = true,
  className,
}) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const ref = useRef(null);

  const presentation = getCasePresentation(caseType);
  const colors = CASE_COLORS[caseType] ?? CASE_COLORS.A;
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
    setShowTooltip(true);
  };

  const label = variant === 'compact' ? presentation.shortLabel : presentation.label;

  return (
    <>
      <Badge
        ref={ref}
        variant="outline"
        className={cn(
          'inline-flex shrink-0 items-center border font-mono',
          getBadgeClasses(variant, size),
          tooltip && 'cursor-help',
          className,
        )}
        style={{
          backgroundColor: colors.bg,
          color: colors.text,
          borderColor: colors.border,
        }}
        aria-label={presentation.label}
        title={!tooltip ? presentation.label : undefined}
        onMouseEnter={tooltip ? handleEnter : undefined}
        onMouseLeave={tooltip ? () => setShowTooltip(false) : undefined}
      >
        {label}
      </Badge>

      {showTooltip && tooltip && (
        <div
          className="fixed z-[9999] w-72 rounded-lg bg-gray-900 px-3.5 py-3 text-white shadow-2xl"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: tooltipPos.flipped ? 'translateX(-50%)' : 'translateX(-50%) translateY(-100%)',
          }}
        >
          <div className="mb-1 text-xs font-bold">{tooltip.title}</div>
          <div className="text-[11px] leading-relaxed text-gray-300">{tooltip.body}</div>
          {tooltip.latex && (
            <div className="mt-2 border-t border-gray-700 pt-2 text-center">
              <Latex math={tooltip.latex} />
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
        </div>
      )}
    </>
  );
}
