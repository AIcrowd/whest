import { Fragment } from 'react';
import { Badge } from '@/components/ui/badge';
import { notationColor } from '../lib/notationSystem.js';
import { cn } from '../lib/utils.js';

const COLOR_V = notationColor('v_free');

function ColoredGroupTail({ text }) {
  const chars = Array.from(String(text ?? ''));
  return chars.map((ch, idx) => {
    if (/[A-Za-z]/.test(ch)) {
      return (
        <span key={idx} style={{ color: COLOR_V, fontWeight: 600 }}>
          {ch}
        </span>
      );
    }
    return <Fragment key={idx}>{ch}</Fragment>;
  });
}

function SymmetryBadgeText({ value }) {
  const text = String(value ?? '').trim();
  if (!text) return null;

  const shorthandMatch = text.match(/^([A-Z]\d+)(\{.*\})?$/);
  if (shorthandMatch) {
    const [, head, tail = ''] = shorthandMatch;
    return (
      <>
        <span className="font-bold text-black">{head}</span>
        {tail ? <ColoredGroupTail text={tail} /> : null}
      </>
    );
  }

  const permGroupMatch = text.match(/^(PermGroup)(.*)$/);
  if (permGroupMatch) {
    const [, head, tail = ''] = permGroupMatch;
    return (
      <>
        <span className="font-semibold text-black">{head}</span>
        {tail ? <ColoredGroupTail text={tail} /> : null}
      </>
    );
  }

  const generatedCyclesMatch = text.match(/^(\u27e8)(.*)(\u27e9)$/);
  if (generatedCyclesMatch) {
    const [, leftBracket, middle = '', rightBracket] = generatedCyclesMatch;
    return (
      <>
        <span className="font-semibold text-black">{leftBracket}</span>
        {middle ? <ColoredGroupTail text={middle} /> : null}
        <span className="font-semibold text-black">{rightBracket}</span>
      </>
    );
  }

  return <span className="font-semibold text-gray-700">{text}</span>;
}

export default function SymmetryBadge({ value, className }) {
  if (!value) return null;

  return (
    <Badge
      variant="outline"
      className={cn(
        'inline-flex h-6 items-center rounded-full border-black/70 bg-white px-2.5 font-mono text-xs tracking-normal text-gray-900',
        className,
      )}
    >
      <SymmetryBadgeText value={value} />
    </Badge>
  );
}
