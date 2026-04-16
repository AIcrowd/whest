import { Badge } from '@/components/ui/badge';
import { cn } from '../lib/utils.js';

export default function SymmetryBadge({ value, className }) {
  if (!value) return null;

  return (
    <Badge
      variant="outline"
      className={cn(
        'rounded-full border-gray-200 bg-white font-mono text-xs font-semibold tracking-normal text-gray-700',
        className,
      )}
    >
      {value}
    </Badge>
  );
}
