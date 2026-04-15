import { Badge } from '@/components/ui/badge';
import { cn } from '../lib/utils.js';

export const ROLE_BADGE_CLASSNAMES = {
  v: {
    base: 'mx-1 inline-flex rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 align-middle font-mono text-[11px] font-semibold text-sky-700',
    interactive: 'hover:bg-sky-50',
  },
  w: {
    base: 'mx-1 inline-flex rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 align-middle font-mono text-[11px] font-semibold text-slate-700',
    interactive: 'hover:bg-slate-100',
  },
};

export default function RoleBadge({ role, as = 'span', interactive = false, className, children }) {
  const roleClasses = ROLE_BADGE_CLASSNAMES[role] ?? ROLE_BADGE_CLASSNAMES.v;
  const badgeClassName = cn(roleClasses.base, interactive && roleClasses.interactive, className);

  if (as === 'badge') {
    return (
      <Badge variant="outline" className={badgeClassName}>
        {children}
      </Badge>
    );
  }

  return <span className={badgeClassName}>{children}</span>;
}
